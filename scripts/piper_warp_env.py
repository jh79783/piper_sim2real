"""GPU-native batched Piper pick-and-place environment.

The physical model is compiled once with the ordinary MuJoCo Python API and
then uploaded to MuJoCo-Warp.  All rollout state, contact solving, actuator
stepping, rewards, observations, and reset masks live on CUDA.  CPU MuJoCo is
used only for reset-time IK templates and the optional four-frame preview.

This module intentionally has no Gym/SB3 dependency.  The small API mirrors
the RSL-RL vector environment contract used by ``train_piper_rsl.py``.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterable
import warnings

import numpy as np

from scripts.piper_training_config import PiperTrainingConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PiperWarpEnv:
    """Batched Piper task backed by MuJoCo-Warp on a CUDA device.

    Parameters
    ----------
    num_envs:
        Number of independent Warp worlds.
    device:
        Explicit CUDA device.  CPU execution is rejected deliberately because
        this class is the GPU training backend, not the CPU regression env.
    seed:
        Seed for deterministic cube/goal layout sampling.
    start_mode:
        ``curriculum`` samples above-cube/home starts from the current stage;
        ``above_cube`` and ``home`` force one reset pose for every world.
    """

    # The actor emits six normalized incremental arm-target commands and one
    # normalized incremental gripper command. The policy no longer runs
    # Cartesian IK inside a rollout step; CPU IK is used only to construct
    # reset states.
    num_actions = 7
    action_semantics = "incremental_joint_targets_v1"
    action_contract = "raw_si_incremental_joint_v1"
    # Reward-v3's placement latch is retained, followed by the previous raw
    # action.  The latter keeps the action-rate term Markov under slew limits.
    num_observations = 63
    reward_version = 3
    recontact_penalty = 0.2
    physics_timestep = 0.002
    control_hz = 50
    frame_skip = 10
    max_episode_length = 600
    gripper_scale = 0.004
    joint_target_rate = 0.035
    settle_steps = 16
    shaping_gamma = float(np.sqrt(0.99))
    time_penalty = 0.005
    cube_half_size = 0.02
    goal_half_size_np = np.array([0.06, 0.05], dtype=np.float32)

    # Named slices are part of the environment API.  They keep actor sensor
    # history and downstream callers aligned when fields are rearranged.
    OBS_ARM_Q = slice(0, 6)
    OBS_ARM_DQ = slice(6, 12)
    OBS_FINGER_Q = slice(12, 14)
    OBS_GRIPPER_TARGET = slice(14, 15)
    OBS_FINGER_DQ = slice(15, 17)
    OBS_TCP = slice(17, 20)
    OBS_CUBE = slice(20, 23)
    OBS_CUBE_QUAT = slice(23, 27)
    OBS_CUBE_DQ = slice(27, 33)
    OBS_CUBE_TCP = slice(33, 36)
    OBS_GOAL = slice(36, 39)
    OBS_GOAL_CUBE = slice(39, 42)
    OBS_GOAL_HALF = slice(42, 44)
    OBS_CONTACTS = slice(44, 46)
    OBS_LIFTED = slice(46, 47)
    OBS_STABLE = slice(47, 48)
    OBS_TIME = slice(48, 49)
    OBS_ARM_CTRL = slice(49, 55)
    OBS_PLACED = slice(55, 56)
    OBS_PREVIOUS_ACTION = slice(56, 63)
    # Sensor columns are delayed/noised for the actor.  Commands, goals and
    # progress time are overlaid from the current world on every read.
    SENSOR_OBS_SLICES = (
        OBS_ARM_Q,
        OBS_ARM_DQ,
        OBS_FINGER_Q,
        OBS_FINGER_DQ,
        OBS_TCP,
        OBS_CUBE,
        OBS_CUBE_QUAT,
        OBS_CUBE_DQ,
        OBS_CUBE_TCP,
        OBS_CONTACTS,
        OBS_LIFTED,
        OBS_STABLE,
        OBS_PLACED,
    )
    CURRENT_OBS_SLICES = (
        OBS_GRIPPER_TARGET,
        OBS_GOAL,
        OBS_GOAL_HALF,
        OBS_TIME,
        OBS_ARM_CTRL,
        OBS_PREVIOUS_ACTION,
    )
    # This field uses the current command goal and the delayed/noisy cube;
    # callers should verify it against those two columns rather than compare
    # it to the clean critic's goal-minus-cube value.
    DERIVED_CURRENT_OBS_SLICES = (OBS_GOAL_CUBE,)

    def __init__(
        self,
        num_envs: int = 128,
        device: str = "cuda",
        seed: int = 0,
        start_mode: str = "curriculum",
        training_config: PiperTrainingConfig | dict | None = None,
    ) -> None:
        if int(num_envs) < 1:
            raise ValueError("num_envs must be positive")
        if start_mode not in ("curriculum", "above_cube", "home"):
            raise ValueError("start_mode must be curriculum, above_cube, or home")
        if training_config is None:
            training_config = PiperTrainingConfig()
        elif isinstance(training_config, dict):
            training_config = PiperTrainingConfig.from_dict(training_config)
        if not isinstance(training_config, PiperTrainingConfig):
            raise TypeError("training_config must be PiperTrainingConfig or a mapping")
        self.training_config = training_config
        # Copy timing/control values from the serializable config so a
        # restored run and a fresh run execute the same contract.
        self.physics_timestep = float(self.training_config.physics_timestep)
        self.control_hz = int(self.training_config.control_hz)
        self.frame_skip = int(self.training_config.decimation)
        self.max_episode_length = int(self.training_config.episode_steps)
        self.settle_steps = int(self.training_config.success_steps)
        self.joint_target_rate = float(self.training_config.joint_target_rate)
        self.gripper_scale = float(self.training_config.gripper_target_rate)
        self.time_penalty = float(self.training_config.time_penalty)
        if not str(device).startswith("cuda"):
            raise ValueError("PiperWarpEnv requires an explicit CUDA device")

        # Keep heavyweight GPU imports out of module import/help paths.
        try:
            import mujoco
            import mujoco_warp as mjw
            import torch
            import warp as wp
        except ImportError as exc:  # pragma: no cover - exercised in host help paths
            raise RuntimeError(
                "PiperWarpEnv requires mujoco, mujoco_warp, torch, and warp in the GPU image"
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("PiperWarpEnv requires CUDA; CPU fallback is intentionally disabled")

        self._mujoco = mujoco
        self._mjw = mjw
        self._torch = torch
        self._wp = wp
        # MuJoCo-Warp imports Warp but does not necessarily initialize its
        # device registry before Torch interop is requested.
        wp.init()
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        if self.device.index is None:
            self.device = torch.device("cuda:0")
        torch_device = self.device
        warp_device = str(torch_device)
        torch.cuda.set_device(torch_device)
        # Keep reset randomization independent of Torch's global policy RNG.
        self._torch_rng = torch.Generator(device=warp_device)
        self._torch_rng.manual_seed(int(seed))
        self.start_mode = start_mode
        self._manual_start_mode = start_mode if start_mode != "curriculum" else None
        self.training_steps = 0
        self._stage = self.training_config.stage_at(self.training_steps)
        self._rng = np.random.default_rng(seed)
        self.seed_value = int(seed)

        # ``build_model`` is shared with the validated CPU contact scene.  It
        # only compiles the model and never steps a CPU world during rollout.
        from scripts.piper_pick_place_env import PiperPickPlaceEnv, build_model

        self.model = build_model()
        # The CPU helper only supplies reset-time IK.  It does not participate
        # in GPU rollout stepping and must receive one of its two concrete
        # reset modes even when the GPU env uses curriculum starts.
        self._ik_env = PiperPickPlaceEnv(start_mode="above_cube")
        self._ik_home = np.array([0.0, 1.57, -1.3485, 0.0, 0.0, 0.0], dtype=np.float64)
        # Use Warp's native blocking stream: unlike a wrapped Torch stream it
        # supports CUDA graph capture in Warp 1.12.  It is a distinct CUDA
        # stream, so explicit Torch↔Warp wait events are installed below.
        self._warp_stream = wp.get_stream(warp_device)
        self._torch_warp_stream = wp.stream_to_torch(self._warp_stream)
        self.cfg = {
            "task": "piper_pick_place",
            "physics": "mujoco_warp",
            "device": warp_device,
            "num_envs": self.num_envs,
            "num_obs": self.num_observations,
            "num_actions": self.num_actions,
            "action_semantics": self.action_semantics,
            "action_contract": self.action_contract,
            "observation_version": "policy_critic_sensor_history_v1",
            "decimation": self.frame_skip,
            "episode_length": self.max_episode_length,
            "physics_timestep": self.physics_timestep,
            "control_hz": self.control_hz,
            "control_period": 1.0 / self.control_hz,
            "episode_seconds": self.max_episode_length / self.control_hz,
            "success_steps": self.settle_steps,
            "success_seconds": self.settle_steps / self.control_hz,
            "shaping_gamma": self.shaping_gamma,
            "joint_target_rate": self.joint_target_rate,
            "gripper_target_rate": self.gripper_scale,
            "time_penalty": self.time_penalty,
            "reward_version": self.reward_version,
            "recontact_penalty": self.recontact_penalty,
            "training_config": self.training_config.to_dict(),
            "training_steps": self.training_steps,
            "curriculum_stage": self._stage_at_runtime(),
        }

        self.tcp_id = int(self.model.site("grasp_tcp").id)
        self.cube_geom = int(self.model.geom("cube_geom").id)
        self.table_geom = int(self.model.geom("table").id)
        self.cube_body = int(self.model.body("cube").id)
        self.goal_body = int(self.model.body("goal_area").id)
        self.goal_mocap_id = int(self.model.body_mocapid[self.goal_body])
        self.cube_qadr = int(self.model.jnt_qposadr[self.model.joint("cube_free").id])
        self.cube_vadr = int(self.model.jnt_dofadr[self.model.joint("cube_free").id])
        self.arm_qadr = np.asarray(
            [self.model.joint(f"joint{i}").qposadr[0] for i in range(1, 7)], dtype=np.int64
        )
        self.arm_dofs = np.asarray(
            [self.model.joint(f"joint{i}").dofadr[0] for i in range(1, 7)], dtype=np.int64
        )
        self.arm_actuators = np.asarray(
            [self.model.actuator(f"joint{i}").id for i in range(1, 7)], dtype=np.int64
        )
        self.gripper_actuator = int(self.model.actuator("gripper").id)
        self.finger_qadr = int(self.model.joint("joint7").qposadr[0])
        self.other_finger_qadr = int(self.model.joint("joint8").qposadr[0])
        self.finger_dofadr = int(self.model.joint("joint7").dofadr[0])
        self.other_finger_dofadr = int(self.model.joint("joint8").dofadr[0])
        self.joint_ranges = np.asarray(
            self.model.jnt_range[
                [self.model.joint(f"joint{i}").id for i in range(1, 7)]
            ],
            dtype=np.float32,
        )
        self._moving_body_ids_np = np.asarray(
            [self.model.body(f"link{i}").id for i in range(1, 9)] + [self.cube_body],
            dtype=np.int64,
        )
        self._finger_geom_ids_np = np.flatnonzero(
            np.isin(self.model.geom_bodyid, [
                int(self.model.body("link7").id),
                int(self.model.body("link8").id),
            ])
            & (self.model.geom_contype != 0)
        ).astype(np.int64)
        self._friction_geom_ids_np = np.unique(
            np.concatenate((
                np.asarray([self.table_geom, self.cube_geom], dtype=np.int64),
                self._finger_geom_ids_np,
            ))
        )

        # Immutable index tensors keep all per-step gathers on CUDA.
        self._arm_qadr = torch.as_tensor(self.arm_qadr, dtype=torch.long, device=torch_device)
        self._arm_dofs = torch.as_tensor(self.arm_dofs, dtype=torch.long, device=torch_device)
        self._arm_actuators = torch.as_tensor(self.arm_actuators, dtype=torch.long, device=torch_device)
        self._joint_low = torch.as_tensor(self.joint_ranges[:, 0], device=torch_device)
        self._joint_high = torch.as_tensor(self.joint_ranges[:, 1], device=torch_device)
        self._joint_mid = 0.5 * (self._joint_low + self._joint_high)
        self._joint_half_range = 0.5 * (self._joint_high - self._joint_low)
        self._action_target_low = self._joint_low + 0.002
        self._action_target_high = self._joint_high - 0.002
        self._goal_half_size = torch.as_tensor(self.goal_half_size_np, device=torch_device)
        self._all_worlds = torch.arange(self.num_envs, dtype=torch.long, device=torch_device)
        self._contact_indices = torch.arange(0, 0, dtype=torch.long, device=torch_device)
        self._arm_ctrl = torch.zeros((self.num_envs, 6), dtype=torch.float32, device=torch_device)

        # Build the only device-side physics object.  Contacts are packed in a
        # global heterogeneous array, so naconmax is intentionally generous;
        # world IDs are interpreted from the packed entries, never reshaped.
        with wp.ScopedDevice(warp_device):
            self._warp_model = mjw.put_model(self.model)
            self._warp_data = mjw.make_data(
                self.model,
                nworld=self.num_envs,
                nconmax=256,
                nccdmax=256,
                njmax=1024,
                njmax_nnz=1024 * self.model.nv,
            )
            # MuJoCo-Warp initially stores world-varying model fields with a
            # single leading row.  Expand them before any reset or CUDA graph
            # capture and retain the allocations for the environment lifetime.
            self._expand_world_model_fields()

        self._qpos = wp.to_torch(self._warp_data.qpos)
        self._qvel = wp.to_torch(self._warp_data.qvel)
        self._ctrl = wp.to_torch(self._warp_data.ctrl)
        self._site_xpos = wp.to_torch(self._warp_data.site_xpos)
        self._site_xmat = wp.to_torch(self._warp_data.site_xmat)
        self._xpos = wp.to_torch(self._warp_data.xpos)
        self._geom_xmat = wp.to_torch(self._warp_data.geom_xmat)
        self._nacon = wp.to_torch(self._warp_data.nacon)
        self._contact_geom = wp.to_torch(self._warp_data.contact.geom)
        self._contact_worldid = wp.to_torch(self._warp_data.contact.worldid)
        self._contact_dist = wp.to_torch(self._warp_data.contact.dist)
        self._mocap_pos = wp.to_torch(self._warp_data.mocap_pos)
        self._bind_model_parameter_views()
        self._finger_geom_tensors = []
        for body_name in ("link7", "link8"):
            body_id = int(self.model.body(body_name).id)
            self._finger_geom_tensors.append(
                torch.as_tensor(
                    np.flatnonzero(
                        (self.model.geom_bodyid == body_id)
                        & (self.model.geom_contype != 0)
                    ),
                    dtype=torch.long,
                    device=torch_device,
                )
            )
        # Keep all collidable robot geometry in the base_link subtree.  This
        # includes arm links and both fingers while excluding task geometry
        # (table/floor/cube/goal), so recontact is not finger-only.
        robot_root = int(self.model.body("base_link").id)
        robot_bodies = []
        for body_id in range(self.model.nbody):
            current = body_id
            while current != 0:
                if current == robot_root:
                    robot_bodies.append(body_id)
                    break
                current = int(self.model.body_parentid[current])
        robot_geom_ids = np.flatnonzero(np.isin(self.model.geom_bodyid, robot_bodies))
        self._robot_geom_tensor = torch.as_tensor(
            robot_geom_ids, dtype=torch.long, device=torch_device
        )

        self.goal_xy = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=torch_device)
        self.goal_position = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=torch_device)
        self.gripper_target = torch.full((self.num_envs,), 0.035, dtype=torch.float32, device=torch_device)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.int32, device=torch_device)
        self.has_lifted = torch.zeros(self.num_envs, dtype=torch.bool, device=torch_device)
        self.has_placed = torch.zeros(self.num_envs, dtype=torch.bool, device=torch_device)
        self.recontacted = torch.zeros(self.num_envs, dtype=torch.bool, device=torch_device)
        self.recontact_penalty_total = torch.zeros(
            self.num_envs, dtype=torch.float32, device=torch_device
        )
        self.stable_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=torch_device)
        self.last_ik_error = torch.zeros(self.num_envs, dtype=torch.float32, device=torch_device)
        self._previous_potential = torch.zeros(self.num_envs, dtype=torch.float32, device=torch_device)
        self._last_success = torch.zeros(self.num_envs, dtype=torch.bool, device=torch_device)
        self._last_actions = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float32, device=torch_device)
        self._last_action_delta = torch.zeros_like(self._last_actions)
        self._joint_zero_offset = torch.zeros((self.num_envs, 6), dtype=torch.float32, device=torch_device)
        self._start_home = torch.zeros(self.num_envs, dtype=torch.bool, device=torch_device)
        self._sensor_delay = torch.zeros(self.num_envs, dtype=torch.long, device=torch_device)
        self._sensor_history = torch.zeros(
            (self.num_envs, self.training_config.max_sensor_delay_steps + 1, self.num_observations),
            dtype=torch.float32,
            device=torch_device,
        )
        self._episode_mass_scale = torch.ones(
            (self.num_envs, len(self._moving_body_ids_np)), dtype=torch.float32, device=torch_device
        )
        self._episode_inertia_scale = torch.ones_like(self._episode_mass_scale)
        self._episode_com_offset = torch.zeros(
            (self.num_envs, len(self._moving_body_ids_np), 3), dtype=torch.float32, device=torch_device
        )
        self._episode_friction_scale = torch.ones(
            (self.num_envs, len(self._friction_geom_ids_np)), dtype=torch.float32, device=torch_device
        )
        self._step_graph = None
        self._step_graph_attempted = False
        self._step_graph_error = None
        self._closed = False

        self._render_data = mujoco.MjData(self.model)
        self._renderer = None
        self._render_camera = mujoco.MjvCamera()
        self._render_camera.lookat[:] = [0.28, 0.0, 0.13]
        self._render_camera.distance = 1.05
        self._render_camera.azimuth = 140
        self._render_camera.elevation = -28
        self._render_option = mujoco.MjvOption()
        self._render_option.geomgroup[3] = 0

        # Initial reset uploads the first batch and performs one device forward
        # pass.  It does not run CPU physics.
        self.reset(seed=seed)

    # ------------------------------------------------------------------
    # Device array and reset helpers
    # ------------------------------------------------------------------
    @contextmanager
    def _stream_scope(self):
        """Run Warp work with explicit dependencies on Torch's current stream."""

        torch_stream = self._torch.cuda.current_stream(self.device)
        self._torch_warp_stream.wait_stream(torch_stream)
        with self._wp.ScopedStream(self._warp_stream, sync_enter=False, sync_exit=False):
            try:
                yield
            finally:
                # Publish Warp writes even when a kernel/capture raises, so
                # subsequent Torch cleanup cannot race the device stream.
                torch_stream.wait_stream(self._torch_warp_stream)

    # These fields are the world-varying arrays consumed by MuJoCo-Warp's
    # dynamics and by ``set_const``.  Other model arrays are immutable topology
    # or geometry data and intentionally remain one-row/shared.
    _WORLD_MODEL_FIELDS = (
        "body_mass",
        "body_inertia",
        "body_ipos",
        "geom_friction",
        "body_subtreemass",
        "body_invweight0",
        "dof_invweight0",
        "tendon_length0",
        "tendon_invweight0",
        "cam_pos0",
        "cam_poscom0",
        "cam_mat0",
        "light_pos0",
        "light_poscom0",
        "light_dir0",
        "actuator_acc0",
        "actuator_biasprm",
    )

    def _expand_world_model_fields(self):
        """Expand MuJoCo-Warp's leading singleton rows before graph capture."""

        torch = self._torch
        wp = self._wp
        self._expanded_model_fields = {}

        def expand(owner, name):
            field = getattr(owner, name, None)
            if field is None or not hasattr(field, "shape"):
                return
            shape = tuple(field.shape)
            if not shape or shape[0] != 1 or self.num_envs == 1:
                return
            value = wp.to_torch(field)
            repeat = (self.num_envs,) + (1,) * (value.ndim - 1)
            expanded = value.repeat(repeat)
            replacement = wp.from_torch(
                expanded,
                dtype=field.dtype,
                requires_grad=False,
            )
            setattr(owner, name, replacement)
            self._expanded_model_fields[name] = replacement

        for name in self._WORLD_MODEL_FIELDS:
            expand(self._warp_model, name)
        # ``meaninertia`` is nested under Model.stat rather than Model itself.
        expand(self._warp_model.stat, "meaninertia")

    def _bind_model_parameter_views(self):
        """Cache CUDA views and immutable CPU baselines for reset randomization."""

        torch = self._torch
        wp = self._wp
        self._warp_body_mass = wp.to_torch(self._warp_model.body_mass)
        self._warp_body_inertia = wp.to_torch(self._warp_model.body_inertia)
        self._warp_body_ipos = wp.to_torch(self._warp_model.body_ipos)
        self._warp_geom_friction = wp.to_torch(self._warp_model.geom_friction)
        self._base_body_mass = torch.as_tensor(
            self.model.body_mass,
            dtype=self._warp_body_mass.dtype,
            device=self.device,
        ).clone()
        self._base_body_inertia = torch.as_tensor(
            self.model.body_inertia,
            dtype=self._warp_body_inertia.dtype,
            device=self.device,
        ).clone()
        self._base_body_ipos = torch.as_tensor(
            self.model.body_ipos,
            dtype=self._warp_body_ipos.dtype,
            device=self.device,
        ).clone()
        self._base_geom_friction = torch.as_tensor(
            self.model.geom_friction,
            dtype=self._warp_geom_friction.dtype,
            device=self.device,
        ).clone()
        # Public aliases make the per-world fields easy to inspect without
        # copying them back to CPU.  They remain stable for the env lifetime.
        self.body_mass = self._warp_body_mass
        self.body_inertia = self._warp_body_inertia
        self.body_ipos = self._warp_body_ipos
        self.geom_friction = self._warp_geom_friction

    def _stage_at_runtime(self):
        stage = self._stage
        return {
            "step": int(stage.step),
            "randomization_scale": float(stage.randomization_scale),
            "layout_scale": float(stage.layout_scale),
            "home_probability": float(stage.home_probability),
            "action_rate_weight": float(stage.action_rate_weight),
        }

    @property
    def curriculum_stage(self):
        return self._stage_at_runtime()

    def set_training_steps(self, count: int):
        """Restore/update cumulative vector-env steps for curriculum lookup."""

        count = int(count)
        if count < 0:
            raise ValueError("training step count must be nonnegative")
        self.training_steps = count
        self._stage = self.training_config.stage_at(count)
        if hasattr(self, "cfg"):
            self.cfg["training_steps"] = self.training_steps
            self.cfg["curriculum_stage"] = self._stage_at_runtime()
        return self._stage_at_runtime()

    def _set_episode_randomization(self, worlds, stage):
        """Write fresh per-world mass/inertia/CoM/friction values on reset."""

        torch = self._torch
        count = int(worlds.numel())
        scale = float(stage.randomization_scale) if self.training_config.domain_randomization else 0.0
        body_ids = torch.as_tensor(self._moving_body_ids_np, dtype=torch.long, device=self.device)
        link_count = len(self._moving_body_ids_np) - 1
        # One factor per body keeps each inertia diagonal a valid uniformly
        # scaled tensor and prevents invalid triangle inequalities.
        mass_fraction = torch.full((count, link_count + 1), self.training_config.link_mass_fraction,
                                   device=self.device, dtype=self._warp_body_mass.dtype)
        mass_fraction[:, -1] = self.training_config.cube_mass_fraction
        mass_scale = 1.0 + (
            2.0 * torch.rand(
                mass_fraction.shape, device=self.device, dtype=mass_fraction.dtype,
                generator=self._torch_rng,
            ) - 1.0
        ) * mass_fraction * scale
        inertia_scale = 1.0 + (
            2.0 * torch.rand(
                mass_fraction.shape, device=self.device, dtype=mass_fraction.dtype,
                generator=self._torch_rng,
            ) - 1.0
        ) * self.training_config.inertia_fraction * scale
        com_range = torch.full((count, link_count + 1, 3), self.training_config.link_com_range,
                               device=self.device, dtype=self._warp_body_ipos.dtype)
        com_range[:, -1] = self.training_config.cube_com_range
        com_offset = (
            2.0 * torch.rand(
                com_range.shape, device=self.device, dtype=com_range.dtype,
                generator=self._torch_rng,
            ) - 1.0
        ) * com_range * scale

        # Restore each selected world from immutable baselines before applying
        # fresh factors.  This prevents reset-to-reset multiplicative drift.
        for column, body_id in enumerate(body_ids.tolist()):
            self._warp_body_mass[worlds, body_id] = self._base_body_mass[body_id] * mass_scale[:, column]
            self._warp_body_inertia[worlds, body_id] = (
                self._base_body_inertia[body_id] * inertia_scale[:, column, None]
            )
            self._warp_body_ipos[worlds, body_id] = self._base_body_ipos[body_id] + com_offset[:, column]

        geom_ids = torch.as_tensor(self._friction_geom_ids_np, dtype=torch.long, device=self.device)
        friction_base = self._base_geom_friction[geom_ids]
        friction_fraction = self.training_config.friction_fraction
        friction_scale = 1.0 + (
            2.0 * torch.rand(
                (count, geom_ids.numel()), device=self.device,
                dtype=self._warp_geom_friction.dtype, generator=self._torch_rng,
            ) - 1.0
        ) * friction_fraction * scale
        for column, geom_id in enumerate(geom_ids.tolist()):
            self._warp_geom_friction[worlds, geom_id] = friction_base[column] * friction_scale[:, column, None]

        self._episode_mass_scale[worlds] = mass_scale
        self._episode_inertia_scale[worlds] = inertia_scale
        self._episode_com_offset[worlds] = com_offset
        self._episode_friction_scale[worlds] = friction_scale

    def _reset_sensor_history(self, worlds, clean_observation):
        """Reset delay buffers for selected worlds without sampling on reads."""

        if worlds.numel() == 0:
            return
        # History is [world, delay slot, observation].  Slot zero is newest;
        # reset fills every slot so a delayed sensor never leaks a prior
        # episode.  Clones keep returned TensorDict transitions independent.
        self._sensor_history[worlds] = clean_observation[worlds].unsqueeze(1).repeat(
            1, self._sensor_history.shape[1], 1
        )
        self._sensor_delay[worlds] = 0

    def _random_sensor_delay(self, worlds):
        torch = self._torch
        if worlds.numel() == 0:
            return
        if not self.training_config.sensor_noise:
            self._sensor_delay[worlds] = 0
            return
        maximum = int(self.training_config.max_sensor_delay_steps)
        if maximum:
            self._sensor_delay[worlds] = torch.randint(
                0,
                maximum + 1,
                (worlds.numel(),),
                device=self.device,
                dtype=torch.long,
                generator=self._torch_rng,
            )
        else:
            self._sensor_delay[worlds] = 0

    def _capture_step_graph(self) -> None:
        """Capture the fixed 10-physics-step decimation on Warp's native stream.

        MJWarp's state/control arrays are stable for the lifetime of an env,
        so graph replay observes the latest CUDA-written ``data.ctrl`` and
        advances the current qpos/qvel in place.  If a future Warp/MuJoCo
        build rejects capture for a model feature, the backend keeps the same
        GPU physics path and falls back to the ordinary 10-call loop while
        retaining the diagnostic in ``cfg``.
        """

        if self._step_graph_attempted:
            return
        self._step_graph_attempted = True
        try:
            with self._stream_scope():
                self._wp.capture_begin(device=str(self.device))
                try:
                    for _ in range(self.frame_skip):
                        self._mjw.step(self._warp_model, self._warp_data)
                except Exception:
                    # Leave capture mode in a defined state before falling
                    # back to the non-graph GPU path.
                    try:
                        self._wp.capture_end(device=str(self.device))
                    except Exception:
                        pass
                    raise
                self._step_graph = self._wp.capture_end(device=str(self.device))
            self.cfg["cuda_graph_decimation"] = True
        except Exception as exc:  # pragma: no cover - version/model dependent
            self._step_graph_error = f"{type(exc).__name__}: {exc}"
            self.cfg["cuda_graph_decimation"] = False
            self.cfg["cuda_graph_error"] = self._step_graph_error
            warnings.warn(
                "MuJoCo-Warp CUDA graph capture unavailable; using the same GPU physics "
                f"decimation loop ({self._step_graph_error})",
                RuntimeWarning,
                stacklevel=2,
            )

    def _step_physics(self) -> None:
        if not self._step_graph_attempted:
            self._capture_step_graph()
        with self._stream_scope():
            if self._step_graph is not None:
                self._wp.capture_launch(self._step_graph)
            else:
                for _ in range(self.frame_skip):
                    self._mjw.step(self._warp_model, self._warp_data)

    def _sample_layout(self, count: int, stage=None) -> tuple[np.ndarray, np.ndarray]:
        stage = self._stage if stage is None else stage
        # The final table region is wider than the original 120 mm x-span,
        # while the first curriculum stage remains easy and always admits the
        # required 130 mm cube/goal separation.
        center = np.array([0.34, 0.0], dtype=np.float32)
        layout_scale = float(stage.layout_scale)
        half_extent = np.array([0.09, 0.18], dtype=np.float32) * layout_scale
        # Keep the exclusion region proportional to the curriculum layout so
        # the compact first stage remains sampleable for central cube points.
        min_separation = 0.13 * layout_scale
        low = center - half_extent
        high = center + half_extent
        cube = self._rng.uniform(low, high, size=(count, 2)).astype(np.float32)
        goal = self._rng.uniform(low, high, size=(count, 2)).astype(np.float32)
        pending = np.ones(count, dtype=bool)
        for _ in range(10_000):
            if not pending.any():
                return cube, goal
            candidates = self._rng.uniform(
                low, high, size=(int(pending.sum()), 2)
            ).astype(np.float32)
            goal[pending] = candidates
            pending[pending] = np.linalg.norm(candidates - cube[pending], axis=1) <= min_separation
        if not pending.any():
            return cube, goal
        raise RuntimeError("could not sample separated cube and goal positions")

    def _solve_initial_joints(self, cube_xy: np.ndarray, start_home: np.ndarray) -> np.ndarray:
        home_joints = None
        joints = np.empty((cube_xy.shape[0], 6), dtype=np.float32)
        for row, xy, is_home in zip(joints, cube_xy, start_home):
            if is_home:
                if home_joints is None:
                    target = np.array([0.33, 0.0, 0.14], dtype=np.float64)
                    home_joints, residual = self._ik_env.solve_ik(
                        target, self._ik_home, iterations=150
                    )
                    if residual > 0.005:
                        raise RuntimeError(f"home IK failed with residual {residual:.4f} m")
                row[:] = home_joints
                continue
            target = np.array([float(xy[0]), float(xy[1]), 0.10], dtype=np.float64)
            solved, residual = self._ik_env.solve_ik(target, self._ik_home, iterations=150)
            if residual > 0.005:
                raise RuntimeError(f"reset IK failed with residual {residual:.4f} m")
            row[:] = solved
        return joints

    def _reset_worlds(self, worlds: torch.Tensor, *, seed: int | None = None) -> None:
        if worlds.numel() == 0:
            return
        torch = self._torch
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
            self.seed_value = int(seed)
            self._torch_rng.manual_seed(int(seed))
        worlds = worlds.to(device=self.device, dtype=torch.long)
        count = int(worlds.numel())
        stage = self._stage
        cube_xy_np, goal_xy_np = self._sample_layout(count, stage)
        if self._manual_start_mode == "home":
            start_home_np = np.ones(count, dtype=bool)
        elif self._manual_start_mode == "above_cube":
            start_home_np = np.zeros(count, dtype=bool)
        else:
            start_home_np = self._rng.random(count) < float(stage.home_probability)
        joints_np = self._solve_initial_joints(cube_xy_np, start_home_np)
        cube_xy = torch.as_tensor(cube_xy_np, device=self.device)
        goal_xy = torch.as_tensor(goal_xy_np, device=self.device)
        joints = torch.as_tensor(joints_np, device=self.device)
        start_home = torch.as_tensor(start_home_np, dtype=torch.bool, device=self.device)

        # All physical and sensor episode parameters are selected once here.
        # They remain fixed until this world is reset again.
        self._set_episode_randomization(worlds, stage)
        if self.training_config.sensor_noise:
            self._joint_zero_offset[worlds] = (
                2.0 * torch.rand(
                    (count, 6), device=self.device, dtype=torch.float32, generator=self._torch_rng
                ) - 1.0
            ) * float(self.training_config.joint_zero_offset)
        else:
            self._joint_zero_offset[worlds] = 0.0
        self._random_sensor_delay(worlds)

        reset_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        reset_mask[worlds] = True
        with self._stream_scope():
            self._mjw.reset_data(
                self._warp_model,
                self._warp_data,
                reset=self._wp.from_torch(reset_mask, dtype=self._wp.bool, requires_grad=False),
            )
            # reset_data itself is queued on the Warp stream.  Wait for it
            # before writing fresh qpos/ctrl/mocap values from Torch.
            torch.cuda.current_stream(self.device).wait_stream(self._torch_warp_stream)
            qpos = self._qpos
            qvel = self._qvel
            ctrl = self._ctrl
            qvel[worlds] = 0.0
            for col, address in enumerate(self.arm_qadr):
                qpos[worlds, int(address)] = joints[:, col]
            qpos[worlds, self.cube_qadr : self.cube_qadr + 3] = torch.cat(
                (cube_xy, torch.full((worlds.numel(), 1), self.cube_half_size + 0.0005, device=self.device)),
                dim=1,
            )
            qpos[worlds, self.cube_qadr + 3 : self.cube_qadr + 7] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], device=self.device
            )
            qpos[worlds, self.finger_qadr] = 0.035
            qpos[worlds, self.other_finger_qadr] = -0.035
            ctrl[worlds] = 0.0
            ctrl[worlds[:, None], self._arm_actuators[None, :]] = joints
            ctrl[worlds, self.gripper_actuator] = 0.035
            self._mocap_pos[worlds, self.goal_mocap_id] = torch.cat(
                (goal_xy, torch.full((worlds.numel(), 1), 0.001, device=self.device)), dim=1
            )
            # qpos/ctrl/mocap assignments above execute on Torch's current
            # stream even while Warp's stream is scoped.  Publish them before
            # rebuilding constants and launching the dependent forward pass.
            self._torch_warp_stream.wait_stream(torch.cuda.current_stream(self.device))
            # set_const temporarily copies qpos0 into qpos while recomputing
            # mass/inertia-dependent constants, then restores current qpos.
            # The explicit forward below refreshes all state-derived outputs.
            self._mjw.set_const(self._warp_model, self._warp_data)
            self._mjw.forward(self._warp_model, self._warp_data)

        self.goal_xy[worlds] = goal_xy
        self.goal_position[worlds] = torch.cat(
            (goal_xy, torch.full((worlds.numel(), 1), self.cube_half_size, device=self.device)), dim=1
        )
        self.gripper_target[worlds] = 0.035
        self._arm_ctrl[worlds] = joints
        self.episode_length_buf[worlds] = 0
        self.has_lifted[worlds] = False
        self.has_placed[worlds] = False
        self.recontacted[worlds] = False
        self.recontact_penalty_total[worlds] = 0.0
        self.stable_steps[worlds] = 0
        self.last_ik_error[worlds] = 0.0
        self._last_success[worlds] = False
        # Incremental actions are zero-centered: zero holds the reset
        # actuator targets. The action history is therefore zero at every
        # episode boundary, avoiding a reset-dependent command bias.
        self._last_actions[worlds] = 0.0
        self._previous_potential[worlds] = self._potential()[worlds]
        self._start_home[worlds] = start_home

        # Build one actor frame at reset and fill every delay slot.  Calling
        # get_observations repeatedly afterwards only clones these cached
        # values; it never samples noise or advances history.
        clean = self._observation_tensor()
        actor = self._sensor_frame(clean, worlds)
        self._sensor_history[worlds] = actor.unsqueeze(1).repeat(
            1, self._sensor_history.shape[1], 1
        )

    def reset(self, seed: int | None = None):
        """Reset all worlds and return a CUDA TensorDict observation."""

        if self._closed:
            raise RuntimeError("PiperWarpEnv is closed")
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
            self.seed_value = int(seed)
            self._torch_rng.manual_seed(int(seed))
        self._reset_worlds(self._all_worlds, seed=None)
        return self.get_observations()

    # ------------------------------------------------------------------
    # GPU controller, contacts, observations, rewards
    # ------------------------------------------------------------------
    def _tcp_position(self) -> "object":
        return self._site_xpos[:, self.tcp_id]

    def _cube_position(self) -> "object":
        return self._xpos[:, self.cube_body]

    def _finger_contacts(self) -> "object":
        """Return bilateral cube/finger contact flags from packed GPU contacts."""

        torch = self._torch
        nacon = self._nacon.reshape(-1)[0].to(dtype=torch.long)
        slots = torch.arange(self._contact_geom.shape[0], device=self.device)
        active = slots < nacon
        world = self._contact_worldid.clamp(min=0, max=self.num_envs - 1).to(dtype=torch.long)
        geom0 = self._contact_geom[:, 0]
        geom1 = self._contact_geom[:, 1]
        cube = self.cube_geom
        finger7, finger8 = self._finger_geom_tensors
        touched7 = active & (((geom0 == cube) & torch.isin(geom1, finger7)) | ((geom1 == cube) & torch.isin(geom0, finger7)))
        touched8 = active & (((geom0 == cube) & torch.isin(geom1, finger8)) | ((geom1 == cube) & torch.isin(geom0, finger8)))
        active &= self._contact_dist <= 0.001
        touched7 &= active
        touched8 &= active
        # CUDA's scatter-reduce kernels do not support Bool tensors in the
        # pinned Torch build; reduce float flags and convert back to Bool.
        result = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        result[:, 0].scatter_reduce_(0, world, touched7.to(torch.float32), reduce="amax", include_self=False)
        result[:, 1].scatter_reduce_(0, world, touched8.to(torch.float32), reduce="amax", include_self=False)
        return result > 0.5

    def _robot_cube_contacts(self) -> "object":
        """Return per-world contact with any collidable robot geom and cube."""

        torch = self._torch
        nacon = self._nacon.reshape(-1)[0].to(dtype=torch.long)
        slots = torch.arange(self._contact_geom.shape[0], device=self.device)
        active = slots < nacon
        world = self._contact_worldid.clamp(min=0, max=self.num_envs - 1).to(dtype=torch.long)
        geom0 = self._contact_geom[:, 0]
        geom1 = self._contact_geom[:, 1]
        robot0 = torch.isin(geom0, self._robot_geom_tensor)
        robot1 = torch.isin(geom1, self._robot_geom_tensor)
        touched = active & (self._contact_dist <= 0.001)
        touched &= ((geom0 == self.cube_geom) & robot1) | ((geom1 == self.cube_geom) & robot0)
        result = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        result.scatter_reduce_(0, world, touched.to(torch.float32), reduce="amax", include_self=False)
        return result > 0.5

    def _check_physics_state(self) -> None:
        """Fail loudly on solver NaNs or packed-contact overflow."""

        torch = self._torch
        if not torch.isfinite(self._qpos).all().item() or not torch.isfinite(self._qvel).all().item():
            raise FloatingPointError("MuJoCo-Warp produced non-finite qpos/qvel")
        if self._nacon.reshape(-1)[0].item() > self._contact_geom.shape[0]:
            raise RuntimeError(
                "MuJoCo-Warp contact buffer overflow; increase nconmax/nccdmax before training"
            )

    def _placement_state(self):
        torch = self._torch
        cube = self._cube_position()
        rotation = self._geom_xmat[:, self.cube_geom]
        extent = torch.abs(rotation) @ torch.full((3,), self.cube_half_size, device=self.device)
        inside = torch.all(
            torch.abs(cube[:, :2] - self.goal_position[:, :2]) + extent[:, :2]
            <= self._goal_half_size - 0.002,
            dim=1,
        )
        on_table = torch.abs(cube[:, 2] - extent[:, 2]) < 0.006
        velocity = self._qvel[:, self.cube_vadr : self.cube_vadr + 6]
        still = (torch.linalg.vector_norm(velocity[:, :3], dim=1) < 0.03) & (
            torch.linalg.vector_norm(velocity[:, 3:], dim=1) < 0.5
        )
        finger = self._finger_contacts()
        released = (self._qpos[:, self.finger_qadr] > 0.028) & (~finger.any(dim=1))
        return inside, on_table, still, released

    def _potential(self):
        torch = self._torch
        cube = self._cube_position()
        tcp = self._tcp_position()
        finger = self._finger_contacts()
        grasped = finger.all(dim=1)
        reach = 1.0 - torch.tanh(10.0 * torch.linalg.vector_norm(tcp - cube, dim=1))
        height = torch.clamp((cube[:, 2] - self.cube_half_size) / 0.06, 0.0, 1.0)
        pre = reach + grasped.to(torch.float32) + 2.0 * height
        inside, on_table, _, _ = self._placement_state()
        xy_score = 1.0 - torch.tanh(
            8.0 * torch.linalg.vector_norm(cube[:, :2] - self.goal_position[:, :2], dim=1)
        )
        place_score = 1.0 - torch.tanh(
            8.0 * torch.linalg.vector_norm(cube - self.goal_position, dim=1)
        )
        release_score = (
            inside & on_table
        ).to(torch.float32) * torch.clamp(self._qpos[:, self.finger_qadr] / 0.035, 0.0, 1.0)
        post = 4.0 + 2.0 * xy_score + 2.0 * place_score + release_score
        return torch.where(self.has_lifted, post, pre)

    def _apply_joint_target_action(self, actions):
        """Apply normalized incremental joint/gripper target commands.

        Each arm action requests one target-rate increment in radians and the
        gripper action requests one target-rate increment in meters. The
        resulting actuator targets are clamped to the existing safe limits;
        qpos is changed only by MuJoCo-Warp integration.
        """

        torch = self._torch
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        if actions.shape != (self.num_envs, self.num_actions):
            raise ValueError(
                f"actions must have shape ({self.num_envs}, {self.num_actions}), got {tuple(actions.shape)}"
            )
        if not torch.isfinite(actions).all():
            raise FloatingPointError("non-finite CUDA action")
        actions = actions.clamp(-1.0, 1.0)
        previous_actions = self._last_actions.clone()
        action_delta = actions - previous_actions

        previous_joints = self._ctrl[:, self._arm_actuators]
        desired_joints = previous_joints + self.joint_target_rate * actions[:, :6]
        desired_joints = torch.clamp(
            desired_joints, self._action_target_low, self._action_target_high
        )
        previous_gripper = self._ctrl[:, self.gripper_actuator]
        desired_gripper = (previous_gripper + self.gripper_scale * actions[:, 6]).clamp(
            0.0, 0.035
        )
        # The target-rate increments are the action slew limit. Position
        # actuator targets are the only command writes; qpos changes below
        # come exclusively from MuJoCo-Warp integration.
        self._arm_ctrl = desired_joints
        self._ctrl[:, self._arm_actuators] = desired_joints
        self._ctrl[:, self.gripper_actuator] = desired_gripper
        self.gripper_target = desired_gripper
        self._last_actions = actions
        self._last_action_delta = action_delta
        self.last_ik_error.zero_()
        return actions, action_delta

    def _observation_tensor(self):
        """Build the clean 56-field SI observation from current Warp state."""

        torch = self._torch
        finger = self._finger_contacts().to(torch.float32)
        cube = self._cube_position()
        tcp = self._tcp_position()
        pieces = (
            self._qpos[:, self._arm_qadr],
            self._qvel[:, self._arm_dofs],
            self._qpos[:, self.finger_qadr].unsqueeze(1),
            self._qpos[:, self.other_finger_qadr].unsqueeze(1),
            self.gripper_target.unsqueeze(1),
            self._qvel[:, [self.finger_dofadr, self.other_finger_dofadr]],
            tcp,
            cube,
            self._qpos[:, self.cube_qadr + 3 : self.cube_qadr + 7],
            self._qvel[:, self.cube_vadr : self.cube_vadr + 6],
            cube - tcp,
            self.goal_position,
            self.goal_position - cube,
            self._goal_half_size.expand(self.num_envs, -1),
            finger,
            self.has_lifted.to(torch.float32).unsqueeze(1),
            (self.stable_steps.to(torch.float32) / self.settle_steps).unsqueeze(1),
            (self.episode_length_buf.to(torch.float32) / self.max_episode_length).unsqueeze(1),
            self._ctrl[:, self._arm_actuators],
            self.has_placed.to(torch.float32).unsqueeze(1),
            self._last_actions,
        )
        result = torch.cat(pieces, dim=1)
        if result.shape[-1] != self.num_observations:
            raise RuntimeError(
                f"observation layout has {result.shape[-1]} fields, expected {self.num_observations}"
            )
        if not torch.isfinite(result).all():
            raise FloatingPointError("MuJoCo-Warp produced non-finite observation")
        return result

    def _uniform_sensor_noise(self, shape, bound, *, dtype):
        torch = self._torch
        if not self.training_config.sensor_noise or bound == 0.0:
            return torch.zeros(shape, dtype=dtype, device=self.device)
        return (2.0 * torch.rand(
            shape, dtype=dtype, device=self.device, generator=self._torch_rng
        ) - 1.0) * float(bound)

    def _sensor_frame(self, clean, worlds):
        """Sample one noisy sensor frame for selected worlds.

        This method is called exactly once per reset and once per environment
        transition.  ``get_observations`` only selects/copies cached frames.
        Derived differences are reconstructed from the same noisy primary
        vectors, so noise cannot create internally inconsistent geometry.
        """

        torch = self._torch
        worlds = worlds.to(device=self.device, dtype=torch.long)
        frame = clean[worlds].clone()
        count = int(worlds.numel())
        if count == 0:
            return frame
        if not self.training_config.sensor_noise:
            return frame
        arm_q = frame[:, self.OBS_ARM_Q]
        arm_q += self._joint_zero_offset[worlds]
        arm_q += self._uniform_sensor_noise(
            arm_q.shape, self.training_config.joint_position_noise, dtype=arm_q.dtype
        )
        frame[:, self.OBS_ARM_DQ] += self._uniform_sensor_noise(
            (count, 6), self.training_config.joint_velocity_noise, dtype=frame.dtype
        )
        frame[:, self.OBS_FINGER_Q] += self._uniform_sensor_noise(
            (count, 2), self.training_config.finger_position_noise, dtype=frame.dtype
        )
        frame[:, self.OBS_FINGER_DQ] += self._uniform_sensor_noise(
            (count, 2), self.training_config.finger_velocity_noise, dtype=frame.dtype
        )
        frame[:, self.OBS_TCP] += self._uniform_sensor_noise(
            (count, 3), self.training_config.position_noise, dtype=frame.dtype
        )
        frame[:, self.OBS_CUBE] += self._uniform_sensor_noise(
            (count, 3), self.training_config.position_noise, dtype=frame.dtype
        )
        # Apply a bounded axis-angle perturbation, then multiply and normalize
        # so every noisy orientation remains a proper unit quaternion.
        quat = frame[:, self.OBS_CUBE_QUAT]
        angle = self._uniform_sensor_noise(
            (count,), self.training_config.orientation_noise, dtype=frame.dtype
        )
        axis = 2.0 * torch.rand(
            (count, 3), device=self.device, dtype=frame.dtype, generator=self._torch_rng
        ) - 1.0
        axis = axis / torch.linalg.vector_norm(axis, dim=1, keepdim=True).clamp_min(1e-8)
        half = 0.5 * angle
        delta_w = torch.cos(half)
        delta_v = axis * torch.sin(half).unsqueeze(1)
        qw, qv = quat[:, :1], quat[:, 1:]
        dw, dv = delta_w.unsqueeze(1), delta_v
        noisy_q = torch.cat(
            (
                qw * dw - (qv * dv).sum(dim=1, keepdim=True),
                qw * dv + dw * qv + torch.cross(qv, dv, dim=1),
            ),
            dim=1,
        )
        frame[:, self.OBS_CUBE_QUAT] = noisy_q / torch.linalg.vector_norm(
            noisy_q, dim=1, keepdim=True
        ).clamp_min(1e-8)
        frame[:, self.OBS_CUBE_DQ.start : self.OBS_CUBE_DQ.start + 3] += self._uniform_sensor_noise(
            (count, 3), self.training_config.linear_velocity_noise, dtype=frame.dtype
        )
        frame[:, self.OBS_CUBE_DQ.start + 3 : self.OBS_CUBE_DQ.stop] += self._uniform_sensor_noise(
            (count, 3), self.training_config.angular_velocity_noise, dtype=frame.dtype
        )
        frame[:, self.OBS_CUBE_TCP] = (
            frame[:, self.OBS_CUBE] - frame[:, self.OBS_TCP]
        )
        return frame

    def _advance_sensor_history(self, clean):
        if self._sensor_history.shape[1] > 1:
            self._sensor_history[:, 1:] = self._sensor_history[:, :-1].clone()
        self._sensor_history[:, 0] = self._sensor_frame(clean, self._all_worlds)

    def get_observations(self):
        from tensordict import TensorDict

        torch = self._torch
        clean = self._observation_tensor()
        world_ids = self._all_worlds
        policy = self._sensor_history[world_ids, self._sensor_delay].clone()
        # Commands/goals/time are current even when physical sensor columns are
        # delayed.  Goal-minus-cube is rebuilt from the delayed/noisy cube so
        # it remains consistent without leaking a clean current cube to actor.
        policy[:, self.OBS_GRIPPER_TARGET] = clean[:, self.OBS_GRIPPER_TARGET]
        policy[:, self.OBS_GOAL] = clean[:, self.OBS_GOAL]
        policy[:, self.OBS_GOAL_CUBE] = (
            clean[:, self.OBS_GOAL] - policy[:, self.OBS_CUBE]
        )
        policy[:, self.OBS_GOAL_HALF] = clean[:, self.OBS_GOAL_HALF]
        policy[:, self.OBS_TIME] = clean[:, self.OBS_TIME]
        policy[:, self.OBS_ARM_CTRL] = clean[:, self.OBS_ARM_CTRL]
        policy[:, self.OBS_PREVIOUS_ACTION] = clean[:, self.OBS_PREVIOUS_ACTION]
        if not torch.isfinite(policy).all():
            raise FloatingPointError("MuJoCo-Warp produced non-finite actor observation")
        # Return independent clones: RSL-RL retains rollout references before
        # the next transition and must never observe our cache being shifted.
        return TensorDict(
            {"policy": policy.clone(), "critic": clean.clone()},
            batch_size=[self.num_envs],
            device=self.device,
        )

    def _extras(
        self,
        done,
        success,
        timeout,
        inside,
        released,
        on_table,
        still,
        terminal_has_lifted,
        terminal_goal_distance,
        terminal_has_placed,
        terminal_recontacted,
        terminal_recontact_penalty,
        recontact,
    ):
        torch = self._torch
        done = done.to(torch.bool)
        # RSL-RL consumes timeout masks and logs only completed-world metrics;
        # empty tensors are intentional on ordinary non-terminal transitions.
        completed = torch.nonzero(done, as_tuple=False).flatten()
        log = {
            "task/success_rate": success[completed].to(torch.float32),
            "task/lift_rate": terminal_has_lifted[completed].to(torch.float32),
            "task/final_goal_distance": terminal_goal_distance[completed],
            "task/recontact_rate": terminal_recontacted[completed].to(torch.float32),
            "task/recontact_penalty": terminal_recontact_penalty[completed],
        }
        extras = {
            "time_outs": timeout,
            "is_success": success,
            "has_lifted": terminal_has_lifted,
            "has_placed": terminal_has_placed,
            "inside_goal": inside,
            "released": released,
            "on_table": on_table,
            "object_still": still,
            "recontact": recontact,
            "recontacted": terminal_recontacted,
            "recontact_penalty": terminal_recontact_penalty,
            "done": done,
            "done_mask": done,
        }
        if completed.numel():
            extras["log"] = log
        return extras

    def step(self, actions):
        """Apply incremental targets, advance ten 2 ms substeps, and reset done worlds."""

        if self._closed:
            raise RuntimeError("PiperWarpEnv is closed")
        torch = self._torch
        actions, action_delta = self._apply_joint_target_action(actions)
        action_rate_weight = float(self._stage.action_rate_weight)
        self._step_physics()
        self._check_physics_state()
        self.episode_length_buf.add_(1)

        contacts = self._finger_contacts()
        cube = self._cube_position()
        self.has_lifted |= contacts.all(dim=1) & (cube[:, 2] > self.cube_half_size + 0.05)
        inside, on_table, still, released = self._placement_state()
        robot_contact = self._robot_cube_contacts()
        was_placed = self.has_placed.clone()
        recontact = was_placed & robot_contact
        self.recontacted |= recontact
        self.recontact_penalty_total += self.recontact_penalty * recontact.to(torch.float32)
        release_gate = self.has_lifted & inside & on_table & released & ~robot_contact
        self.has_placed |= release_gate
        valid_place = release_gate & still
        self.stable_steps = torch.where(valid_place, self.stable_steps + 1, torch.zeros_like(self.stable_steps))
        success = self.stable_steps >= self.settle_steps
        failed = (cube[:, 2] < -0.025) | (torch.linalg.vector_norm(cube[:, :2], dim=1) > 0.75)
        timeout = (self.episode_length_buf >= self.max_episode_length) & ~success & ~failed
        done = success | failed | timeout

        terminal = success | failed
        potential = self._potential()
        potential = torch.where(terminal, torch.zeros_like(potential), potential)
        rewards = self.shaping_gamma * potential - self._previous_potential
        rewards = rewards - self.time_penalty
        # ``action_delta`` was captured before _last_actions was overwritten in
        # _apply_joint_target_action, so this remains the intended transition
        # penalty even under target slew saturation.
        rewards = rewards - action_rate_weight * torch.square(action_delta).sum(dim=1)
        rewards = rewards + 20.0 * success.to(torch.float32) - 5.0 * failed.to(torch.float32)
        rewards = rewards - self.recontact_penalty * recontact.to(torch.float32)
        self._previous_potential = potential
        terminal_has_lifted = self.has_lifted.clone()
        terminal_has_placed = self.has_placed.clone()
        terminal_recontacted = self.recontacted.clone()
        terminal_recontact_penalty = self.recontact_penalty_total.clone()
        terminal_goal_distance = torch.linalg.vector_norm(
            cube[:, :2] - self.goal_position[:, :2], dim=1
        )
        extras = self._extras(
            done,
            success,
            timeout,
            inside,
            released,
            on_table,
            still,
            terminal_has_lifted,
            terminal_goal_distance,
            terminal_has_placed,
            terminal_recontacted,
            terminal_recontact_penalty,
            recontact,
        )

        # Exactly one cached sensor sample is produced per vector transition.
        # Reads of get_observations below only select from this history.
        self._advance_sensor_history(self._observation_tensor())
        # One call to step is one curriculum tick, regardless of how many
        # independent worlds were advanced in that call.
        self.training_steps += 1
        self._stage = self.training_config.stage_at(self.training_steps)
        self.cfg["training_steps"] = self.training_steps
        self.cfg["curriculum_stage"] = self._stage_at_runtime()

        # Capture terminal metrics before reset, then initialize only those
        # worlds.  Returned observations are always valid post-reset states.
        done_worlds = torch.nonzero(done, as_tuple=False).flatten()
        if done_worlds.numel():
            self._reset_worlds(done_worlds)
        if not torch.isfinite(rewards).all():
            raise FloatingPointError("MuJoCo-Warp produced non-finite reward")
        return self.get_observations(), rewards, done, extras

    # ------------------------------------------------------------------
    # Optional CPU preview (copy, never CPU-step)
    # ------------------------------------------------------------------
    def render_frames(self, indices: Iterable[int] = (0, 1, 2, 3)):
        if self._closed:
            raise RuntimeError("PiperWarpEnv is closed")
        selected = [int(i) for i in indices]
        if any(i < 0 or i >= self.num_envs for i in selected):
            raise IndexError("render index outside the batched environment")
        self._wp.synchronize()
        if self._renderer is None:
            self._renderer = self._mujoco.Renderer(self.model, height=360, width=480)
        frames = []
        for world_id in selected:
            self._mjw.get_data_into(self._render_data, self.model, self._warp_data, world_id=world_id)
            self._renderer.update_scene(
                self._render_data,
                camera=self._render_camera,
                scene_option=self._render_option,
            )
            frames.append(self._renderer.render().copy())
        return frames

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._warp_data = None
        self._warp_model = None
        self._ik_env.close()
