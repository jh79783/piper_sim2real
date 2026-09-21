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
        ``above_cube`` starts each TCP over its cube; ``home`` starts at the
        fixed reachable home TCP pose.
    """

    num_actions = 4
    num_observations = 58
    max_episode_length = 300
    frame_skip = 20
    action_scale = 0.01
    gripper_scale = 0.008
    settle_steps = 8
    shaping_gamma = 0.99
    cube_half_size = 0.02
    workspace_low = np.array([0.24, -0.18, 0.025], dtype=np.float32)
    workspace_high = np.array([0.43, 0.18, 0.15], dtype=np.float32)
    goal_half_size_np = np.array([0.06, 0.05], dtype=np.float32)
    target_rotation_np = np.array(
        [
            [np.cos(2.8), 0.0, np.sin(2.8)],
            [0.0, 1.0, 0.0],
            [-np.sin(2.8), 0.0, np.cos(2.8)],
        ],
        dtype=np.float32,
    )

    def __init__(
        self,
        num_envs: int = 128,
        device: str = "cuda",
        seed: int = 0,
        start_mode: str = "above_cube",
    ) -> None:
        if int(num_envs) < 1:
            raise ValueError("num_envs must be positive")
        if start_mode not in ("above_cube", "home"):
            raise ValueError("start_mode must be above_cube or home")
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
        self.start_mode = start_mode
        self._rng = np.random.default_rng(seed)
        self.seed_value = int(seed)

        # ``build_model`` is shared with the validated CPU contact scene.  It
        # only compiles the model and never steps a CPU world during rollout.
        from scripts.piper_pick_place_env import PiperPickPlaceEnv, build_model

        self.model = build_model()
        self._ik_env = PiperPickPlaceEnv(start_mode=start_mode)
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
            "decimation": self.frame_skip,
            "episode_length": self.max_episode_length,
        }

        self.tcp_id = int(self.model.site("grasp_tcp").id)
        self.cube_geom = int(self.model.geom("cube_geom").id)
        self.cube_body = int(self.model.body("cube").id)
        self.goal_body = int(self.model.body("goal_area").id)
        self.goal_mocap_id = int(self.model.body_mocapid[self.goal_body])
        self.link6_body = int(self.model.body("link6").id)
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

        # Immutable index tensors keep all per-step gathers on CUDA.
        self._arm_qadr = torch.as_tensor(self.arm_qadr, dtype=torch.long, device=torch_device)
        self._arm_dofs = torch.as_tensor(self.arm_dofs, dtype=torch.long, device=torch_device)
        self._arm_actuators = torch.as_tensor(self.arm_actuators, dtype=torch.long, device=torch_device)
        self._joint_low = torch.as_tensor(self.joint_ranges[:, 0], device=torch_device)
        self._joint_high = torch.as_tensor(self.joint_ranges[:, 1], device=torch_device)
        self._goal_half_size = torch.as_tensor(self.goal_half_size_np, device=torch_device)
        self._target_rotation = torch.as_tensor(self.target_rotation_np, device=torch_device)
        self._workspace_low = torch.as_tensor(self.workspace_low, device=torch_device)
        self._workspace_high = torch.as_tensor(self.workspace_high, device=torch_device)
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

        self.goal_xy = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=torch_device)
        self.goal_position = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=torch_device)
        self.target_position = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=torch_device)
        self.gripper_target = torch.full((self.num_envs,), 0.035, dtype=torch.float32, device=torch_device)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.int32, device=torch_device)
        self.has_lifted = torch.zeros(self.num_envs, dtype=torch.bool, device=torch_device)
        self.stable_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=torch_device)
        self.last_ik_error = torch.zeros(self.num_envs, dtype=torch.float32, device=torch_device)
        self._previous_potential = torch.zeros(self.num_envs, dtype=torch.float32, device=torch_device)
        self._last_success = torch.zeros(self.num_envs, dtype=torch.bool, device=torch_device)
        self._last_actions = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float32, device=torch_device)
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

    def _capture_step_graph(self) -> None:
        """Capture the fixed 20-physics-step decimation on Warp's native stream.

        MJWarp's state/control arrays are stable for the lifetime of an env,
        so graph replay observes the latest CUDA-written ``data.ctrl`` and
        advances the current qpos/qvel in place.  If a future Warp/MuJoCo
        build rejects capture for a model feature, the backend keeps the same
        GPU physics path and falls back to the ordinary 20-call loop while
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

    def _sample_layout(self, count: int) -> tuple[np.ndarray, np.ndarray]:
        cube = self._rng.uniform([0.28, -0.14], [0.40, 0.14], size=(count, 2)).astype(np.float32)
        goal = self._rng.uniform([0.28, -0.14], [0.40, 0.14], size=(count, 2)).astype(np.float32)
        pending = np.ones(count, dtype=bool)
        for _ in range(1000):
            pending = np.linalg.norm(goal - cube, axis=1) <= 0.13
            if not pending.any():
                return cube, goal
            goal[pending] = self._rng.uniform(
                [0.28, -0.14], [0.40, 0.14], size=(int(pending.sum()), 2)
            ).astype(np.float32)
        raise RuntimeError("could not sample separated cube and goal positions")

    def _solve_initial_joints(self, cube_xy: np.ndarray) -> np.ndarray:
        if self.start_mode == "home":
            target = np.array([0.33, 0.0, 0.14], dtype=np.float64)
            joints, residual = self._ik_env.solve_ik(target, self._ik_home, iterations=150)
            if residual > 0.005:
                raise RuntimeError(f"home IK failed with residual {residual:.4f} m")
            return np.repeat(joints[None, :], cube_xy.shape[0], axis=0).astype(np.float32)
        joints = np.empty((cube_xy.shape[0], 6), dtype=np.float32)
        for row, xy in zip(joints, cube_xy):
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
        worlds = worlds.to(device=self.device, dtype=torch.long)
        cube_xy_np, goal_xy_np = self._sample_layout(int(worlds.numel()))
        joints_np = self._solve_initial_joints(cube_xy_np)
        cube_xy = torch.as_tensor(cube_xy_np, device=self.device)
        goal_xy = torch.as_tensor(goal_xy_np, device=self.device)
        joints = torch.as_tensor(joints_np, device=self.device)

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
            # launching the dependent device-side forward pass.
            self._torch_warp_stream.wait_stream(torch.cuda.current_stream(self.device))
            self._mjw.forward(self._warp_model, self._warp_data)

        self.goal_xy[worlds] = goal_xy
        self.goal_position[worlds] = torch.cat(
            (goal_xy, torch.full((worlds.numel(), 1), self.cube_half_size, device=self.device)), dim=1
        )
        if self.start_mode == "above_cube":
            self.target_position[worlds] = torch.cat(
                (cube_xy, torch.full((worlds.numel(), 1), 0.10, device=self.device)), dim=1
            )
        else:
            self.target_position[worlds] = torch.tensor(
                [0.33, 0.0, 0.14], device=self.device
            )
        self.gripper_target[worlds] = 0.035
        self.episode_length_buf[worlds] = 0
        self.has_lifted[worlds] = False
        self.stable_steps[worlds] = 0
        self.last_ik_error[worlds] = 0.0
        self._last_success[worlds] = False
        self._previous_potential[worlds] = self._potential()[worlds]

    def reset(self, seed: int | None = None):
        """Reset all worlds and return a CUDA TensorDict observation."""

        if self._closed:
            raise RuntimeError("PiperWarpEnv is closed")
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
            self.seed_value = int(seed)
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

    def _apply_cartesian_action(self, actions):
        torch = self._torch
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        if actions.shape != (self.num_envs, self.num_actions):
            raise ValueError(f"actions must have shape ({self.num_envs}, 4), got {tuple(actions.shape)}")
        if not torch.isfinite(actions).all():
            raise FloatingPointError("non-finite CUDA action")
        actions = actions.clamp(-1.0, 1.0)
        desired = torch.clamp(
            self.target_position + self.action_scale * actions[:, :3],
            self._workspace_low,
            self._workspace_high,
        )
        point = self._site_xpos[:, self.tcp_id]
        body = torch.full((self.num_envs,), self.link6_body, dtype=torch.int32, device=self.device)
        jacp = self._wp.zeros((self.num_envs, 3, self.model.nv), dtype=self._wp.float32, device=str(self.device))
        jacr = self._wp.zeros((self.num_envs, 3, self.model.nv), dtype=self._wp.float32, device=str(self.device))
        with self._stream_scope():
            self._mjw.jac(
                self._warp_model,
                self._warp_data,
                jacp,
                jacr,
                self._wp.from_torch(point, dtype=self._wp.vec3, requires_grad=False),
                self._wp.from_torch(body, dtype=self._wp.int32, requires_grad=False),
            )
        jacp_t = self._wp.to_torch(jacp)[:, :, self._arm_dofs]
        jacr_t = self._wp.to_torch(jacr)[:, :, self._arm_dofs]
        current_rotation = self._site_xmat[:, self.tcp_id]
        rotation_error = 0.5 * torch.cross(
            current_rotation.transpose(-1, -2),
            self._target_rotation.expand(self.num_envs, -1, -1).transpose(-1, -2),
            dim=-1,
        ).sum(dim=-2)
        error = torch.cat((desired - point, 0.3 * rotation_error), dim=1)
        jac = torch.cat((jacp_t, 0.3 * jacr_t), dim=1)
        eye = torch.eye(6, device=self.device).expand(self.num_envs, -1, -1)
        lhs = jac @ jac.transpose(-1, -2) + 0.003**2 * eye
        delta = jac.transpose(-1, -2) @ torch.linalg.solve(lhs, error.unsqueeze(-1))
        delta = delta.squeeze(-1).clamp(-0.1, 0.1)
        current = self._qpos[:, self._arm_qadr]
        joints = torch.clamp(current + delta, self._joint_low + 0.002, self._joint_high - 0.002)
        self.last_ik_error = torch.linalg.vector_norm(desired - point, dim=1)
        self.target_position = desired
        self.gripper_target = torch.clamp(
            self.gripper_target + self.gripper_scale * actions[:, 3], 0.0, 0.035
        )
        # Position actuator targets are the only command writes during step;
        # qpos changes below come exclusively from MuJoCo-Warp integration.
        previous = self._ctrl[:, self._arm_actuators].clone()
        self._arm_ctrl = torch.maximum(torch.minimum(joints, previous + 0.07), previous - 0.07)
        self._ctrl[:, self._arm_actuators] = self._arm_ctrl
        self._ctrl[:, self.gripper_actuator] = self.gripper_target
        self._last_actions = actions
        return actions

    def _observation_tensor(self):
        torch = self._torch
        finger = self._finger_contacts().to(torch.float32)
        cube = self._cube_position()
        tcp = self._tcp_position()
        pieces = (
            self._qpos[:, self._arm_qadr],
            self._qvel[:, self._arm_dofs] * 0.1,
            (self._qpos[:, self.finger_qadr] / 0.035).unsqueeze(1),
            (self._qpos[:, self.other_finger_qadr] / 0.035).unsqueeze(1),
            (self.gripper_target / 0.035).unsqueeze(1),
            self._qvel[:, [self.finger_dofadr, self.other_finger_dofadr]] * 0.1,
            tcp,
            self.target_position,
            cube,
            self._qpos[:, self.cube_qadr + 3 : self.cube_qadr + 7],
            self._qvel[:, self.cube_vadr : self.cube_vadr + 6] * 0.1,
            cube - tcp,
            self.goal_position,
            self.goal_position - cube,
            self._goal_half_size.expand(self.num_envs, -1),
            finger,
            self.has_lifted.to(torch.float32).unsqueeze(1),
            (self.stable_steps.to(torch.float32) / self.settle_steps).unsqueeze(1),
            (self.episode_length_buf.to(torch.float32) / self.max_episode_length).unsqueeze(1),
            self._ctrl[:, self._arm_actuators],
        )
        result = torch.cat(pieces, dim=1)
        if result.shape[-1] != self.num_observations:
            raise RuntimeError(f"observation layout has {result.shape[-1]} fields, expected 58")
        if not torch.isfinite(result).all():
            raise FloatingPointError("MuJoCo-Warp produced non-finite observation")
        return result

    def get_observations(self):
        from tensordict import TensorDict

        obs = self._observation_tensor()
        return TensorDict({"policy": obs}, batch_size=[self.num_envs], device=self.device)

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
        }
        extras = {
            "time_outs": timeout,
            "is_success": success,
            "has_lifted": terminal_has_lifted,
            "inside_goal": inside,
            "released": released,
            "on_table": on_table,
            "object_still": still,
            "done": done,
            "done_mask": done,
        }
        if completed.numel():
            extras["log"] = log
        return extras

    def step(self, actions):
        """Apply batched actions, advance 20 Warp substeps, and auto-reset done worlds."""

        if self._closed:
            raise RuntimeError("PiperWarpEnv is closed")
        torch = self._torch
        actions = self._apply_cartesian_action(actions)
        self._step_physics()
        self._check_physics_state()
        self.episode_length_buf.add_(1)

        contacts = self._finger_contacts()
        cube = self._cube_position()
        self.has_lifted |= contacts.all(dim=1) & (cube[:, 2] > self.cube_half_size + 0.05)
        inside, on_table, still, released = self._placement_state()
        valid_place = self.has_lifted & inside & on_table & still & released
        self.stable_steps = torch.where(valid_place, self.stable_steps + 1, torch.zeros_like(self.stable_steps))
        success = self.stable_steps >= self.settle_steps
        failed = (cube[:, 2] < -0.025) | (torch.linalg.vector_norm(cube[:, :2], dim=1) > 0.75)
        timeout = (self.episode_length_buf >= self.max_episode_length) & ~success & ~failed
        done = success | failed | timeout

        terminal = success | failed
        potential = self._potential()
        potential = torch.where(terminal, torch.zeros_like(potential), potential)
        rewards = self.shaping_gamma * potential - self._previous_potential
        rewards = rewards - 0.01 - 0.001 * torch.square(actions).sum(dim=1)
        rewards = rewards + 20.0 * success.to(torch.float32) - 5.0 * failed.to(torch.float32)
        self._previous_potential = potential
        terminal_has_lifted = self.has_lifted.clone()
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
        )

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
