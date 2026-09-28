"""Isaac Sim (PhysX) Piper pick-and-place with the MuJoCo-era policy contract.

Observation layout (63 fields, raw SI) is identical to ``PiperWarpEnv`` so the
named slices below remain the API shared by tests and future deploy adapters.
The actor receives a noisy, 0-2 tick delayed copy of the sensor columns; the
critic receives the clean observation.  Commands, goal and time are always
current for both.

Deviations from the MuJoCo implementation are deliberate and documented in
``README.md``: contact flags come from PhysX normal forces
(threshold ``contact_force_threshold``) instead of MuJoCo's 1 mm contact
distance, cube angular velocity is world-frame, reset IK is a GPU lookup table,
and PPO uses a clamped Gaussian instead of the tanh-squashed distribution.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import ContactSensor

from isaac_piper.kinematics import PiperKinematics
from isaac_piper.pick_place_env_cfg import CONTACT_LINKS, CUBE_HALF_SIZE, GOAL_HALF_SIZE, PiperPickPlaceEnvCfg
from isaac_piper.robot_cfg import ARM_JOINTS, FINGER_JOINTS, FINGER_OPEN

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
NUM_OBS = 63
# Columns overwritten with current (not delayed) values on the actor view.
CURRENT_OBS_SLICES = (
    OBS_GRIPPER_TARGET,
    OBS_GOAL,
    OBS_GOAL_HALF,
    OBS_TIME,
    OBS_ARM_CTRL,
    OBS_PREVIOUS_ACTION,
)

# Reset IK lookup grid (above-cube TCP height 0.10 m) covering the final layout.
_IK_GRID_X = (0.24, 0.44, 101)
_IK_GRID_Y = (-0.19, 0.19, 191)
_ABOVE_CUBE_HEIGHT = 0.10
_HOME_TCP = (0.33, 0.0, 0.14)
_LAYOUT_CENTER = (0.34, 0.0)
_LAYOUT_HALF_EXTENT = (0.09, 0.18)
_MIN_SEPARATION = 0.13


class PiperPickPlaceEnv(DirectRLEnv):
    cfg: PiperPickPlaceEnvCfg

    def __init__(self, cfg: PiperPickPlaceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        device = self.device
        n = self.num_envs
        self.robot: Articulation = self.scene["robot"]
        self.cube: RigidObject = self.scene["cube"]
        self.contacts: list[ContactSensor] = [self.scene[f"contact_{link}"] for link in CONTACT_LINKS]
        self._contact_finger_bodies = [CONTACT_LINKS.index("link7"), CONTACT_LINKS.index("link8")]

        self._arm_ids = torch.tensor(self.robot.find_joints(ARM_JOINTS, preserve_order=True)[0], device=device)
        self._finger_ids = torch.tensor(
            self.robot.find_joints(FINGER_JOINTS, preserve_order=True)[0], device=device
        )
        self._moving_body_ids = self.robot.find_bodies([f"link{i}" for i in range(1, 9)], preserve_order=True)[0]

        self.kin = PiperKinematics(device=device)
        limits = self.robot.data.joint_pos_limits[0, self._arm_ids]
        self._target_low = limits[:, 0] + cfg.joint_limit_margin
        self._target_high = limits[:, 1] - cfg.joint_limit_margin
        if not torch.allclose(limits[:, 0], self.kin.joint_low, atol=1e-4) or not torch.allclose(
            limits[:, 1], self.kin.joint_high, atol=1e-4
        ):
            raise RuntimeError(f"USD joint limits {limits.tolist()} differ from the Menagerie MJCF")

        self.max_episode_steps = int(self.max_episode_length)
        self.settle_steps = int(round(cfg.success_seconds / self.step_dt))
        self._goal_half = torch.tensor(GOAL_HALF_SIZE, device=device)
        self._cube_extent_local = torch.full((3,), CUBE_HALF_SIZE, device=device)

        # Per-env task state.
        zeros = lambda *shape, dtype=torch.float32: torch.zeros((n, *shape), dtype=dtype, device=device)  # noqa: E731
        self.goal_position = zeros(3)
        self.arm_target = zeros(6)
        self.gripper_target = torch.full((n,), FINGER_OPEN, device=device)
        self.actions = zeros(7)
        self.previous_actions = zeros(7)
        self.action_delta = zeros(7)
        self.has_lifted = zeros(dtype=torch.bool)
        self.has_placed = zeros(dtype=torch.bool)
        self.recontacted = zeros(dtype=torch.bool)
        self.recontact_penalty_total = zeros()
        self.stable_steps = zeros(dtype=torch.long)
        self.finger_contacts = zeros(2, dtype=torch.bool)
        self.robot_contact = zeros(dtype=torch.bool)
        self.previous_potential = zeros()
        self._step = {}

        # Sensor model state.
        self._rng = torch.Generator(device=device)
        self._rng.manual_seed(int(cfg.seed) if cfg.seed is not None else 0)
        self._cpu_rng = torch.Generator()
        self._cpu_rng.manual_seed(int(cfg.seed) + 1 if cfg.seed is not None else 1)
        rand = cfg.randomization
        self._joint_zero_offset = zeros(6)
        self._sensor_delay = zeros(dtype=torch.long)
        self._sensor_history = zeros(rand.max_sensor_delay_steps + 1, NUM_OBS)

        self._init_randomization_defaults()
        self._build_ik_table()
        self._home_joints = self._solve_ik(torch.tensor([_HOME_TCP], device=device))[0]
        self.goal_marker = VisualizationMarkers(cfg.goal_marker)
        self._all_ids = torch.arange(n, device=device)
        # DirectRLEnv.reset() is the first reset; stage 0 until then.
        self.stage = cfg.stages[0]

    # ------------------------------------------------------------------ scene
    def _setup_scene(self):
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

    # -------------------------------------------------------------- curriculum
    @property
    def training_steps(self) -> int:
        """Vector-env ticks, independent of env count (the curriculum clock)."""
        return int(self.common_step_counter)

    def _current_stage(self):
        if not self.cfg.curriculum:
            return self.cfg.stages[-1]
        return next(s for s in reversed(self.cfg.stages) if self.training_steps >= s.step)

    # --------------------------------------------------------------- reset IK
    def _build_ik_table(self):
        """Solve above-cube IK once on a 2 mm grid; resets interpolate it."""
        xs = torch.linspace(*_IK_GRID_X, device=self.device)
        ys = torch.linspace(*_IK_GRID_Y, device=self.device)
        grid = torch.stack(torch.meshgrid(xs, ys, indexing="ij"), dim=-1).reshape(-1, 2)
        targets = torch.cat((grid, torch.full((grid.shape[0], 1), _ABOVE_CUBE_HEIGHT, device=self.device)), 1)
        joints = self._solve_ik(targets)
        self._ik_x, self._ik_y = xs, ys
        self._ik_table = joints.reshape(len(xs), len(ys), 6)

    def _solve_ik(self, targets):
        joints, residual = self.kin.solve_ik(targets)
        if residual.max().item() > 0.005:
            raise RuntimeError(f"Piper reset IK failed: max residual {residual.max().item():.4f} m")
        return joints

    def _above_cube_joints(self, cube_xy: torch.Tensor) -> torch.Tensor:
        xs, ys, table = self._ik_x, self._ik_y, self._ik_table
        fx = ((cube_xy[:, 0] - xs[0]) / (xs[1] - xs[0])).clamp(0, len(xs) - 1 - 1e-4)
        fy = ((cube_xy[:, 1] - ys[0]) / (ys[1] - ys[0])).clamp(0, len(ys) - 1 - 1e-4)
        ix, iy = fx.floor().long(), fy.floor().long()
        tx, ty = (fx - ix)[:, None], (fy - iy)[:, None]
        return (
            table[ix, iy] * (1 - tx) * (1 - ty)
            + table[ix + 1, iy] * tx * (1 - ty)
            + table[ix, iy + 1] * (1 - tx) * ty
            + table[ix + 1, iy + 1] * tx * ty
        )

    # ------------------------------------------------------- randomization
    def _init_randomization_defaults(self):
        rv, cv = self.robot.root_physx_view, self.cube.root_physx_view
        self._robot_mass0 = rv.get_masses().clone()
        self._robot_inertia0 = rv.get_inertias().clone()
        self._robot_com0 = rv.get_coms().clone()
        self._robot_material0 = rv.get_material_properties().clone()
        self._cube_mass0 = cv.get_masses().clone()
        self._cube_inertia0 = cv.get_inertias().clone()
        self._cube_com0 = cv.get_coms().clone()
        self._cube_material0 = cv.get_material_properties().clone()

    def _uniform_cpu(self, shape, half_range, scale):
        u = 2.0 * torch.rand(shape, generator=self._cpu_rng) - 1.0
        return u * float(half_range) * float(scale)

    def _bucketed_scale(self, shape, fraction, scale):
        buckets = int(self.cfg.randomization.friction_buckets)
        index = torch.randint(0, buckets, shape, generator=self._cpu_rng).float()
        u = -1.0 + 2.0 * index / max(buckets - 1, 1)
        return 1.0 + u * float(fraction) * float(scale)

    def _randomize_physics(self, env_ids: torch.Tensor, scale: float):
        """Restore nominal parameters, then apply this episode's perturbation."""
        rand = self.cfg.randomization
        if not rand.domain_randomization:
            scale = 0.0
        ids = env_ids.cpu()
        count = len(ids)
        bodies = torch.tensor(self._moving_body_ids)

        rv = self.robot.root_physx_view
        masses, inertias, coms = rv.get_masses(), rv.get_inertias(), rv.get_coms()
        mass_scale = 1.0 + self._uniform_cpu((count, len(bodies)), rand.link_mass_fraction, scale)
        inertia_scale = mass_scale * (1.0 + self._uniform_cpu((count, len(bodies)), rand.inertia_fraction, scale))
        rows, cols = ids[:, None], bodies[None, :]
        masses[ids] = self._robot_mass0[ids]
        masses[rows, cols] = self._robot_mass0[rows, cols] * mass_scale
        inertias[ids] = self._robot_inertia0[ids]
        inertias[rows, cols] = self._robot_inertia0[rows, cols] * inertia_scale[..., None]
        coms[ids] = self._robot_com0[ids]
        coms[rows, cols, :3] += self._uniform_cpu((count, len(bodies), 3), rand.link_com_range, scale)
        rv.set_masses(masses, ids)
        rv.set_inertias(inertias, ids)
        rv.set_coms(coms, ids)
        materials = rv.get_material_properties()
        friction = self._bucketed_scale((count, 1), rand.friction_fraction, scale)
        materials[ids] = self._robot_material0[ids]
        materials[ids, :, :2] *= friction[..., None]
        rv.set_material_properties(materials, ids)

        cv = self.cube.root_physx_view
        masses, inertias, coms = cv.get_masses(), cv.get_inertias(), cv.get_coms()
        mass_scale = 1.0 + self._uniform_cpu((count, 1), rand.cube_mass_fraction, scale)
        inertia_scale = mass_scale * (1.0 + self._uniform_cpu((count, 1), rand.inertia_fraction, scale))
        masses[ids] = self._cube_mass0[ids] * mass_scale.reshape(masses[ids].shape)
        inertias[ids] = self._cube_inertia0[ids] * inertia_scale.reshape(-1, *([1] * (inertias.dim() - 1)))
        coms[ids] = self._cube_com0[ids]
        com_view = coms.reshape(coms.shape[0], -1)
        com_view[ids, :3] += self._uniform_cpu((count, 3), rand.cube_com_range, scale)
        cv.set_masses(masses, ids)
        cv.set_inertias(inertias, ids)
        cv.set_coms(coms, ids)
        materials = cv.get_material_properties()
        friction = self._bucketed_scale((count, 1), rand.friction_fraction, scale)
        materials[ids] = self._cube_material0[ids]
        materials[ids, :, :2] *= friction[..., None]
        cv.set_material_properties(materials, ids)

    # ----------------------------------------------------------- layout
    def _sample_layout(self, count: int, layout_scale: float):
        center = torch.tensor(_LAYOUT_CENTER, device=self.device)
        half = torch.tensor(_LAYOUT_HALF_EXTENT, device=self.device) * layout_scale
        min_sep = _MIN_SEPARATION * layout_scale

        def uniform(k):
            return center - half + 2.0 * half * torch.rand((k, 2), device=self.device, generator=self._rng)

        cube, goal = uniform(count), uniform(count)
        pending = torch.linalg.vector_norm(goal - cube, dim=1) <= min_sep
        for _ in range(10_000):
            if not pending.any():
                return cube, goal
            goal[pending] = uniform(int(pending.sum()))
            pending = torch.linalg.vector_norm(goal - cube, dim=1) <= min_sep
        raise RuntimeError("could not sample separated cube and goal positions")

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._all_ids
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._log_terminal(env_ids)
        super()._reset_idx(env_ids)
        self.stage = self._current_stage()
        count = len(env_ids)
        rand = self.cfg.randomization

        cube_xy, goal_xy = self._sample_layout(count, self.stage.layout_scale)
        if self.cfg.start_mode == "home":
            start_home = torch.ones(count, dtype=torch.bool, device=self.device)
        elif self.cfg.start_mode == "above_cube":
            start_home = torch.zeros(count, dtype=torch.bool, device=self.device)
        else:
            start_home = torch.rand(count, device=self.device, generator=self._rng) < self.stage.home_probability
        joints = torch.where(start_home[:, None], self._home_joints.expand(count, 6), self._above_cube_joints(cube_xy))

        self._randomize_physics(env_ids, self.stage.randomization_scale)

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_pos[:, self._arm_ids] = joints
        joint_pos[:, self._finger_ids] = torch.tensor([FINGER_OPEN, -FINGER_OPEN], device=self.device)
        self.robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)

        origins = self.scene.env_origins[env_ids]
        cube_state = self.cube.data.default_root_state[env_ids].clone()
        cube_state[:, :2] = cube_xy + origins[:, :2]
        cube_state[:, 2] = CUBE_HALF_SIZE + 0.0005 + origins[:, 2]
        cube_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        cube_state[:, 7:] = 0.0
        self.cube.write_root_state_to_sim(cube_state, env_ids=env_ids)

        self.goal_position[env_ids] = torch.cat(
            (goal_xy, torch.full((count, 1), CUBE_HALF_SIZE, device=self.device)), dim=1
        )
        self.arm_target[env_ids] = joints
        self.gripper_target[env_ids] = FINGER_OPEN
        self.actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0
        self.action_delta[env_ids] = 0.0
        self.has_lifted[env_ids] = False
        self.has_placed[env_ids] = False
        self.recontacted[env_ids] = False
        self.recontact_penalty_total[env_ids] = 0.0
        self.stable_steps[env_ids] = 0
        # The contact sensor still holds the previous episode's last frame.
        self.finger_contacts[env_ids] = False
        self.robot_contact[env_ids] = False

        if rand.sensor_noise:
            self._joint_zero_offset[env_ids] = self._uniform((count, 6), rand.joint_zero_offset)
            self._sensor_delay[env_ids] = torch.randint(
                0, rand.max_sensor_delay_steps + 1, (count,), device=self.device, generator=self._rng
            )
        else:
            self._joint_zero_offset[env_ids] = 0.0
            self._sensor_delay[env_ids] = 0

        self.previous_potential[env_ids] = self._potential()[env_ids]
        frame = self._sensor_frame(self._clean_observation(), env_ids)
        self._sensor_history[env_ids] = frame.unsqueeze(1).expand(-1, self._sensor_history.shape[1], -1)

        goal_w = self.goal_position.clone()
        goal_w[:, 2] = 0.001
        self.goal_marker.visualize(translations=goal_w + self.scene.env_origins)

    def _log_terminal(self, env_ids: torch.Tensor):
        if not self._step:
            return
        s = self._step
        self.extras["log"] = {
            "task/success_rate": s["success"][env_ids].float().mean(),
            "task/lift_rate": self.has_lifted[env_ids].float().mean(),
            "task/placed_rate": self.has_placed[env_ids].float().mean(),
            "task/failure_rate": s["failed"][env_ids].float().mean(),
            "task/final_goal_distance": s["goal_distance"][env_ids].mean(),
            "task/recontact_rate": self.recontacted[env_ids].float().mean(),
            "task/recontact_penalty": self.recontact_penalty_total[env_ids].mean(),
            "curriculum/stage": torch.tensor(float(self.cfg.stages.index(self.stage))),
            "curriculum/training_steps": torch.tensor(float(self.training_steps)),
        }

    # ------------------------------------------------------------- actions
    def _pre_physics_step(self, actions: torch.Tensor):
        if not torch.isfinite(actions).all():
            raise FloatingPointError("non-finite action")
        # Episode metrics are published only on steps that reset some env.
        self.extras.pop("log", None)
        actions = actions.clamp(-1.0, 1.0)
        self.action_delta = actions - self.previous_actions
        self.previous_actions = actions.clone()
        self.actions = actions
        self.arm_target = torch.clamp(
            self.arm_target + self.cfg.joint_target_rate * actions[:, :6], self._target_low, self._target_high
        )
        self.gripper_target = torch.clamp(
            self.gripper_target + self.cfg.gripper_target_rate * actions[:, 6], 0.0, FINGER_OPEN
        )
        self.robot.set_joint_position_target(self.arm_target, joint_ids=self._arm_ids)
        fingers = torch.stack((self.gripper_target, -self.gripper_target), dim=1)
        self.robot.set_joint_position_target(fingers, joint_ids=self._finger_ids)

    def _apply_action(self):
        # Targets are latched in _pre_physics_step; scene.write_data_to_sim
        # pushes them to PhysX on every substep.
        pass

    # ------------------------------------------------------------ state
    def _tcp(self) -> torch.Tensor:
        return self.kin.tcp_position(self.robot.data.joint_pos[:, self._arm_ids])

    def _cube_pos(self) -> torch.Tensor:
        return self.cube.data.root_pos_w - self.scene.env_origins

    def _read_contacts(self):
        forces = torch.stack(
            [torch.linalg.vector_norm(s.data.force_matrix_w[:, 0], dim=-1).sum(-1) for s in self.contacts], dim=1
        )
        touching = forces > self.cfg.contact_force_threshold
        self.finger_contacts = touching[:, self._contact_finger_bodies]
        self.robot_contact = touching.any(dim=1)

    def _placement_state(self):
        cube = self._cube_pos()
        rotation = math_utils.matrix_from_quat(self.cube.data.root_quat_w)
        extent = torch.abs(rotation) @ self._cube_extent_local
        inside = torch.all(
            torch.abs(cube[:, :2] - self.goal_position[:, :2]) + extent[:, :2] <= self._goal_half - 0.002, dim=1
        )
        on_table = torch.abs(cube[:, 2] - extent[:, 2]) < 0.006
        still = (torch.linalg.vector_norm(self.cube.data.root_lin_vel_w, dim=1) < 0.03) & (
            torch.linalg.vector_norm(self.cube.data.root_ang_vel_w, dim=1) < 0.5
        )
        finger_q = self.robot.data.joint_pos[:, self._finger_ids[0]]
        released = (finger_q > 0.028) & ~self.finger_contacts.any(dim=1)
        return inside, on_table, still, released

    def _potential(self) -> torch.Tensor:
        cube, tcp = self._cube_pos(), self._tcp()
        reach = 1.0 - torch.tanh(10.0 * torch.linalg.vector_norm(tcp - cube, dim=1))
        grasped = self.finger_contacts.all(dim=1).float()
        height = torch.clamp((cube[:, 2] - CUBE_HALF_SIZE) / 0.06, 0.0, 1.0)
        pre = reach + grasped + 2.0 * height
        inside, on_table, _, _ = self._placement_state()
        xy_score = 1.0 - torch.tanh(8.0 * torch.linalg.vector_norm(cube[:, :2] - self.goal_position[:, :2], dim=1))
        place_score = 1.0 - torch.tanh(8.0 * torch.linalg.vector_norm(cube - self.goal_position, dim=1))
        finger_q = self.robot.data.joint_pos[:, self._finger_ids[0]]
        release_score = (inside & on_table).float() * torch.clamp(finger_q / FINGER_OPEN, 0.0, 1.0)
        post = 4.0 + 2.0 * xy_score + 2.0 * place_score + release_score
        return torch.where(self.has_lifted, post, pre)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not torch.isfinite(self.robot.data.joint_pos).all() or not torch.isfinite(self.cube.data.root_pos_w).all():
            raise FloatingPointError("PhysX produced a non-finite state")
        self._read_contacts()
        cube = self._cube_pos()
        self.has_lifted |= self.finger_contacts.all(dim=1) & (cube[:, 2] > CUBE_HALF_SIZE + 0.05)
        inside, on_table, still, released = self._placement_state()
        recontact = self.has_placed & self.robot_contact
        self.recontacted |= recontact
        self.recontact_penalty_total += self.cfg.recontact_penalty * recontact.float()
        release_gate = self.has_lifted & inside & on_table & released & ~self.robot_contact
        self.has_placed |= release_gate
        valid_place = release_gate & still
        self.stable_steps = torch.where(valid_place, self.stable_steps + 1, torch.zeros_like(self.stable_steps))
        success = self.stable_steps >= self.settle_steps
        failed = (cube[:, 2] < -0.025) | (torch.linalg.vector_norm(cube[:, :2], dim=1) > 0.75)
        terminated = success | failed
        time_out = (self.episode_length_buf >= self.max_episode_steps) & ~terminated
        self._step = {
            "success": success,
            "failed": failed,
            "recontact": recontact,
            "goal_distance": torch.linalg.vector_norm(cube[:, :2] - self.goal_position[:, :2], dim=1),
        }
        return terminated, time_out

    def _get_rewards(self) -> torch.Tensor:
        s = self._step
        terminal = s["success"] | s["failed"]
        potential = torch.where(terminal, torch.zeros_like(self.previous_potential), self._potential())
        rewards = self.cfg.shaping_gamma * potential - self.previous_potential
        rewards -= self.cfg.time_penalty
        rewards -= self.stage.action_rate_weight * torch.square(self.action_delta).sum(dim=1)
        rewards += self.cfg.success_bonus * s["success"].float() - self.cfg.failure_penalty * s["failed"].float()
        rewards -= self.cfg.recontact_penalty * s["recontact"].float()
        self.previous_potential = potential
        # Exactly one sensor sample per transition; reset envs are refilled
        # with their post-reset frame in _reset_idx.
        self._sensor_history = torch.roll(self._sensor_history, shifts=1, dims=1)
        self._sensor_history[:, 0] = self._sensor_frame(self._clean_observation(), self._all_ids)
        if not torch.isfinite(rewards).all():
            raise FloatingPointError("non-finite reward")
        return rewards

    # -------------------------------------------------------- observations
    def _clean_observation(self) -> torch.Tensor:
        joint_pos, joint_vel = self.robot.data.joint_pos, self.robot.data.joint_vel
        cube, tcp = self._cube_pos(), self._tcp()
        obs = torch.cat(
            (
                joint_pos[:, self._arm_ids],
                joint_vel[:, self._arm_ids],
                joint_pos[:, self._finger_ids],
                self.gripper_target[:, None],
                joint_vel[:, self._finger_ids],
                tcp,
                cube,
                self.cube.data.root_quat_w,
                self.cube.data.root_lin_vel_w,
                self.cube.data.root_ang_vel_w,
                cube - tcp,
                self.goal_position,
                self.goal_position - cube,
                self._goal_half.expand(self.num_envs, 2),
                self.finger_contacts.float(),
                self.has_lifted.float()[:, None],
                (self.stable_steps.float() / self.settle_steps)[:, None],
                (self.episode_length_buf.float() / self.max_episode_steps)[:, None],
                self.arm_target,
                self.has_placed.float()[:, None],
                self.previous_actions,
            ),
            dim=1,
        )
        if obs.shape[1] != NUM_OBS:
            raise RuntimeError(f"observation has {obs.shape[1]} fields, expected {NUM_OBS}")
        return obs

    def _uniform(self, shape, bound):
        if not self.cfg.randomization.sensor_noise or bound == 0.0:
            return torch.zeros(shape, device=self.device)
        return (2.0 * torch.rand(shape, device=self.device, generator=self._rng) - 1.0) * float(bound)

    def _sensor_frame(self, clean: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        """One noisy sensor sample for ``env_ids`` (zero offset + uniform noise)."""
        frame = clean[env_ids].clone()
        rand = self.cfg.randomization
        if not rand.sensor_noise:
            return frame
        k = len(env_ids)
        frame[:, OBS_ARM_Q] += self._joint_zero_offset[env_ids] + self._uniform((k, 6), rand.joint_position_noise)
        frame[:, OBS_ARM_DQ] += self._uniform((k, 6), rand.joint_velocity_noise)
        frame[:, OBS_FINGER_Q] += self._uniform((k, 2), rand.finger_position_noise)
        frame[:, OBS_FINGER_DQ] += self._uniform((k, 2), rand.finger_velocity_noise)
        frame[:, OBS_TCP] += self._uniform((k, 3), rand.position_noise)
        frame[:, OBS_CUBE] += self._uniform((k, 3), rand.position_noise)
        # Bounded axis-angle perturbation keeps a unit quaternion.
        axis = torch.nn.functional.normalize(
            2.0 * torch.rand((k, 3), device=self.device, generator=self._rng) - 1.0, dim=1, eps=1e-8
        )
        angle = self._uniform((k,), rand.orientation_noise)
        delta = math_utils.quat_from_angle_axis(angle, axis)
        frame[:, OBS_CUBE_QUAT] = math_utils.normalize(math_utils.quat_mul(frame[:, OBS_CUBE_QUAT], delta))
        frame[:, OBS_CUBE_DQ.start : OBS_CUBE_DQ.start + 3] += self._uniform((k, 3), rand.linear_velocity_noise)
        frame[:, OBS_CUBE_DQ.start + 3 : OBS_CUBE_DQ.stop] += self._uniform((k, 3), rand.angular_velocity_noise)
        frame[:, OBS_CUBE_TCP] = frame[:, OBS_CUBE] - frame[:, OBS_TCP]
        return frame

    def _get_observations(self) -> dict:
        clean = self._clean_observation()
        policy = self._sensor_history[self._all_ids, self._sensor_delay].clone()
        for columns in CURRENT_OBS_SLICES:
            policy[:, columns] = clean[:, columns]
        policy[:, OBS_GOAL_CUBE] = clean[:, OBS_GOAL] - policy[:, OBS_CUBE]
        if not torch.isfinite(policy).all():
            raise FloatingPointError("non-finite actor observation")
        return {"policy": policy, "critic": clean}
