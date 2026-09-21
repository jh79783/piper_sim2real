"""Contact-based Piper pick-and-place with independently randomized cube and goal."""

from pathlib import Path
import xml.etree.ElementTree as ET

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_model():
    robot_dir = PROJECT_ROOT / "third_party/mujoco_menagerie/agilex_piper"
    root = ET.parse(robot_dir / "piper.xml").getroot()
    root.find("compiler").set("meshdir", str(robot_dir / "assets"))
    # The original keyframe does not include the new object's free joint.
    for keyframes in root.findall("keyframe"):
        root.remove(keyframes)
    ET.SubElement(root.find(".//body[@name='link6']"), "site",
                  name="grasp_tcp", pos="0 0 0.12", size="0.004", rgba="0.1 0.4 1 0.7")
    for section in ET.parse(PROJECT_ROOT / "scenes/piper_pick_place.xml").getroot():
        existing = root.find(section.tag)
        if existing is None:
            root.append(section)
        else:
            existing.attrib.update(section.attrib)
            existing.extend(list(section))
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


class PiperPickPlaceEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}
    cube_half_size = 0.02
    goal_half_size = np.array([0.06, 0.05])
    frame_skip = 20
    action_scale = 0.01
    max_episode_steps = 300
    settle_steps = 8
    shaping_gamma = 0.99
    workspace_low = np.array([0.24, -0.18, 0.025])
    workspace_high = np.array([0.43, 0.18, 0.15])

    def __init__(self, render_mode=None, width=480, height=360, start_mode="above_cube"):
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or rgb_array")
        if start_mode not in ("above_cube", "home"):
            raise ValueError("start_mode must be above_cube or home")
        self.render_mode = render_mode
        self.width, self.height = width, height
        self.start_mode = start_mode
        self.model = build_model()
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)
        self.tcp_id = self.model.site("grasp_tcp").id
        self.cube_geom = self.model.geom("cube_geom").id
        self.cube_body = self.model.body("cube").id
        self.cube_qadr = int(self.model.jnt_qposadr[self.model.joint("cube_free").id])
        self.cube_vadr = int(self.model.jnt_dofadr[self.model.joint("cube_free").id])
        self.arm_qadr = np.array([self.model.joint(f"joint{i}").qposadr[0] for i in range(1, 7)])
        self.arm_dofs = np.array([self.model.joint(f"joint{i}").dofadr[0] for i in range(1, 7)])
        self.arm_actuators = np.array([self.model.actuator(f"joint{i}").id for i in range(1, 7)])
        self.joint_ranges = self.model.jnt_range[[self.model.joint(f"joint{i}").id for i in range(1, 7)]]
        self.finger_qadr = int(self.model.joint("joint7").qposadr[0])
        self.other_finger_qadr = int(self.model.joint("joint8").qposadr[0])
        self.grip_actuator = self.model.actuator("gripper").id
        self.goal_mocap = self.model.body_mocapid[self.model.body("goal_area").id]
        self.finger_geoms = []
        for body in ("link7", "link8"):
            ids = np.flatnonzero((self.model.geom_bodyid == self.model.body(body).id)
                                 & (self.model.geom_contype != 0))
            self.finger_geoms.append(set(ids.tolist()))
        # A slightly tilted downward approach stays within Piper's wrist limits.
        angle = 2.8
        self.target_rotation = np.array([
            [np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ])
        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))
        self.action_space = spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (58,), dtype=np.float32)
        self.renderer = None
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [0.28, 0, 0.13]
        self.camera.distance = 1.05
        self.camera.azimuth = 140
        self.camera.elevation = -28
        self.visual_options = mujoco.MjvOption()
        self.visual_options.geomgroup[3] = 0

    @property
    def tcp_position(self):
        return self.data.site_xpos[self.tcp_id].copy()

    @property
    def cube_position(self):
        return self.data.xpos[self.cube_body].copy()

    def solve_ik(self, target, initial_q, iterations=30):
        """Fixed downward orientation; only actuators move the real simulation."""
        self.ik_data.qpos[:] = self.data.qpos
        self.ik_data.qpos[self.arm_qadr] = initial_q
        for _ in range(iterations):
            mujoco.mj_forward(self.model, self.ik_data)
            rotation = self.ik_data.site_xmat[self.tcp_id].reshape(3, 3)
            rotation_error = 0.5 * np.cross(rotation.T, self.target_rotation.T).sum(axis=0)
            error = np.r_[target - self.ik_data.site_xpos[self.tcp_id], 0.3 * rotation_error]
            if np.linalg.norm(error) < 2e-5:
                break
            mujoco.mj_jacSite(self.model, self.ik_data, self._jacp, self._jacr, self.tcp_id)
            jac = np.vstack([self._jacp[:, self.arm_dofs], 0.3 * self._jacr[:, self.arm_dofs]])
            delta = jac.T @ np.linalg.solve(jac @ jac.T + 0.003**2 * np.eye(6), error)
            self.ik_data.qpos[self.arm_qadr] = np.clip(
                self.ik_data.qpos[self.arm_qadr] + np.clip(delta, -0.1, 0.1),
                self.joint_ranges[:, 0] + 0.002, self.joint_ranges[:, 1] - 0.002,
            )
        mujoco.mj_forward(self.model, self.ik_data)
        residual = float(np.linalg.norm(target - self.ik_data.site_xpos[self.tcp_id]))
        return self.ik_data.qpos[self.arm_qadr].copy(), residual

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        # Both positions vary independently, except that an already-solved layout is excluded.
        cube_xy = self.np_random.uniform([0.28, -0.14], [0.40, 0.14])
        for _ in range(1000):
            goal_xy = self.np_random.uniform([0.28, -0.14], [0.40, 0.14])
            if np.linalg.norm(goal_xy - cube_xy) > 0.13:
                break
        else:
            raise RuntimeError("Could not sample separated cube and goal positions")
        self.goal_position = np.r_[goal_xy, self.cube_half_size]
        self.data.mocap_pos[self.goal_mocap] = [*goal_xy, 0.001]
        self.data.qpos[self.cube_qadr:self.cube_qadr + 7] = [*cube_xy, 0.0205, 1, 0, 0, 0]
        self.data.qpos[self.finger_qadr] = 0.035
        self.data.qpos[self.other_finger_qadr] = -0.035
        start = np.r_[cube_xy, 0.10] if self.start_mode == "above_cube" else np.array([0.33, 0, 0.14])
        home = np.array([0, 1.57, -1.3485, 0, 0, 0])
        joints, residual = self.solve_ik(start, home, iterations=150)
        if residual > 0.005:
            raise RuntimeError(f"Piper reset IK failed: residual={residual:.4f} m")
        self.data.qpos[self.arm_qadr] = joints
        self.data.ctrl[self.arm_actuators] = joints
        self.gripper_target = 0.035
        self.data.ctrl[self.grip_actuator] = self.gripper_target
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data, nstep=100)
        self.target_position = self.tcp_position
        self.episode_steps = 0
        self.has_lifted = False
        self.stable_steps = 0
        self.last_ik_error = 0.0
        self._previous_potential = self._potential()
        return self._observation(), self._info()

    def _finger_contacts(self):
        touched = [False, False]
        for contact in self.data.contact:
            if contact.geom1 == self.cube_geom:
                other = contact.geom2
            elif contact.geom2 == self.cube_geom:
                other = contact.geom1
            else:
                continue
            if contact.dist <= 0.001:
                for index, ids in enumerate(self.finger_geoms):
                    touched[index] |= other in ids
        return np.array(touched, dtype=bool)

    def _placement_state(self):
        cube = self.cube_position
        rotation = self.data.geom_xmat[self.cube_geom].reshape(3, 3)
        extent = np.abs(rotation) @ np.full(3, self.cube_half_size)
        inside = bool(np.all(np.abs(cube[:2] - self.goal_position[:2]) + extent[:2]
                             <= self.goal_half_size - 0.002))
        on_table = abs(cube[2] - extent[2]) < 0.006
        velocity = self.data.qvel[self.cube_vadr:self.cube_vadr + 6]
        still = np.linalg.norm(velocity[:3]) < 0.03 and np.linalg.norm(velocity[3:]) < 0.5
        released = self.data.qpos[self.finger_qadr] > 0.028 and not self._finger_contacts().any()
        return inside, bool(on_table), bool(still), bool(released)

    def _potential(self):
        cube = self.cube_position
        reach = 1.0 - np.tanh(10.0 * np.linalg.norm(self.tcp_position - cube))
        grasped = bool(self._finger_contacts().all())
        if not self.has_lifted:
            height = np.clip((cube[2] - self.cube_half_size) / 0.06, 0, 1)
            return float(reach + grasped + 2.0 * height)
        xy_score = 1.0 - np.tanh(8.0 * np.linalg.norm(cube[:2] - self.goal_position[:2]))
        place_score = 1.0 - np.tanh(8.0 * np.linalg.norm(cube - self.goal_position))
        inside, on_table, _, _ = self._placement_state()
        release_score = float(inside and on_table) * np.clip(self.data.qpos[self.finger_qadr] / 0.035, 0, 1)
        return float(4.0 + 2.0 * xy_score + 2.0 * place_score + release_score)

    def _info(self):
        inside, on_table, still, released = self._placement_state()
        return {
            "is_success": bool(self.stable_steps >= self.settle_steps),
            "is_grasped": bool(self._finger_contacts().all()),
            "has_lifted": bool(self.has_lifted),
            "inside_goal": inside,
            "released": released,
            "on_table": on_table,
            "object_still": still,
            "goal_distance": float(np.linalg.norm(self.cube_position[:2] - self.goal_position[:2])),
            "ik_error": self.last_ik_error,
        }

    def _observation(self):
        # Positions, velocities and controller/history state make the task goal-conditioned.
        obs = np.concatenate([
            self.data.qpos[self.arm_qadr], self.data.qvel[self.arm_dofs] * 0.1,
            [self.data.qpos[self.finger_qadr] / 0.035,
             self.data.qpos[self.other_finger_qadr] / 0.035, self.gripper_target / 0.035],
            self.data.qvel[6:8] * 0.1,
            self.tcp_position, self.target_position,
            self.cube_position, self.data.qpos[self.cube_qadr + 3:self.cube_qadr + 7],
            self.data.qvel[self.cube_vadr:self.cube_vadr + 6] * 0.1,
            self.cube_position - self.tcp_position, self.goal_position,
            self.goal_position - self.cube_position, self.goal_half_size,
            self._finger_contacts().astype(float),
            [float(self.has_lifted), self.stable_steps / self.settle_steps,
             self.episode_steps / self.max_episode_steps],
            self.data.ctrl[self.arm_actuators],
        ])
        return obs.astype(np.float32)

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (4,) or not np.isfinite(action).all():
            raise ValueError("action must be a finite 4D vector")
        action = np.clip(action, -1, 1)
        desired = np.clip(self.target_position + self.action_scale * action[:3],
                          self.workspace_low, self.workspace_high)
        joints, self.last_ik_error = self.solve_ik(desired, self.data.qpos[self.arm_qadr])
        self.target_position = desired
        self.gripper_target = float(np.clip(self.gripper_target + 0.008 * action[3], 0, 0.035))
        # Position controllers, not qpos teleportation or object/gripper welds.
        self.data.ctrl[self.arm_actuators] = np.clip(
            joints, self.data.ctrl[self.arm_actuators] - 0.07,
            self.data.ctrl[self.arm_actuators] + 0.07,
        )
        self.data.ctrl[self.grip_actuator] = self.gripper_target
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        self.episode_steps += 1
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError("MuJoCo produced non-finite state")
        if self._finger_contacts().all() and self.cube_position[2] > self.cube_half_size + 0.05:
            self.has_lifted = True
        inside, on_table, still, released = self._placement_state()
        valid_place = self.has_lifted and inside and on_table and still and released
        self.stable_steps = self.stable_steps + 1 if valid_place else 0
        success = self.stable_steps >= self.settle_steps
        failed = self.cube_position[2] < -0.025 or np.linalg.norm(self.cube_position[:2]) > 0.75
        potential = self._potential()
        if success or failed:
            potential = 0.0  # Absorbing terminal state in potential-based shaping.
        reward = self.shaping_gamma * potential - self._previous_potential
        reward += -0.01 - 0.001 * float(np.square(action).sum()) + 20.0 * success - 5.0 * failed
        self._previous_potential = potential
        terminated = bool(success or failed)
        truncated = bool(self.episode_steps >= self.max_episode_steps and not terminated)
        return self._observation(), float(reward), terminated, truncated, self._info()

    def render(self):
        if self.render_mode is None:
            return None
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.visual_options)
        return self.renderer.render().copy()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
