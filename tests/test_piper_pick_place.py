"""Headless physics checks for the contact-based Piper task.

The oracle in this file is intentionally a validation aid, not a policy claim.
It sends normal Cartesian actions through :meth:`PiperPickPlaceEnv.step` and
therefore exercises the same position actuators and contacts as training.
"""

import numpy as np
import unittest
from types import SimpleNamespace

from scripts.piper_pick_place_env import PiperPickPlaceEnv


def _action_towards(env, target, gripper):
    """Convert a Cartesian target into one bounded environment action."""

    target = np.asarray(target, dtype=np.float64)
    arm_action = np.clip(
        (target - env.target_position) / env.action_scale,
        -1.0,
        1.0,
    )
    grip_action = np.clip(
        (gripper - env.gripper_target) / 0.008,
        -1.0,
        1.0,
    )
    return np.r_[arm_action, grip_action]


def _drive(env, target, gripper, steps):
    """Drive one phase, stopping if the task terminates."""

    trace = []
    for _ in range(steps):
        result = env.step(_action_towards(env, target, gripper))
        trace.append((env._finger_contacts().copy(), env.cube_position.copy(), result[4]))
        if result[2] or result[3]:
            break
    return trace


def _oracle(env):
    """Run a deterministic physical pick/place sequence from above-cube reset."""

    cube_xy = env.cube_position[:2].copy()
    goal_xy = env.goal_position[:2].copy()
    phases = (
        (np.r_[cube_xy, 0.026], 0.035, 40),  # descend with the fingers open
        (np.r_[cube_xy, 0.026], 0.0, 30),  # close around the cube
        (np.r_[cube_xy, 0.12], 0.0, 50),  # lift while closed
        (np.r_[goal_xy, 0.12], 0.0, 60),  # carry above the goal
        (np.r_[goal_xy, 0.026], 0.0, 40),  # lower onto the table
        (np.r_[goal_xy, 0.026], 0.035, 30),  # release
        (np.r_[goal_xy, 0.12], 0.035, 40),  # retreat (usually terminal first)
    )
    trace = []
    for target, gripper, steps in phases:
        trace.extend(_drive(env, target, gripper, steps))
        if trace[-1][2]["is_success"] or env.episode_steps >= env.max_episode_steps:
            break
    return trace


class PiperPickPlaceTests(unittest.TestCase):
    def test_reset_randomizes_cube_and_goal_reproducibly(self):
        first = PiperPickPlaceEnv()
        second = PiperPickPlaceEnv()
        samples = []
        try:
            for seed in range(8):
                first.reset(seed=seed)
                second.reset(seed=seed)
                np.testing.assert_allclose(first.cube_position, second.cube_position)
                np.testing.assert_allclose(first.goal_position, second.goal_position)
                samples.append(np.r_[first.cube_position[:2], first.goal_position[:2]])
                self.assertGreater(np.linalg.norm(first.cube_position[:2] - first.goal_position[:2]), 0.13)
            samples = np.asarray(samples)
            self.assertGreater(np.unique(samples[:, :2], axis=0).shape[0], 1)
            self.assertGreater(np.unique(samples[:, 2:], axis=0).shape[0], 1)
            # The two samplers are independent, not one fixed offset copied per episode.
            self.assertGreater(np.unique(samples[:, 2:] - samples[:, :2], axis=0).shape[0], 1)
        finally:
            first.close()
            second.close()

    def test_observation_and_action_state_remain_finite(self):
        env = PiperPickPlaceEnv()
        try:
            observation, info = env.reset(seed=3)
            self.assertEqual(observation.shape, (59,))
            self.assertEqual(observation.dtype, np.float32)
            self.assertTrue(np.isfinite(observation).all())
            self.assertFalse(info["is_success"])
            for action in (np.zeros(4), np.ones(4), -np.ones(4)):
                observation, reward, terminated, truncated, info = env.step(action)
                self.assertTrue(np.isfinite(observation).all())
                self.assertTrue(np.isfinite(reward))
                self.assertFalse(terminated or truncated)
                self.assertTrue(np.isfinite(env.data.qpos).all())
                self.assertTrue(np.isfinite(env.data.qvel).all())
                self.assertTrue(np.isfinite(info["ik_error"]))
        finally:
            env.close()

    def test_contact_oracle_physically_picks_lifts_and_places(self):
        # Three layouts cover positive/negative y and a near-minimum allowed
        # cube-goal separation while keeping this regression test quick.
        for seed in (0, 7, 19):
            env = PiperPickPlaceEnv()
            try:
                env.reset(seed=seed)
                trace = _oracle(env)
                self.assertTrue(trace, "oracle did not advance")
                had_bilateral_contact = any(contacts.all() for contacts, _, _ in trace)
                had_lift = any(
                    contacts.all() and cube[2] > env.cube_half_size + 0.05
                    for contacts, cube, _ in trace
                )
                self.assertTrue(had_bilateral_contact)
                self.assertTrue(had_lift)
                info = env._info()
                self.assertTrue(info["is_success"])
                self.assertTrue(info["has_lifted"])
                self.assertTrue(info["inside_goal"])
                self.assertTrue(info["released"])
                self.assertTrue(info["on_table"])
                self.assertTrue(info["object_still"])
                self.assertFalse(info["recontacted"])
                self.assertEqual(info["recontact_penalty"], 0.0)
            finally:
                env.close()

    def test_release_latch_and_recontact_penalty_semantics(self):
        env = PiperPickPlaceEnv()
        try:
            env.reset(seed=0)
            link6_geom = int(np.flatnonzero(
                self._body_geom_mask(env, "link6")
            )[0])
            link7_geom = int(np.flatnonzero(
                self._body_geom_mask(env, "link7")
            )[0])
            table_geom = int(env.model.geom("table").id)
            fake = SimpleNamespace(
                data=SimpleNamespace(contact=[
                    SimpleNamespace(geom1=env.cube_geom, geom2=link6_geom, dist=0.0),
                ]),
                cube_geom=env.cube_geom,
                robot_geoms=env.robot_geoms,
            )
            self.assertTrue(PiperPickPlaceEnv._robot_cube_contact(fake))
            fake.data.contact = [
                SimpleNamespace(geom1=link7_geom, geom2=env.cube_geom, dist=0.0),
            ]
            self.assertTrue(PiperPickPlaceEnv._robot_cube_contact(fake))
            fake.data.contact = [
                SimpleNamespace(geom1=table_geom, geom2=env.cube_geom, dist=0.0),
            ]
            self.assertFalse(PiperPickPlaceEnv._robot_cube_contact(fake))
            fake.data.contact = [
                SimpleNamespace(geom1=env.cube_geom, geom2=link6_geom, dist=0.002),
            ]
            self.assertFalse(PiperPickPlaceEnv._robot_cube_contact(fake))

            # A contact before a valid release is not charged.
            env.reset(seed=1)
            env._robot_cube_contact = lambda: True
            _, _, _, _, info = env.step(np.zeros(4))
            self.assertFalse(info["recontacted"])
            self.assertEqual(info["recontact_penalty"], 0.0)

            # A release latch needs lift/inside/on-table/open/no-robot-contact;
            # it intentionally does not require the eight-step success settle.
            env.reset(seed=2)
            env.has_lifted = True
            env._placement_state = lambda: (True, True, False, True)
            env._robot_cube_contact = lambda: False
            env.step(np.zeros(4))
            self.assertTrue(env.has_placed)
            self.assertEqual(env.recontact_penalty_total, 0.0)

            # Stability cannot accumulate on a robot-cube contact tick even
            # when all placement geometry/release flags look valid.
            env.reset(seed=5)
            env.has_lifted = True
            env.stable_steps = env.settle_steps - 1
            env._placement_state = lambda: (True, True, True, True)
            env._robot_cube_contact = lambda: True
            _, _, terminated, truncated, info = env.step(np.zeros(4))
            self.assertFalse(terminated or truncated)
            self.assertEqual(env.stable_steps, 0)
            self.assertFalse(info["is_success"])

            env._robot_cube_contact = lambda: False
            for index in range(env.settle_steps):
                _, _, terminated, truncated, info = env.step(np.zeros(4))
                if index < env.settle_steps - 1:
                    self.assertFalse(terminated or truncated)
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertTrue(info["is_success"])

            # Once latched, a recontact costs exactly the configured amount.
            no_contact = PiperPickPlaceEnv()
            with_contact = PiperPickPlaceEnv()
            try:
                no_contact.reset(seed=3)
                with_contact.reset(seed=3)
                no_contact.has_placed = True
                with_contact.has_placed = True
                no_contact._robot_cube_contact = lambda: False
                with_contact._robot_cube_contact = lambda: True
                _, reward_no_contact, _, _, _ = no_contact.step(np.zeros(4))
                _, reward_contact, _, _, info = with_contact.step(np.zeros(4))
                self.assertAlmostEqual(
                    reward_no_contact - reward_contact,
                    PiperPickPlaceEnv.recontact_penalty,
                    places=5,
                )
                self.assertTrue(info["recontacted"])
                self.assertAlmostEqual(info["recontact_penalty"], 0.2, places=6)
            finally:
                no_contact.close()
                with_contact.close()

            # The latch persists after the cube leaves the goal and only reset
            # clears it; this prevents a later arm touch from becoming free.
            env._placement_state = lambda: (False, False, False, True)
            env._robot_cube_contact = lambda: False
            env.step(np.zeros(4))
            self.assertTrue(env.has_placed)
            env.reset(seed=4)
            self.assertFalse(env.has_placed)
        finally:
            env.close()

    @staticmethod
    def _body_geom_mask(env, body_name):
        body_id = env.model.body(body_name).id
        return env.model.geom_bodyid == body_id

    def test_success_rejects_pushing_holding_and_border_overlap(self):
        env = PiperPickPlaceEnv()
        try:
            env.reset(seed=0)

            # Pushing a resting cube into the goal has no lift history and is not
            # a successful placement, even when the final AABB is inside.
            env.goal_position[:2] = env.cube_position[:2]
            env.has_lifted = False
            env.stable_steps = 0
            inside, on_table, still, released = env._placement_state()
            self.assertTrue(inside and on_table and still and released)
            self.assertFalse(env._info()["is_success"])

            # A cube carried into the goal while both finger pads touch it is still
            # a hold, not a release.  The public success state remains false.
            env.reset(seed=0)
            cube_xy = env.cube_position[:2].copy()
            _drive(env, np.r_[cube_xy, 0.026], 0.035, 40)
            _drive(env, np.r_[cube_xy, 0.026], 0.0, 30)
            env.goal_position[:2] = cube_xy
            env.has_lifted = True
            inside, _, _, released = env._placement_state()
            self.assertTrue(inside and not released)
            self.assertFalse(env._info()["is_success"])

            # Geometric overlap is not enough: the complete rotated cube AABB
            # must fit, with the configured two-millimetre safety margin.
            env.reset(seed=0)
            env.goal_position[:2] = env.cube_position[:2]
            env.data.qpos[env.cube_qadr : env.cube_qadr + 3] = [
                env.goal_position[0] + env.goal_half_size[0] - env.cube_half_size + 0.001,
                env.goal_position[1],
                env.cube_half_size,
            ]
            mujoco_forward = __import__("mujoco")
            mujoco_forward.mj_forward(env.model, env.data)
            env.has_lifted = True
            env.stable_steps = 0
            inside, _, _, _ = env._placement_state()
            self.assertFalse(inside)
            self.assertFalse(env._info()["is_success"])
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
