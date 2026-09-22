"""CPU-only invariants for incremental Piper actuator-target actions."""

import unittest

import torch

from scripts.piper_warp_env import PiperWarpEnv


class IncrementalActionTests(unittest.TestCase):
    def make_env(self):
        env = object.__new__(PiperWarpEnv)
        env._torch = torch
        env.device = torch.device("cpu")
        env.num_envs = 2
        env.num_actions = 7
        env.joint_target_rate = 0.035
        env.gripper_scale = 0.004
        env._arm_actuators = torch.arange(6, dtype=torch.long)
        env.gripper_actuator = 6
        env._action_target_low = torch.tensor([-1.0, -0.5, -2.0, -1.0, -1.0, -3.0])
        env._action_target_high = torch.tensor([1.0, 2.5, 0.0, 1.0, 1.0, 3.0])
        env._ctrl = torch.tensor(
            [
                [-0.2, 1.2, -1.0, 0.3, 0.75, -0.5, 0.027],
                [0.8, 0.1, -1.8, -0.3, -0.7, 2.8, 0.004],
            ],
            dtype=torch.float32,
        )
        env._qpos = torch.zeros((2, 8), dtype=torch.float32)
        env._arm_ctrl = torch.zeros((2, 6), dtype=torch.float32)
        env.gripper_target = torch.zeros(2, dtype=torch.float32)
        env.last_ik_error = torch.zeros(2, dtype=torch.float32)
        env._last_actions = torch.zeros((2, 7), dtype=torch.float32)
        env._last_action_delta = torch.zeros((2, 7), dtype=torch.float32)
        return env

    def test_zero_preserves_arbitrary_reset_targets_and_qpos(self):
        env = self.make_env()
        before_ctrl = env._ctrl.clone()
        before_qpos = env._qpos.clone()
        actions = torch.zeros((2, 7))

        returned, delta = env._apply_joint_target_action(actions)

        torch.testing.assert_close(returned, actions)
        torch.testing.assert_close(delta, actions)
        torch.testing.assert_close(env._ctrl, before_ctrl)
        torch.testing.assert_close(env._arm_ctrl, before_ctrl[:, :6])
        torch.testing.assert_close(env.gripper_target, before_ctrl[:, 6])
        torch.testing.assert_close(env._qpos, before_qpos)

    def test_full_increment_moves_each_target_by_one_rate(self):
        env = self.make_env()
        before = env._ctrl.clone()

        env._apply_joint_target_action(torch.ones((2, 7)))

        expected_arm = torch.minimum(
            before[:, :6] + env.joint_target_rate,
            env._action_target_high,
        )
        expected_gripper = (before[:, 6] + env.gripper_scale).clamp(0.0, 0.035)
        torch.testing.assert_close(env._ctrl[:, :6], expected_arm)
        torch.testing.assert_close(env._ctrl[:, 6], expected_gripper)

        env._apply_joint_target_action(-torch.ones((2, 7)))
        expected_arm = torch.maximum(
            expected_arm - env.joint_target_rate,
            env._action_target_low,
        )
        expected_gripper = (expected_gripper - env.gripper_scale).clamp(0.0, 0.035)
        torch.testing.assert_close(env._ctrl[:, :6], expected_arm)
        torch.testing.assert_close(env._ctrl[:, 6], expected_gripper)

    def test_incremental_targets_clamp_at_existing_limits(self):
        env = self.make_env()
        env._ctrl[:, :6] = env._action_target_high + 0.001
        env._ctrl[:, 6] = 0.035
        env._apply_joint_target_action(torch.ones((2, 7)))
        torch.testing.assert_close(env._ctrl[:, :6], env._action_target_high.expand(2, -1))
        self.assertTrue(torch.all(env._ctrl[:, 6] <= 0.035).item())

        env._ctrl[:, :6] = env._action_target_low - 0.001
        env._ctrl[:, 6] = 0.0
        env._apply_joint_target_action(-torch.ones((2, 7)))
        torch.testing.assert_close(env._ctrl[:, :6], env._action_target_low.expand(2, -1))
        self.assertTrue(torch.all(env._ctrl[:, 6] >= 0.0).item())

    def test_action_history_is_delta_history_and_semantics_are_versioned(self):
        env = self.make_env()
        actions = torch.tensor([[0.5, -0.5, 0.0, 1.0, -1.0, 0.25, -0.75]]).repeat(2, 1)
        _, delta = env._apply_joint_target_action(actions)
        torch.testing.assert_close(delta, actions)
        torch.testing.assert_close(env._last_actions, actions)
        self.assertEqual(env.action_semantics, "incremental_joint_targets_v1")
        self.assertEqual(env.action_contract, "raw_si_incremental_joint_v1")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
