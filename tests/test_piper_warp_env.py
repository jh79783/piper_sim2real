"""CUDA/MuJoCo-Warp contract and contact-solvability regression checks."""

import unittest

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - host-only test collection
    torch = None

from scripts.piper_warp_env import PiperWarpEnv


GPU_TESTS_AVAILABLE = torch is not None and torch.cuda.is_available()


@unittest.skipUnless(GPU_TESTS_AVAILABLE, "requires CUDA Torch and the piper-rsl image")
class PiperWarpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = PiperWarpEnv(num_envs=4, device="cuda:0", seed=7)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_metadata_and_cuda_observations(self):
        env = self.env
        self.assertEqual(env.num_actions, 4)
        self.assertEqual(env.num_observations, 58)
        self.assertEqual(env.episode_length_buf.device.type, "cuda")
        obs = env.get_observations()["policy"]
        self.assertEqual(tuple(obs.shape), (4, 58))
        self.assertEqual(obs.device.type, "cuda")
        self.assertTrue(torch.isfinite(obs).all().item())

    def test_cuda_graph_decimation_is_enabled(self):
        env = self.env
        env.step(torch.zeros((env.num_envs, env.num_actions), device=env.device))
        self.assertTrue(env.cfg.get("cuda_graph_decimation"), env.cfg.get("cuda_graph_error"))

    def test_seed_reproducibility_and_independent_layouts(self):
        env = self.env
        layouts = []
        for seed in range(4):
            env.reset(seed=seed)
            first_cube = env._cube_position().detach().cpu().numpy().copy()
            first_goal = env.goal_xy.detach().cpu().numpy().copy()
            env.reset(seed=seed)
            np.testing.assert_allclose(first_cube, env._cube_position().detach().cpu().numpy())
            np.testing.assert_allclose(first_goal, env.goal_xy.detach().cpu().numpy())
            layouts.append(np.c_[first_cube[:, :2], first_goal])
        layouts = np.concatenate(layouts, axis=0)
        self.assertGreater(np.unique(layouts[:, :2], axis=0).shape[0], 1)
        self.assertGreater(np.unique(layouts[:, 2:], axis=0).shape[0], 1)
        self.assertGreater(np.unique(layouts[:, 2:] - layouts[:, :2], axis=0).shape[0], 1)

    def test_action_motion_and_partial_timeout_reset_stay_on_cuda(self):
        env = self.env
        env.reset(seed=9)
        old_tcp = env._tcp_position().detach().clone()
        action = torch.tensor([[0.0, 0.0, 1.0, 0.0]] * env.num_envs, device=env.device)
        obs, reward, done, extras = env.step(action)
        self.assertTrue(torch.isfinite(obs["policy"]).all().item())
        self.assertTrue(torch.isfinite(reward).all().item())
        self.assertFalse(done.any().item())
        self.assertGreater(torch.linalg.vector_norm(env.target_position - old_tcp, dim=1).min().item(), 0.0)

        env.episode_length_buf[1] = env.max_episode_length - 1
        _, _, done, extras = env.step(torch.zeros_like(action))
        self.assertTrue(done[1].item())
        self.assertTrue(extras["time_outs"][1].item())
        self.assertFalse(done[0].item())
        self.assertEqual(env.episode_length_buf[1].item(), 0)
        self.assertIn("log", extras)
        self.assertEqual(extras["log"]["task/success_rate"].numel(), 1)

    def test_gpu_contact_oracle_has_real_lift_and_release(self):
        # One world keeps this physical regression quick while exercising all
        # 20 Warp substeps per action and the actual finger/cube contacts.
        env = PiperWarpEnv(num_envs=1, device="cuda:0", seed=0)
        try:
            env.reset(seed=0)
            cube_xy = env._cube_position()[:, :2].detach().clone()
            goal_xy = env.goal_xy.detach().clone()

            def target(xy, z):
                return torch.cat((xy, torch.full((1, 1), z, device=env.device)), dim=1)

            def drive(destination, aperture, count):
                for _ in range(count):
                    action = torch.cat(
                        (
                            (destination - env.target_position) / env.action_scale,
                            (aperture - env.gripper_target[:, None]) / env.gripper_scale,
                        ),
                        dim=1,
                    ).clamp(-1.0, 1.0)
                    _, _, done, extras = env.step(action)
                    if done.any().item():
                        return extras
                return None

            phases = (
                (target(cube_xy, 0.026), 0.035, 40),
                (target(cube_xy, 0.026), 0.0, 30),
                (target(cube_xy, 0.12), 0.0, 50),
                (target(goal_xy, 0.12), 0.0, 60),
                (target(goal_xy, 0.026), 0.0, 40),
                (target(goal_xy, 0.026), 0.035, 30),
            )
            terminal = None
            for destination, aperture, count in phases:
                terminal = drive(destination, aperture, count)
                if terminal is not None:
                    break
            self.assertIsNotNone(terminal, "GPU contact oracle did not terminate")
            self.assertTrue(terminal["is_success"].item())
            self.assertTrue(terminal["has_lifted"].item())
            self.assertTrue(terminal["inside_goal"].item())
            self.assertTrue(terminal["released"].item())
            self.assertTrue(terminal["on_table"].item())
            self.assertTrue(terminal["object_still"].item())
            self.assertFalse(terminal["time_outs"].item())
            self.assertTrue(terminal["log"]["task/success_rate"].item() == 1.0)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
