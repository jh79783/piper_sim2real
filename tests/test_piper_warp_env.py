"""CUDA/MuJoCo-Warp contracts and contact-solvability regression checks."""

import unittest

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - host-only test collection
    torch = None

from scripts.piper_warp_env import PiperWarpEnv
from scripts.piper_training_config import PiperTrainingConfig


GPU_TESTS_AVAILABLE = torch is not None and torch.cuda.is_available()


def _cpu(value):
    """Copy a CUDA tensor into a NumPy array for test-side comparisons."""

    return value.detach().cpu().numpy().copy()


def _policy_and_critic(env):
    observations = env.get_observations()
    return observations["policy"], observations["critic"]


def _incremental_target_action(env, destination, aperture):
    """Convert a TCP target to the env's seven incremental-target action.

    The conversion is deliberately test-only. The rollout environment still
    must not run Cartesian IK; this helper only turns the desired joint target
    into one bounded actuator-target increment for the physics oracle.
    """

    current = _cpu(env._qpos[:, env._arm_qadr][0])
    target, residual = env._ik_env.solve_ik(
        np.asarray(destination, dtype=np.float64), current, iterations=80
    )
    if residual > 0.01:
        raise AssertionError(f"test-side reset IK failed: residual={residual:.4f} m")
    previous = _cpu(env._ctrl[:, env._arm_actuators][0])
    action = np.empty((1, env.num_actions), dtype=np.float32)
    action[0, :6] = np.clip((target - previous) / env.joint_target_rate, -1.0, 1.0)
    previous_gripper = float(_cpu(env._ctrl[:, env.gripper_actuator][0]))
    action[0, 6] = np.clip((float(aperture) - previous_gripper) / env.gripper_scale, -1.0, 1.0)
    return torch.as_tensor(action, device=env.device)


@unittest.skipUnless(GPU_TESTS_AVAILABLE, "requires CUDA Torch and the piper-rsl image")
class PiperWarpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Keep the default instance real: DR and actor sensor uncertainty are
        # tested separately from the nominal contact oracle below.
        cls.env = PiperWarpEnv(num_envs=4, device="cuda:0", seed=7)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        # Tests share one CUDA allocation, so restore the curriculum counter
        # and episode-local random state before every assertion.
        self.env.set_training_steps(0)
        self.env.reset(seed=7)

    def test_metadata_timing_and_cuda_observations(self):
        env = self.env
        self.assertEqual(env.num_actions, 7)
        self.assertEqual(env.num_observations, 63)
        self.assertEqual(env.physics_timestep, 0.002)
        self.assertEqual(env.frame_skip, 10)
        self.assertEqual(env.control_hz, 50)
        self.assertEqual(env.max_episode_length, 600)
        self.assertEqual(env.settle_steps, 16)
        self.assertAlmostEqual(env.frame_skip * env.physics_timestep, 0.02)
        self.assertAlmostEqual(env.settle_steps / env.control_hz, 0.32)
        self.assertAlmostEqual(env.shaping_gamma, np.sqrt(0.99), places=7)
        self.assertEqual(env.episode_length_buf.device.type, "cuda")

        policy, critic = _policy_and_critic(env)
        self.assertEqual(tuple(policy.shape), (4, 63))
        self.assertEqual(tuple(critic.shape), (4, 63))
        self.assertEqual(policy.device.type, "cuda")
        self.assertEqual(critic.device.type, "cuda")
        self.assertTrue(torch.isfinite(policy).all().item())
        self.assertTrue(torch.isfinite(critic).all().item())

    def test_default_contract_enables_uncertainty_and_curriculum(self):
        env = self.env
        config = env.training_config
        self.assertTrue(config.domain_randomization)
        self.assertTrue(config.sensor_noise)
        self.assertTrue(config.curriculum)
        self.assertEqual(env.start_mode, "curriculum")
        self.assertEqual(env.training_steps, 0)
        self.assertEqual(env._stage_at_runtime()["step"], 0)

    def test_noisy_policy_clean_critic_and_derived_object_difference(self):
        env = self.env
        env.reset(seed=31)
        policy, critic = _policy_and_critic(env)

        # Default actor noise is real; critic remains a clean observation of
        # the same state and therefore differs without adding dimensions.
        self.assertFalse(torch.equal(policy, critic))
        for observation in (policy, critic):
            cube = observation[:, env.OBS_CUBE]
            tcp = observation[:, env.OBS_TCP]
            derived = observation[:, env.OBS_CUBE_TCP]
            torch.testing.assert_close(derived, cube - tcp, atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(
                observation[:, env.OBS_PREVIOUS_ACTION], env._last_actions, atol=0.0, rtol=0.0
            )

    def test_sensor_delay_selects_controlled_history_and_repeated_reads_are_stable(self):
        config = PiperTrainingConfig(domain_randomization=False, sensor_noise=True, curriculum=False)
        env = PiperWarpEnv(num_envs=1, device="cuda:0", seed=32, training_config=config)
        try:
            clean = env._observation_tensor()
            sentinel = torch.full_like(env._sensor_history, 17.0)
            env._sensor_history.copy_(sentinel)
            env._sensor_delay[:] = 2
            first = env.get_observations()
            second = env.get_observations()
            torch.testing.assert_close(first["policy"], second["policy"], atol=0.0, rtol=0.0)
            for field in env.SENSOR_OBS_SLICES:
                torch.testing.assert_close(first["policy"][:, field], sentinel[:, 2, field])
            for field in env.CURRENT_OBS_SLICES:
                torch.testing.assert_close(first["policy"][:, field], clean[:, field])
            torch.testing.assert_close(first["critic"], clean)
        finally:
            env.close()

    def test_sensor_history_reset_has_no_foreign_episode_sample(self):
        config = PiperTrainingConfig(domain_randomization=False, sensor_noise=True, curriculum=False)
        env = PiperWarpEnv(num_envs=1, device="cuda:0", seed=33, training_config=config)
        try:
            env._sensor_history.fill_(23.0)
            env._sensor_delay[:] = 2
            env.reset(seed=34)
            policy = env.get_observations()["policy"]
            self.assertFalse(torch.any(policy == 23.0).item())
            self.assertFalse(torch.any(env._sensor_history == 23.0).item())
        finally:
            env.close()

    def test_cuda_graph_decimation_is_enabled(self):
        env = self.env
        before = env._wp.to_torch(env._warp_data.time).detach().clone()
        env.step(torch.zeros((env.num_envs, env.num_actions), device=env.device))
        after = env._wp.to_torch(env._warp_data.time).detach().clone()
        torch.testing.assert_close(after - before, torch.full_like(before, 0.02), atol=2e-5, rtol=0.0)
        self.assertTrue(env.cfg.get("cuda_graph_decimation"), env.cfg.get("cuda_graph_error"))

    def test_seed_reproducibility_includes_layout_randomization_and_sensor_state(self):
        env = self.env
        snapshots = []
        for seed in (0, 1, 2):
            env.reset(seed=seed)
            first = {
                "cube": _cpu(env._cube_position()),
                "goal": _cpu(env.goal_xy),
                "mass": _cpu(env.body_mass),
                "inertia": _cpu(env.body_inertia),
                "friction": _cpu(env.geom_friction),
                "zero": _cpu(env._joint_zero_offset),
                "delay": _cpu(env._sensor_delay),
            }
            env.reset(seed=seed)
            second = {
                "cube": _cpu(env._cube_position()),
                "goal": _cpu(env.goal_xy),
                "mass": _cpu(env.body_mass),
                "inertia": _cpu(env.body_inertia),
                "friction": _cpu(env.geom_friction),
                "zero": _cpu(env._joint_zero_offset),
                "delay": _cpu(env._sensor_delay),
            }
            for key in first:
                np.testing.assert_allclose(first[key], second[key], err_msg=key)
            snapshots.append(first)
        self.assertGreater(np.unique(np.concatenate([item["cube"][:, :2] for item in snapshots]), axis=0).shape[0], 1)
        self.assertGreater(np.unique(np.concatenate([item["mass"] for item in snapshots]), axis=0).shape[0], 1)

    def test_action_motion_and_partial_timeout_reset_stay_on_cuda(self):
        env = self.env
        env.reset(seed=9)
        old_qpos = env._qpos[:, env._arm_qadr].detach().clone()
        action = torch.zeros((env.num_envs, env.num_actions), device=env.device)
        obs, reward, done, extras = env.step(action)
        self.assertTrue(torch.isfinite(obs["policy"]).all().item())
        self.assertTrue(torch.isfinite(obs["critic"]).all().item())
        self.assertTrue(torch.isfinite(reward).all().item())
        self.assertFalse(done.any().item())
        self.assertGreater(torch.linalg.vector_norm(env._qpos[:, env._arm_qadr] - old_qpos, dim=1).max().item(), 0.0)

        env.episode_length_buf[1] = env.max_episode_length - 1
        _, _, done, extras = env.step(torch.zeros_like(action))
        self.assertTrue(done[1].item())
        self.assertTrue(extras["time_outs"][1].item())
        self.assertFalse(done[0].item())
        self.assertEqual(env.episode_length_buf[1].item(), 0)
        self.assertIn("log", extras)

    def test_incremental_action_mapping_and_slew_are_joint_space(self):
        env = self.env
        env.reset(seed=11)
        start = env._ctrl[:, env._arm_actuators].detach().clone()
        high = torch.ones((env.num_envs, env.num_actions), device=env.device)
        env._apply_joint_target_action(high)
        command = env._ctrl[:, env._arm_actuators]
        delta = torch.abs(command - start)
        self.assertTrue(torch.all(delta <= env.joint_target_rate + 1e-6).item())
        self.assertTrue(torch.all(command >= env._action_target_low - 1e-6).item())
        self.assertTrue(torch.all(command <= env._action_target_high + 1e-6).item())
        self.assertTrue(torch.all(env._ctrl[:, env.gripper_actuator] <= 0.035).item())

    def test_randomization_is_per_world_positive_and_bounded(self):
        env = self.env
        env.reset(seed=12)
        config = env.training_config
        nominal_mass = _cpu(env._base_body_mass)
        mass = _cpu(env.body_mass)
        moving = np.asarray(env._moving_body_ids_np)
        link_ids = moving[:-1]
        cube_id = int(moving[-1])
        ratio = mass[:, link_ids] / nominal_mass[link_ids][None, :]
        self.assertTrue(np.all(ratio > 0.0))
        self.assertTrue(np.all(ratio >= 1.0 - config.link_mass_fraction - 1e-5))
        self.assertTrue(np.all(ratio <= 1.0 + config.link_mass_fraction + 1e-5))
        cube_ratio = mass[:, cube_id] / nominal_mass[cube_id]
        self.assertTrue(np.all(cube_ratio >= 1.0 - config.cube_mass_fraction - 1e-5))
        self.assertTrue(np.all(cube_ratio <= 1.0 + config.cube_mass_fraction + 1e-5))
        self.assertTrue(np.all(_cpu(env.body_inertia)[:, moving] > 0.0))
        com_delta = _cpu(env.body_ipos)[:, moving] - _cpu(env._base_body_ipos)[moving][None, :]
        self.assertTrue(np.all(np.abs(com_delta[:, :-1]) <= config.link_com_range + 1e-5))
        self.assertTrue(np.all(np.abs(com_delta[:, -1]) <= config.cube_com_range + 1e-5))
        friction = _cpu(env.geom_friction)
        self.assertTrue(np.all(friction[:, env._friction_geom_ids_np] > 0.0))
        self.assertGreater(np.unique(mass[:, link_ids], axis=0).shape[0], 1)

    def test_repeated_randomization_is_centered_on_nominal_parameters(self):
        env = self.env
        env.set_training_steps(96_000)
        nominal_mass = _cpu(env._base_body_mass)
        body_id = int(env._moving_body_ids_np[0])
        cube_id = int(env._moving_body_ids_np[-1])
        link_samples = []
        cube_samples = []
        for seed in range(24):
            env.reset(seed=100 + seed)
            mass = _cpu(env.body_mass)
            link_samples.extend((mass[:, body_id] / nominal_mass[body_id]).tolist())
            cube_samples.extend((mass[:, cube_id] / nominal_mass[cube_id]).tolist())
        self.assertAlmostEqual(float(np.mean(link_samples)), 1.0, delta=0.04)
        self.assertAlmostEqual(float(np.mean(cube_samples)), 1.0, delta=0.08)

    def test_partial_reset_preserves_other_world_physics_and_state(self):
        env = self.env
        env.reset(seed=13)
        before_qpos = _cpu(env._qpos)
        before_mass = _cpu(env.body_mass)
        before_inertia = _cpu(env.body_inertia)
        before_friction = _cpu(env.geom_friction)
        worlds = torch.tensor([1, 3], dtype=torch.long, device=env.device)
        env._reset_worlds(worlds)
        np.testing.assert_allclose(before_qpos[[0, 2]], _cpu(env._qpos)[[0, 2]], atol=1e-5)
        np.testing.assert_allclose(before_mass[[0, 2]], _cpu(env.body_mass)[[0, 2]])
        np.testing.assert_allclose(before_inertia[[0, 2]], _cpu(env.body_inertia)[[0, 2]])
        np.testing.assert_allclose(before_friction[[0, 2]], _cpu(env.geom_friction)[[0, 2]])

    def test_curriculum_boundaries_and_restored_training_steps(self):
        env = self.env
        expected = ((0, 0.2), (32_000, 0.5), (64_000, 0.75), (96_000, 1.0))
        for steps, scale in expected:
            env.set_training_steps(steps)
            self.assertEqual(env.training_steps, steps)
            stage = env._stage_at_runtime()
            self.assertEqual(stage["step"], steps)
            self.assertAlmostEqual(stage["randomization_scale"], scale)
            self.assertEqual(env.cfg["training_steps"], steps)
        env.set_training_steps(64_000)
        env.reset(seed=14)
        self.assertEqual(env.cfg["training_steps"], 64_000)
        final_config = PiperTrainingConfig(
            domain_randomization=False,
            sensor_noise=False,
            curriculum=False,
        )
        final_env = PiperWarpEnv(
            num_envs=1,
            device="cuda:0",
            seed=14,
            training_config=final_config,
        )
        try:
            self.assertEqual(final_env._stage_at_runtime()["step"], 96_000)
            self.assertEqual(final_env._stage_at_runtime()["randomization_scale"], 1.0)
        finally:
            final_env.close()

    def test_training_steps_count_vector_environment_ticks(self):
        env = self.env
        env.set_training_steps(0)
        env.step(torch.zeros((env.num_envs, env.num_actions), device=env.device))
        self.assertEqual(env.training_steps, 1)
        self.assertEqual(env.cfg["training_steps"], 1)

    def test_action_rate_penalty_uses_previous_actions_and_reset_command(self):
        env = PiperWarpEnv(
            num_envs=1,
            device="cuda:0",
            seed=15,
            start_mode="above_cube",
            training_config=PiperTrainingConfig(
                domain_randomization=False,
                sensor_noise=False,
                curriculum=False,
                time_penalty=0.0,
            ),
        )
        try:
            env._step_physics = lambda: None
            env._check_physics_state = lambda: None
            env._finger_contacts = lambda: torch.zeros((1, 2), dtype=torch.bool, device=env.device)
            env._placement_state = lambda: (
                torch.zeros(1, dtype=torch.bool, device=env.device),
                torch.zeros(1, dtype=torch.bool, device=env.device),
                torch.zeros(1, dtype=torch.bool, device=env.device),
                torch.zeros(1, dtype=torch.bool, device=env.device),
            )
            env._robot_cube_contacts = lambda: torch.zeros(1, dtype=torch.bool, device=env.device)
            env._potential = lambda: torch.zeros(1, dtype=torch.float32, device=env.device)
            env.reset(seed=15)
            reset_prev = env._last_actions.detach().clone()
            hold = reset_prev.clone()
            _, hold_reward, _, _ = env.step(hold)
            jump = torch.ones_like(hold)
            _, jump_reward, _, _ = env.step(jump)
            self.assertGreater(hold_reward.item(), jump_reward.item())
            expected = env._stage.action_rate_weight * torch.square(jump - hold).sum()
            self.assertAlmostEqual(
                jump_reward.item(),
                hold_reward.item() - expected.item(),
                places=4,
            )
        finally:
            env.close()

    def test_robot_contact_cannot_start_or_continue_success_stability(self):
        """Reward-v3 requires every stable tick to remain contact-free."""

        env = PiperWarpEnv(
            num_envs=1,
            device="cuda:0",
            seed=16,
            start_mode="above_cube",
            training_config=PiperTrainingConfig(
                domain_randomization=False,
                sensor_noise=False,
                curriculum=False,
                time_penalty=0.0,
            ),
        )
        try:
            env.reset(seed=16)
            env._step_physics = lambda: None
            env._check_physics_state = lambda: None
            env._finger_contacts = lambda: torch.zeros(
                (1, 2), dtype=torch.bool, device=env.device
            )
            env._placement_state = lambda: tuple(
                torch.ones(1, dtype=torch.bool, device=env.device) for _ in range(4)
            )
            env._potential = lambda: torch.zeros(1, dtype=torch.float32, device=env.device)
            env.has_lifted[:] = True
            env.stable_steps[:] = env.settle_steps - 1
            env._robot_cube_contacts = lambda: torch.ones(
                1, dtype=torch.bool, device=env.device
            )

            _, _, done, extras = env.step(torch.zeros((1, env.num_actions), device=env.device))
            self.assertFalse(done.item())
            self.assertEqual(env.stable_steps.item(), 0)
            self.assertFalse(extras["is_success"].item())

            env._robot_cube_contacts = lambda: torch.zeros(
                1, dtype=torch.bool, device=env.device
            )
            for index in range(env.settle_steps):
                _, _, done, extras = env.step(
                    torch.zeros((1, env.num_actions), device=env.device)
                )
                if index < env.settle_steps - 1:
                    self.assertFalse(done.item())
            self.assertTrue(done.item())
            self.assertTrue(extras["is_success"].item())
        finally:
            env.close()

    def test_gpu_contact_oracle_has_real_lift_and_release(self):
        # Keep the nominal oracle separate from the default robust training
        # configuration so contact failures cannot be hidden by noise/DR.
        env = PiperWarpEnv(
            num_envs=1,
            device="cuda:0",
            seed=0,
            start_mode="above_cube",
            training_config=PiperTrainingConfig(
                domain_randomization=False,
                sensor_noise=False,
                curriculum=False,
            ),
        )
        try:
            env.reset(seed=0)
            cube_xy = env._cube_position()[:, :2].detach().clone()
            goal_xy = env.goal_xy.detach().clone()

            def target(xy, z):
                return torch.cat((xy, torch.full((1, 1), z, device=env.device)), dim=1)

            def drive(destination, aperture, count):
                for _ in range(count):
                    action = _incremental_target_action(
                        env,
                        destination[0].detach().cpu().numpy(),
                        aperture,
                    )
                    _, _, done, extras = env.step(action)
                    if done.any().item():
                        return extras
                return None

            # 2x the old 25Hz counts preserves the physical phase durations
            # after switching to the 50Hz policy clock.
            phases = (
                (target(cube_xy, 0.026), 0.035, 80),
                (target(cube_xy, 0.026), 0.0, 60),
                (target(cube_xy, 0.12), 0.0, 100),
                (target(goal_xy, 0.12), 0.0, 120),
                (target(goal_xy, 0.026), 0.0, 80),
                (target(goal_xy, 0.026), 0.035, 60),
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
            self.assertTrue(terminal["has_placed"].item())
            self.assertFalse(terminal["recontacted"].item())
            self.assertEqual(terminal["recontact_penalty"].item(), 0.0)
            self.assertFalse(terminal["time_outs"].item())
            self.assertTrue(terminal["log"]["task/success_rate"].item() == 1.0)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
