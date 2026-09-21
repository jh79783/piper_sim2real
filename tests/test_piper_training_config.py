"""Host-side contract checks for the Piper robustness configuration."""

import unittest

from scripts.piper_training_config import CurriculumStage, PiperTrainingConfig


class PiperTrainingConfigTests(unittest.TestCase):
    def test_default_uncertainty_and_curriculum_are_explicit(self):
        config = PiperTrainingConfig()

        self.assertTrue(config.domain_randomization)
        self.assertTrue(config.sensor_noise)
        self.assertTrue(config.curriculum)
        self.assertEqual(config.physics_timestep, 0.002)
        self.assertEqual(config.control_hz, 50)
        self.assertEqual(config.decimation, 10)
        self.assertEqual(config.episode_steps, 600)
        self.assertEqual(config.success_steps, 16)
        self.assertEqual(config.num_actions, 7)
        self.assertEqual(config.joint_target_rate, 0.035)
        self.assertEqual(config.gripper_target_rate, 0.004)
        self.assertEqual(config.link_mass_fraction, 0.10)
        self.assertEqual(config.cube_mass_fraction, 0.25)
        self.assertEqual(config.inertia_fraction, 0.10)
        self.assertEqual(config.link_com_range, 0.002)
        self.assertEqual(config.cube_com_range, 0.002)
        self.assertEqual(config.friction_fraction, 0.25)
        self.assertEqual(config.joint_position_noise, 0.001)
        self.assertEqual(config.joint_velocity_noise, 0.25)
        self.assertEqual(config.joint_zero_offset, 0.005)
        self.assertEqual(config.max_sensor_delay_steps, 2)

        self.assertEqual(
            [(stage.step, stage.randomization_scale, stage.layout_scale,
              stage.home_probability, stage.action_rate_weight)
             for stage in config.stages],
            [
                (0, 0.20, 0.55, 0.00, 0.001),
                (32_000, 0.50, 0.75, 0.25, 0.002),
                (64_000, 0.75, 0.90, 0.50, 0.005),
                (96_000, 1.00, 1.00, 0.80, 0.010),
            ],
        )

    def test_stage_boundaries_are_vector_step_boundaries(self):
        config = PiperTrainingConfig()

        self.assertEqual(config.stage_at(0), config.stages[0])
        self.assertEqual(config.stage_at(31_999), config.stages[0])
        self.assertEqual(config.stage_at(32_000), config.stages[1])
        self.assertEqual(config.stage_at(63_999), config.stages[1])
        self.assertEqual(config.stage_at(64_000), config.stages[2])
        self.assertEqual(config.stage_at(95_999), config.stages[2])
        self.assertEqual(config.stage_at(96_000), config.stages[3])
        self.assertEqual(config.stage_at(10_000_000), config.stages[-1])

        final_config = PiperTrainingConfig(curriculum=False)
        for step in (0, 1, 32_000, 96_000):
            self.assertEqual(final_config.stage_at(step), final_config.stages[-1])

    def test_invalid_curriculum_is_rejected(self):
        with self.assertRaises(ValueError):
            PiperTrainingConfig(
                stages=(
                    CurriculumStage(0, 0.2, 0.55, 0.0, 0.001),
                    CurriculumStage(0, 0.5, 0.75, 0.25, 0.002),
                )
            )

        with self.assertRaises(ValueError):
            PiperTrainingConfig(
                stages=(CurriculumStage(0, 1.1, 0.55, 0.0, 0.001),)
            )

    def test_serialization_preserves_stage_values(self):
        config = PiperTrainingConfig()
        payload = config.to_dict()

        self.assertEqual(payload["max_sensor_delay_steps"], 2)
        self.assertEqual(payload["stages"][1]["step"], 32_000)
        self.assertEqual(payload["stages"][-1]["action_rate_weight"], 0.01)


if __name__ == "__main__":
    unittest.main()
