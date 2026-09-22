"""Math and contract tests for bounded Piper policy actions."""

import math
import unittest

import torch

from scripts.piper_action_distribution import TanhGaussianDistribution


class TanhGaussianDistributionTests(unittest.TestCase):
    def test_actions_and_deterministic_export_are_bounded_and_equal(self):
        distribution = TanhGaussianDistribution(3, init_std=0.7)
        mean = torch.tensor([[0.0, 20.0, -20.0]])
        distribution.update(mean)
        action = distribution.sample()
        self.assertTrue(torch.all(action.abs() <= 1.0))
        deterministic = distribution.deterministic_output(mean)
        exported = distribution.as_deterministic_output_module()(mean)
        self.assertTrue(torch.all(deterministic.abs() <= 1.0))
        torch.testing.assert_close(deterministic, exported)
        self.assertTrue(torch.isfinite(distribution.log_prob(action)).all())

    def test_log_prob_matches_change_of_variables_and_kl_is_zero(self):
        distribution = TanhGaussianDistribution(2, init_std=0.8)
        mean = torch.tensor([[0.2, -0.4]], dtype=torch.float64)
        distribution.update(mean)
        action = torch.tensor([[0.3, -0.7]], dtype=torch.float64)
        latent = torch.atanh(action)
        normal = torch.distributions.Normal(mean, torch.full_like(mean, 0.8))
        logdet = 2.0 * (
            math.log(2.0) - latent - torch.nn.functional.softplus(-2.0 * latent)
        )
        expected = (normal.log_prob(latent) - logdet).sum(dim=-1)
        torch.testing.assert_close(distribution.log_prob(action), expected)
        old_params = distribution.params
        distribution.update(mean)
        torch.testing.assert_close(
            distribution.kl_divergence(old_params, distribution.params),
            torch.zeros(1, dtype=torch.float64),
        )

        shifted = (mean + 0.6, torch.full_like(mean, 1.1))
        expected_kl = torch.distributions.kl_divergence(
            torch.distributions.Normal(*old_params),
            torch.distributions.Normal(*shifted),
        ).sum(dim=-1)
        torch.testing.assert_close(distribution.kl_divergence(old_params, shifted), expected_kl)

    def test_same_parameters_have_exact_logprob_ratio_one_near_bounds(self):
        distribution = TanhGaussianDistribution(3, init_std=0.2)
        mean = torch.tensor([[8.0, -8.0, 0.3]], dtype=torch.float64)
        distribution.update(mean)
        torch.manual_seed(7)
        actions = distribution.sample()
        old_log_prob = distribution.log_prob(actions)
        old_params = distribution.params
        distribution.update(mean)
        new_log_prob = distribution.log_prob(actions)
        torch.testing.assert_close(torch.exp(new_log_prob - old_log_prob), torch.ones(1, dtype=torch.float64))
        torch.testing.assert_close(
            distribution.kl_divergence(old_params, distribution.params),
            torch.zeros(1, dtype=torch.float64),
        )

    def test_transformed_entropy_gradient_points_away_from_saturation(self):
        grads = []
        entropies = []
        for value in (0.0, 5.0, -5.0):
            distribution = TanhGaussianDistribution(1, init_std=0.2)
            mean = torch.tensor([[value]], dtype=torch.float64, requires_grad=True)
            distribution.update(mean)
            entropy = distribution.entropy.sum()
            entropy.backward()
            grads.append(float(mean.grad.item()))
            entropies.append(float(entropy.detach().item()))
        self.assertAlmostEqual(grads[0], 0.0, delta=1e-8)
        self.assertLess(grads[1], 0.0)
        self.assertGreater(grads[2], 0.0)
        self.assertGreater(entropies[0], entropies[1])
        self.assertGreater(entropies[0], entropies[2])

    def test_entropy_is_deterministic_without_sampling(self):
        distribution = TanhGaussianDistribution(2, init_std=0.8)
        distribution.update(torch.tensor([[0.4, -1.2]], dtype=torch.float64))
        first = distribution.entropy
        torch.manual_seed(9876)
        second = distribution.entropy
        torch.testing.assert_close(first, second)

    def test_entropy_and_moments_match_monte_carlo_reference(self):
        distribution = TanhGaussianDistribution(1, init_std=0.8)
        mean = torch.tensor([[0.7]], dtype=torch.float64)
        distribution.update(mean)
        bounded_mean = float(distribution.mean.item())
        bounded_std = float(distribution.std.item())
        bounded_entropy = float(distribution.entropy.item())

        torch.manual_seed(11)
        latent = mean + 0.8 * torch.randn(300_000, 1, dtype=torch.float64)
        actions = torch.tanh(latent)
        monte_carlo_entropy = float((-distribution.log_prob(actions).mean()).item())
        torch.testing.assert_close(
            torch.tensor(bounded_mean, dtype=torch.float64), actions.mean(), rtol=0.0, atol=0.004
        )
        torch.testing.assert_close(
            torch.tensor(bounded_std, dtype=torch.float64), actions.std(unbiased=False), rtol=0.0, atol=0.004
        )
        self.assertAlmostEqual(bounded_entropy, monte_carlo_entropy, delta=0.01)

    def test_bounded_std_is_at_most_one_and_vanishes_near_limits(self):
        distribution = TanhGaussianDistribution(1, init_std=0.8)
        distribution.update(torch.tensor([[0.0]], dtype=torch.float64))
        center_std = distribution.std
        distribution.update(torch.tensor([[10.0]], dtype=torch.float64))
        positive_limit_std = distribution.std
        distribution.update(torch.tensor([[-10.0]], dtype=torch.float64))
        negative_limit_std = distribution.std
        self.assertLessEqual(float(center_std.detach()), 1.0 + 1e-12)
        self.assertLess(float(positive_limit_std.detach()), 1e-6)
        self.assertLess(float(negative_limit_std.detach()), 1e-6)

    def test_extreme_means_and_positive_scale_remain_finite(self):
        distribution = TanhGaussianDistribution(2, init_std=1.0)
        mean = torch.tensor([[100.0, -100.0]], dtype=torch.float64, requires_grad=True)
        distribution.update(mean)
        values = torch.cat(
            (distribution.mean, distribution.std, distribution.entropy.unsqueeze(1)), dim=1
        )
        loss = values.square().sum()
        loss.backward()
        self.assertTrue(torch.isfinite(values).all())
        self.assertTrue(torch.isfinite(mean.grad).all())
        self.assertTrue(torch.isfinite(distribution.params[1]).all())
        self.assertTrue((distribution.params[1] > 0.0).all())

        for std_type in ("log", "scalar"):
            positive = TanhGaussianDistribution(2, init_std=1.0, std_type=std_type)
            if std_type == "log":
                positive.log_std_param.data.fill_(-100.0)
            else:
                positive.std_param.data.fill_(-100.0)
            positive.update(torch.zeros(1, 2))
            self.assertTrue(torch.isfinite(positive.params[1]).all())
            self.assertTrue((positive.params[1] > 0.0).all())
        large = TanhGaussianDistribution(2, init_std=1.0)
        large.log_std_param.data.fill_(100.0)
        large.update(torch.zeros(1, 2))
        self.assertTrue(torch.isfinite(large.params[1]).all())

    def test_invalid_initialization_is_rejected(self):
        default = TanhGaussianDistribution(1)
        self.assertEqual(default.std_type, "log")
        default.update(torch.zeros(1, 1))
        self.assertTrue((default.params[1] > 0.0).all())
        with self.assertRaises(ValueError):
            TanhGaussianDistribution(1, init_std=0.0)
        with self.assertRaises(ValueError):
            TanhGaussianDistribution(1, eps=0.5)
        with self.assertRaises(ValueError):
            TanhGaussianDistribution(1, scale_eps=0.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
