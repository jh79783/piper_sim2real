"""Bounded action distribution for Piper's normalized target contract.

The policy stores a diagonal latent Normal distribution and exposes its
bijective ``tanh`` transform in the environment's ``[-1, 1]`` action box.
The latent parameters are retained for PPO's KL calculation. Entropy and
reported moments are evaluated in the bounded action space with deterministic
Gauss-Hermite quadrature, so entropy remains differentiable at saturation and
does not depend on a fresh Monte-Carlo sample during every PPO minibatch.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

from rsl_rl.modules.distribution import Distribution


class _TanhDeterministicOutput(nn.Module):
    def forward(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return torch.tanh(mlp_output)


class TanhGaussianDistribution(Distribution):
    """A diagonal latent Normal transformed to the open ``[-1, 1]`` box.

    ``mean`` and ``std`` report actual bounded-action moments. The
    deterministic policy output is ``tanh(latent_mean)`` and is therefore a
    deterministic center, rather than the expectation reported by ``mean``.
    ``params`` deliberately remains ``(latent_mean, latent_std)`` because KL
    is invariant under the shared bijective tanh transform.

    ``std_type='log'`` is the preferred parameterization. ``'scalar'`` is
    accepted for old configuration files but uses a positive softplus
    parameter, never an unconstrained Normal scale.
    """

    _GH_ORDER = 64
    _MAX_LOG_STD = 20.0

    def __init__(
        self,
        output_dim: int,
        init_std: float = 1.0,
        std_type: str = "log",
        eps: float = 1e-6,
        scale_eps: float = 1e-6,
    ) -> None:
        super().__init__(output_dim)
        if not math.isfinite(float(init_std)) or float(init_std) <= 0.0:
            raise ValueError("init_std must be finite and positive")
        if std_type not in ("log", "scalar"):
            raise ValueError(f"Unknown std_type {std_type!r}; expected 'log' or 'scalar'")
        if not 0.0 < float(eps) < 0.5:
            raise ValueError("eps must be in (0, 0.5)")
        if not math.isfinite(float(scale_eps)) or float(scale_eps) <= 0.0:
            raise ValueError("scale_eps must be finite and positive")

        self.std_type = std_type
        self.eps = float(eps)
        self.scale_eps = float(scale_eps)
        if std_type == "log":
            self.log_std_param = nn.Parameter(
                torch.full((output_dim,), math.log(float(init_std)), dtype=torch.float32)
            )
        else:
            # Store inverse-softplus(init_std) for backwards-compatible
            # scalar configs while keeping the actual Normal scale positive.
            inverse = float(init_std) + math.log(-math.expm1(-float(init_std)))
            self.std_param = nn.Parameter(
                torch.full((output_dim,), inverse, dtype=torch.float32)
            )

        nodes, weights = np.polynomial.hermite.hermgauss(self._GH_ORDER)
        self.register_buffer(
            "_gh_nodes",
            torch.as_tensor(nodes, dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "_gh_weights",
            torch.as_tensor(weights, dtype=torch.float64),
            persistent=False,
        )
        Normal.set_default_validate_args(False)
        self._distribution: Normal | None = None

    def _latent_std(self, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if self.std_type == "log":
            # Keep exp finite even if an optimizer step overshoots. The
            # lower bound is handled after exp so the stored parameter remains
            # a true log standard deviation.
            log_std = self.log_std_param.clamp(
                min=math.log(self.scale_eps), max=self._MAX_LOG_STD
            )
            scale = torch.exp(log_std)
        else:
            scale = F.softplus(self.std_param)
        return scale.to(device=device, dtype=dtype).clamp_min(self.scale_eps)

    def update(self, mlp_output: torch.Tensor) -> None:
        mean = mlp_output
        std = self._latent_std(dtype=mean.dtype, device=mean.device).expand_as(mean)
        self._distribution = Normal(mean, std)

    def sample(self) -> torch.Tensor:
        if self._distribution is None:
            raise RuntimeError("distribution must be updated before sampling")
        # Reparameterization keeps this component useful in standalone tests
        # and does not affect RSL-RL, which detaches sampled rollout actions.
        return torch.tanh(self._distribution.rsample())

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return torch.tanh(mlp_output)

    def as_deterministic_output_module(self) -> nn.Module:
        return _TanhDeterministicOutput()

    @property
    def input_dim(self) -> int:
        return self.output_dim

    def _quadrature_samples(self) -> torch.Tensor:
        if self._distribution is None:
            raise RuntimeError("distribution must be updated before computing moments")
        mean = self._distribution.mean.unsqueeze(-1)
        std = self._distribution.stddev.unsqueeze(-1)
        nodes = self._gh_nodes.to(device=mean.device, dtype=mean.dtype)
        return mean + math.sqrt(2.0) * std * nodes

    def _quadrature_expectation(self, values: torch.Tensor) -> torch.Tensor:
        weights = self._gh_weights.to(device=values.device, dtype=values.dtype)
        return (values * weights).sum(dim=-1) / math.sqrt(math.pi)

    @staticmethod
    def _latent_log_abs_det_jacobian(latent: torch.Tensor) -> torch.Tensor:
        # log(1 - tanh(z)^2), stable for large positive and negative z.
        return 2.0 * (math.log(2.0) - latent - F.softplus(-2.0 * latent))

    @property
    def mean(self) -> torch.Tensor:
        latent = self._quadrature_samples()
        return self._quadrature_expectation(torch.tanh(latent))

    @property
    def std(self) -> torch.Tensor:
        latent = self._quadrature_samples()
        action = torch.tanh(latent)
        mean = self._quadrature_expectation(action).unsqueeze(-1)
        variance = self._quadrature_expectation((action - mean).square())
        # Keep the square-root derivative finite when all quadrature samples
        # round to the same saturated float value.
        tiny = torch.finfo(variance.dtype).tiny
        return torch.sqrt(variance.clamp_min(tiny))

    @property
    def entropy(self) -> torch.Tensor:
        if self._distribution is None:
            raise RuntimeError("distribution must be updated before reading entropy")
        latent = self._quadrature_samples()
        logdet = self._quadrature_expectation(
            self._latent_log_abs_det_jacobian(latent)
        )
        return self._distribution.entropy().sum(dim=-1) + logdet.sum(dim=-1)

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        if self._distribution is None:
            raise RuntimeError("distribution must be updated before reading params")
        return (self._distribution.mean, self._distribution.stddev)

    @staticmethod
    def _atanh(action: torch.Tensor, eps: float) -> torch.Tensor:
        bounded = action.clamp(-1.0 + eps, 1.0 - eps)
        return 0.5 * (torch.log1p(bounded) - torch.log1p(-bounded))

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        if self._distribution is None:
            raise RuntimeError("distribution must be updated before computing log_prob")
        latent = self._atanh(outputs, self.eps)
        return (
            self._distribution.log_prob(latent)
            - self._latent_log_abs_det_jacobian(latent)
        ).sum(dim=-1)

    def kl_divergence(
        self,
        old_params: tuple[torch.Tensor, ...],
        new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        old_mean, old_std = old_params
        new_mean, new_std = new_params
        old_distribution = Normal(old_mean, old_std)
        new_distribution = Normal(new_mean, new_std)
        # A shared bijective transform leaves KL invariant.
        return torch.distributions.kl_divergence(old_distribution, new_distribution).sum(dim=-1)


__all__ = ["TanhGaussianDistribution"]
