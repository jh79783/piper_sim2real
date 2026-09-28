"""RSL-RL PPO settings carried over from the MuJoCo-Warp Piper run.

Network sizes, activation, PPO coefficients and the 50 Hz-adjusted discount
(gamma = sqrt(0.99), lambda = sqrt(0.95)) match ``scripts/train_piper_rsl.py``.
The action distribution is RSL-RL's Gaussian with env-side clamping, because
rsl-rl-lib 3.1.2 (bundled with Isaac Lab 2.3.2) has no pluggable
tanh-squashed distribution.
"""

import math

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PiperPickPlacePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 3000
    save_interval = 100
    experiment_name = "piper_pick_place"
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    clip_actions = None
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="tanh",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=math.sqrt(0.99),
        lam=math.sqrt(0.95),
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
