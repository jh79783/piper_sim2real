"""Physics sanity check: a scripted pick-and-place through the policy action API.

The script drives the environment only through the normalized incremental
action contract (no state teleporting): descend over the cube, close, lift,
carry above the goal, lower, open and retreat.  It reports lift/place/success
rates and fails loudly on shape or finiteness errors.  It is a simulator
validation, not a policy benchmark.

    bash scripts/run_isaac.sh python scripts/isaac/check_env.py --num_envs 64
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--randomize", action="store_true", help="Keep DR and sensor noise on.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import time  # noqa: E402

import torch  # noqa: E402

from isaac_piper.pick_place_env import NUM_OBS, PiperPickPlaceEnv  # noqa: E402
from isaac_piper.pick_place_env_cfg import CUBE_HALF_SIZE, PiperPickPlaceEnvCfg  # noqa: E402


def joint_action(env, target_xyz, gripper_cmd):
    """Action that moves the arm target toward the IK solution of target_xyz."""
    q, _ = env.kin.solve_ik(target_xyz, env.arm_target, iterations=20)
    arm = ((q - env.arm_target) / env.cfg.joint_target_rate).clamp(-1, 1)
    return torch.cat((arm, gripper_cmd[:, None]), dim=1)


def main():
    cfg = PiperPickPlaceEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = 0
    cfg.start_mode = "above_cube"
    cfg.curriculum = False
    cfg.randomization.domain_randomization = args.randomize
    cfg.randomization.sensor_noise = args.randomize
    env = PiperPickPlaceEnv(cfg)
    obs, _ = env.reset()
    assert obs["policy"].shape == (args.num_envs, NUM_OBS), obs["policy"].shape
    assert obs["critic"].shape == (args.num_envs, NUM_OBS), obs["critic"].shape

    n = env.num_envs
    ones = torch.ones(n, device=env.device)
    lifted = torch.zeros(n, dtype=torch.bool, device=env.device)
    placed = torch.zeros_like(lifted)
    succeeded = torch.zeros_like(lifted)
    ended = torch.zeros_like(lifted)
    grasp_z = CUBE_HALF_SIZE + 0.005
    carry_z = 0.12
    # (ticks, target builder, gripper command)
    phases = [
        (40, lambda c, g: torch.cat((c[:, :2], torch.full_like(c[:, :1], 0.10)), 1), ones),
        (60, lambda c, g: torch.cat((c[:, :2], torch.full_like(c[:, :1], grasp_z)), 1), ones),
        (30, None, -ones),
        (60, lambda c, g: torch.cat((c[:, :2], torch.full_like(c[:, :1], carry_z)), 1), -ones),
        (90, lambda c, g: torch.cat((g[:, :2], torch.full_like(c[:, :1], carry_z)), 1), -ones),
        (60, lambda c, g: torch.cat((g[:, :2], torch.full_like(c[:, :1], grasp_z + 0.002)), 1), -ones),
        (25, None, ones),
        (60, lambda c, g: torch.cat((g[:, :2], torch.full_like(c[:, :1], 0.12)), 1), ones),
    ]
    start = time.perf_counter()
    ticks = 0
    for duration, target_fn, grip in phases:
        hold = None
        for _ in range(duration):
            cube = env._cube_pos()
            if target_fn is None:
                if hold is None:
                    hold = env._tcp().clone()
                action = joint_action(env, hold, grip)
            else:
                action = joint_action(env, target_fn(cube, env.goal_position), grip)
            obs, reward, terminated, truncated, extras = env.step(action)
            ticks += 1
            for key in ("policy", "critic"):
                if not torch.isfinite(obs[key]).all():
                    raise FloatingPointError(f"non-finite {key} observation")
            if not torch.isfinite(reward).all():
                raise FloatingPointError("non-finite reward")
            live = ~ended
            lifted |= env.has_lifted & live
            placed |= env.has_placed & live
            succeeded |= env._step["success"] & live
            ended |= terminated | truncated
    elapsed = time.perf_counter() - start
    print(
        f"[check] envs={n} ticks={ticks} lift={lifted.float().mean():.3f} "
        f"placed={placed.float().mean():.3f} success={succeeded.float().mean():.3f} "
        f"env_steps_per_s={n * ticks / elapsed:,.0f}"
    )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
