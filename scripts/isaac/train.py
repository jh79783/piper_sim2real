"""Train the Piper pick-and-place state policy with RSL-RL PPO in Isaac Sim.

Based on Isaac Lab 2.3.2 ``scripts/reinforcement_learning/rsl_rl/train.py``.
Differences: the task defaults to Piper, logs go to ``runs/isaac/<experiment>``,
a resumed run restores the curriculum clock, and the MuJoCo-era toggles
(``--no-curriculum`` ...) map onto env config fields.  Any other env/agent
field can be overridden Hydra-style, e.g. ``env.scene.num_envs=1024``.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Piper-PickPlace-Direct-v0")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--run_name", default=None)
parser.add_argument("--resume", default=None, help="Checkpoint .pt to resume from.")
parser.add_argument("--start-mode", choices=("curriculum", "above_cube", "home"), default=None)
parser.add_argument("--no-curriculum", action="store_true")
parser.add_argument("--no-domain-randomization", action="store_true")
parser.add_argument("--no-sensor-noise", action="store_true")
parser.add_argument("--video", action="store_true", help="Record training rollouts (enables cameras).")
parser.add_argument("--video_length", type=int, default=300)
parser.add_argument("--video_interval", type=int, default=20000)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
if args.video:
    args.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args).app

import os  # noqa: E402
from datetime import datetime  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import isaac_piper  # noqa: E402,F401  (registers the task)
from isaac_piper.paths import PIPER_USD, PROJECT_ROOT  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    if not PIPER_USD.exists():
        raise SystemExit(f"{PIPER_USD} is missing; run `bash scripts/run_isaac.sh convert` first")
    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    if args.seed is not None:
        agent_cfg.seed = args.seed
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    if args.run_name:
        agent_cfg.run_name = args.run_name
    if args.start_mode:
        env_cfg.start_mode = args.start_mode
    env_cfg.curriculum &= not args.no_curriculum
    env_cfg.randomization.domain_randomization &= not args.no_domain_randomization
    env_cfg.randomization.sensor_noise &= not args.no_sensor_noise
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args.device if args.device is not None else env_cfg.sim.device
    agent_cfg.device = env_cfg.sim.device

    log_root = PROJECT_ROOT / "runs/isaac" / agent_cfg.experiment_name
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = str(log_root / log_dir)
    env_cfg.log_dir = log_dir
    print(f"[piper] logging to {log_dir}")

    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
    if args.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args.video_interval == 0,
            "video_length": args.video_length,
            "disable_logger": True,
        }
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)
    if args.resume:
        print(f"[piper] resuming from {args.resume}")
        runner.load(args.resume)
        # The curriculum clock counts vector-env ticks; restore it with the
        # optimizer state so the stage does not restart from zero.
        env.unwrapped.common_step_counter = runner.current_learning_iteration * agent_cfg.num_steps_per_env

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
