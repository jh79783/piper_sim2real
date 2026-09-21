#!/usr/bin/env python3
"""Train PPO in parallel MuJoCo environments with a live 2x2 grid and TensorBoard."""

import argparse
from collections import deque
from datetime import datetime
from functools import partial
import json
from pathlib import Path
import signal
import time

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("reacher", "piper_pick_place"), default="reacher")
    parser.add_argument("--steps", type=positive_int, default=None,
                        help="total transitions: default 40000 for Reacher, 1000000 for Piper")
    parser.add_argument("--n-envs", type=positive_int, default=4,
                        help="parallel environments; at most 4 with the 2x2 viewer")
    parser.add_argument("--fps", type=positive_int, default=15,
                        help="maximum preview refresh rate, not simulation speed")
    parser.add_argument("--headless", action="store_true", help="disable rendering entirely")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=positive_int, default=5000,
                        help="evaluate after this many total transitions")
    parser.add_argument("--eval-episodes", type=positive_int, default=5)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--start-mode", choices=("above_cube", "home"), default="above_cube",
                        help="Piper: start near the cube for learning, or at a fixed home pose")
    parser.add_argument("--resume", type=Path, help="continue a trusted PPO model.zip from this task")
    args = parser.parse_args(argv)
    if not args.headless and args.n_envs > 4:
        parser.error("the 2x2 viewer supports 1 to 4 environments; use --headless for more")
    if args.steps is None:
        args.steps = 40_000 if args.task == "reacher" else 1_000_000
    if args.output_dir is None:
        args.output_dir = Path("runs/reacher_parallel" if args.task == "reacher" else "runs/piper_pick_place")
    return args


def create_task_env(task, render_mode, start_mode):
    if task == "piper_pick_place":
        from scripts.piper_pick_place_env import PiperPickPlaceEnv
        return PiperPickPlaceEnv(render_mode=render_mode, start_mode=start_mode)
    return gym.make("Reacher-v5", render_mode=render_mode, width=480, height=360)


def make_env(rank, seed, headless, task="reacher", start_mode="above_cube"):
    # Only the parent handles Ctrl+C, so it can drain pipes and close workers cleanly.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.set_num_threads(1)
    env = create_task_env(task, None if headless else "rgb_array", start_mode)
    env.action_space.seed(seed + rank)
    return Monitor(env)


def make_grid(frames):
    """Top-left, top-right, bottom-left, bottom-right; unused cells stay black."""
    if not 1 <= len(frames) <= 4:
        raise ValueError("expected 1 to 4 RGB frames")
    first = np.asarray(frames[0])
    if first.ndim != 3 or first.shape[2] != 3 or first.dtype != np.uint8:
        raise ValueError("expected uint8 RGB frames")
    height, width, _ = first.shape
    grid = np.zeros((height * 2, width * 2, 3), dtype=np.uint8)
    for index, frame in enumerate(frames):
        frame = np.asarray(frame)
        if frame.shape != first.shape or frame.dtype != np.uint8:
            raise ValueError("all RGB frames must have the same shape and dtype")
        row, col = divmod(index, 2)
        grid[row * height:(row + 1) * height, col * width:(col + 1) * width] = frame
    return grid


class GridWindow:
    """Display worker RGB frames using MuJoCo's pixel drawing API, without OpenCV."""

    def __init__(self, title="Reacher PPO"):
        import glfw
        import mujoco

        self.glfw, self.mujoco = glfw, mujoco
        self.title = title
        self.window = None
        self.context = None
        self.frames_drawn = 0
        self.stop_requested = False
        if not glfw.init():
            raise RuntimeError("GLFW initialization failed: check DISPLAY and the WSLg X11 mount")
        try:
            glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
            self.window = glfw.create_window(960, 720, f"{self.title} | 2x2 | Esc: save and exit", None, None)
            if not self.window:
                raise RuntimeError("Could not open the WSLg preview window")
            glfw.make_context_current(self.window)
            glfw.swap_interval(0)
            # This empty model only provides a drawing context, not another simulation.
            self.display_model = mujoco.MjModel.from_xml_string("<mujoco/>")
            self.context = mujoco.MjrContext(self.display_model, mujoco.mjtFontScale.mjFONTSCALE_100)
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.context)
        except BaseException:
            self.close()
            raise

    def poll(self):
        self.glfw.poll_events()
        self.stop_requested = bool(
            self.glfw.window_should_close(self.window)
            or self.glfw.get_key(self.window, self.glfw.KEY_ESCAPE) == self.glfw.PRESS
        )
        return not self.stop_requested

    def draw(self, frames, steps, device):
        grid = make_grid(frames)
        glfw, mj = self.glfw, self.mujoco
        glfw.make_context_current(self.window)
        width, height = glfw.get_framebuffer_size(self.window)
        if width <= 0 or height <= 0:  # Minimized window.
            return
        # Accommodate resizing and WSLg/Windows HiDPI framebuffer sizes.
        rows = np.arange(height) * grid.shape[0] // height
        cols = np.arange(width) * grid.shape[1] // width
        pixels = np.ascontiguousarray(grid[rows[:, None], cols[None, :]][::-1])
        mj.mjr_drawPixels(pixels.ravel(), None, mj.MjrRect(0, 0, width, height), self.context)
        for index in range(4):
            row, col = divmod(index, 2)
            left, right = col * width // 2, (col + 1) * width // 2
            bottom, top = (1 - row) * height // 2, (2 - row) * height // 2
            rect = mj.MjrRect(left, bottom, right - left, top - bottom)
            label = f"Env {index + 1}" if index < len(frames) else "Unused"
            mj.mjr_overlay(mj.mjtFontScale.mjFONTSCALE_100, mj.mjtGridPos.mjGRID_TOPLEFT,
                           rect, label, "", self.context)
        glfw.set_window_title(self.window, f"{self.title} | {len(frames)} envs | {device} | "
                              f"{steps:,} total steps | Esc: save and exit")
        glfw.swap_buffers(self.window)
        self.frames_drawn += 1

    def close(self):
        if self.window:
            self.glfw.make_context_current(self.window)
        if self.context:
            self.context.free()
            self.context = None
        if self.window:
            self.glfw.destroy_window(self.window)
            self.window = None
        self.glfw.terminate()


class GridCallback(BaseCallback):
    def __init__(self, window, fps):
        super().__init__()
        self.window = window
        self.interval = 1.0 / fps
        self.last_draw = float("-inf")
        self.last_grid = None

    def _draw(self):
        frames = self.training_env.get_images()
        self.window.draw(frames, self.num_timesteps, str(self.model.device))
        self.last_grid = make_grid(frames)
        self.last_draw = time.monotonic()

    def _on_training_start(self):
        self._draw()

    def _on_step(self):
        if not self.window.poll():
            return False
        if time.monotonic() - self.last_draw >= self.interval:
            self._draw()
        return True


class FixedSeedEvalCallback(EvalCallback):
    """Compare deterministic policies on the same held-out sequence, including step 0."""

    def __init__(self, *args, eval_seed, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_seed = eval_seed

    def _on_training_start(self):
        self._on_step()

    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.eval_env.seed(self.eval_seed)
        return super()._on_step()


class TaskMetricsCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episodes = deque(maxlen=100)

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done and "has_lifted" in info:
                self.episodes.append(info)
        return True

    def _on_rollout_end(self):
        if self.episodes:
            for key, label in (("is_success", "success_rate"), ("has_lifted", "lift_rate"),
                               ("goal_distance", "final_goal_distance")):
                self.logger.record(f"task/{label}", np.mean([info[key] for info in self.episodes]))


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(1)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Run with --device nvidia.com/gpu=all in Podman; "
                           "or explicitly select --device cpu for a CPU-only test.")

    # A unique directory preserves earlier runs, models and TensorBoard logs.
    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    config = vars(args).copy()
    config["algorithm"] = "PPO"
    config["n_steps_per_env"] = 256
    config["output_dir"] = str(args.output_dir)
    config["resume"] = str(args.resume) if args.resume is not None else None
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Run directory: {run_dir.resolve()}", flush=True)
    print(f"{args.n_envs} CPU environments -> one PPO policy on {args.device}", flush=True)

    env = eval_env = window = model = callback = None
    status = "error"
    try:
        env = SubprocVecEnv(
            [partial(make_env, rank, args.seed, args.headless, args.task, args.start_mode)
             for rank in range(args.n_envs)],
            start_method="spawn",
        )
        env.seed(args.seed)
        if not args.headless:
            window = GridWindow("Piper Pick & Place PPO" if args.task == "piper_pick_place" else "Reacher PPO")
            callback = GridCallback(window, args.fps)
        eval_env = Monitor(create_task_env(args.task, None, args.start_mode))
        evaluation = FixedSeedEvalCallback(
            eval_env, eval_seed=args.seed + 10_000,
            eval_freq=max(1, args.eval_every // args.n_envs),
            n_eval_episodes=args.eval_episodes,
            deterministic=True, render=False,
            best_model_save_path=str(run_dir / "best"),
            log_path=str(run_dir / "evaluation"),
        )
        if args.resume is None:
            model = PPO(
                "MlpPolicy", env, device=args.device, seed=args.seed,
                n_steps=256, batch_size=256, n_epochs=10, gamma=0.99,
                tensorboard_log=str(run_dir / "tensorboard"), verbose=1,
            )
        else:
            model = PPO.load(args.resume, env=env, device=args.device,
                             tensorboard_log=str(run_dir / "tensorboard"))
        callbacks = [evaluation, TaskMetricsCallback()]
        if callback is not None:
            callbacks.insert(0, callback)
        model.learn(total_timesteps=args.steps, callback=callbacks, log_interval=1,
                    tb_log_name="PPO", reset_num_timesteps=args.resume is None)
        status = "stopped" if window and window.stop_requested else "completed"
    except KeyboardInterrupt:
        status = "interrupted"
        print("Interrupted; saving the current model.", flush=True)
    finally:
        # Cleanup still runs if model saving fails (e.g. a full disk).
        try:
            if model is not None:
                if hasattr(model, "_logger") and model.logger.name_to_value:
                    model.logger.dump(step=model.num_timesteps)
                model.save(run_dir / "model")
                if callback is not None and callback.last_grid is not None:
                    import imageio.v3 as iio
                    iio.imwrite(run_dir / "preview.png", callback.last_grid)
                summary = {
                    "status": status,
                    "algorithm": "PPO",
                    "task": args.task,
                    "total_timesteps": model.num_timesteps,
                    "optimization_epochs": model._n_updates,
                    "device": str(model.device),
                    "n_envs": args.n_envs,
                    "preview_frames": window.frames_drawn if window else 0,
                }
                (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
                print(f"Saved: {run_dir / 'model.zip'} ({status})", flush=True)
        finally:
            try:
                if env is not None:
                    env.close()
            finally:
                try:
                    if eval_env is not None:
                        eval_env.close()
                finally:
                    if window is not None:
                        window.close()


if __name__ == "__main__":
    main()
