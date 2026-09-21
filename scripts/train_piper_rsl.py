#!/usr/bin/env python3
"""Train the GPU-native Piper pick-and-place task with RSL-RL PPO.

The Warp environment owns all simulation state and stepping.  This launcher
only builds the RSL-RL policy, drives rollouts on CUDA, optionally copies four
RGB snapshots to the small GLFW grid viewer, and records checkpoints/metrics.
It deliberately imports heavy CUDA/RSL dependencies only inside ``main`` so
that ``--help`` and argument-validation tests also work on the host machine.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any


CHECKPOINT_SCHEMA = 1
DEFAULT_OUTPUT_DIR = Path("runs/piper_pick_place")
DEFAULT_STEPS_PER_ENV = 64
DEFAULT_ITERATIONS = 10_000
DEFAULT_TENSORBOARD_PORT = 6006


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def positive_float(value: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return number


def tensorboard_port(value: str) -> int:
    number = positive_int(value)
    if not 1024 <= number <= 65535:
        raise argparse.ArgumentTypeError("must be between 1024 and 65535")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-envs",
        "--n-envs",
        dest="num_envs",
        type=positive_int,
        default=128,
        help="number of GPU Warp environments (default: 128); first four feed the 2x2 preview",
    )
    parser.add_argument(
        "--iterations",
        type=positive_int,
        default=DEFAULT_ITERATIONS,
        help=f"PPO updates (default: {DEFAULT_ITERATIONS:,})",
    )
    parser.add_argument(
        "--steps-per-env",
        type=positive_int,
        default=DEFAULT_STEPS_PER_ENV,
        help=f"rollout transitions per environment per update (default: {DEFAULT_STEPS_PER_ENV})",
    )
    parser.add_argument("--headless", action="store_true", help="disable the 2x2 preview window")
    parser.add_argument("--fps", type=positive_float, default=10.0, help="maximum preview refresh rate")
    parser.add_argument(
        "--device",
        choices=("cuda",),
        default="cuda",
        help="training device; CUDA Warp/RSL-RL is required (CPU is intentionally unsupported)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--start-mode",
        choices=("above_cube", "home"),
        default="above_cube",
        help="Piper reset pose: near the cube for learning, or fixed home pose",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="resume a compatible RSL-RL .pt checkpoint (SB3 .zip files are incompatible)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--save-interval", type=positive_int, default=100)
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="do not start the owned TensorBoard child process",
    )
    parser.add_argument(
        "--tensorboard-port",
        type=tensorboard_port,
        default=DEFAULT_TENSORBOARD_PORT,
        help=f"host-local TensorBoard port (default: {DEFAULT_TENSORBOARD_PORT})",
    )
    args = parser.parse_args(argv)
    # A window can only display the first four selected environments.  Headless
    # runs remain intentionally unrestricted for GPU throughput experiments.
    if not args.headless and args.num_envs < 1:
        parser.error("--num-envs must be at least one")
    return args


def _terminate_process_group(process, *, timeout: float = 5.0) -> None:
    """Terminate only a child process group owned by this launcher."""

    if process is None:
        return
    try:
        running = process.poll() is None
    except Exception:
        running = True
    if not running:
        return
    pid = getattr(process, "pid", None)
    try:
        if pid and hasattr(os, "killpg"):
            os.killpg(pid, signal.SIGTERM)
        else:
            process.terminate()
    except (ProcessLookupError, OSError):
        pass
    try:
        process.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        if pid and hasattr(os, "killpg"):
            os.killpg(pid, signal.SIGKILL)
        else:
            process.kill()
    except (ProcessLookupError, OSError):
        pass
    try:
        process.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, ProcessLookupError, OSError):
        pass


def start_tensorboard(logdir: Path, port: int):
    """Start one owned TensorBoard process bound to the container interface.

    The Podman wrapper publishes this container port only on host
    ``127.0.0.1``.  No process is reused or killed when the requested host port
    is already occupied; the wrapper rejects that case before Podman starts.
    """

    command = [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir",
        str(logdir.resolve()),
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--reload_interval",
        "5",
    ]
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
    except OSError as exc:
        raise RuntimeError(
            "Could not start TensorBoard. Install the image's TensorBoard package "
            "or pass --no-tensorboard."
        ) from exc

    # Catch missing TensorBoard/import errors promptly, while allowing its
    # normal Python startup/import work a few seconds.
    deadline = time.monotonic() + 5.0
    try:
        while time.monotonic() < deadline:
            return_code = process.poll()
            if return_code is not None:
                _terminate_process_group(process)
                raise RuntimeError(
                    f"TensorBoard exited during startup (code {return_code}) on port {port}; "
                    "use --no-tensorboard or choose another --tensorboard-port."
                )
            time.sleep(0.05)
    except BaseException:
        # ``process`` has not yet been returned to main(), so it must be
        # cleaned here if Ctrl+C/failure arrives during startup polling.
        _terminate_process_group(process)
        raise
    print(f"TensorBoard (Windows): http://localhost:{port}", flush=True)
    return process


def _to_device(value, device):
    if hasattr(value, "to"):
        return value.to(device)
    return value


def _as_obs_tensordict(value, *, device):
    """Normalize environment observations to RSL-RL's TensorDict contract."""

    from tensordict import TensorDict

    if isinstance(value, tuple):
        value = value[0]
    if isinstance(value, TensorDict):
        return value.to(device)
    if isinstance(value, dict):
        tensors = {str(key): _to_device(item, device) for key, item in value.items()}
        if not tensors:
            raise ValueError("Warp environment returned an empty observation mapping")
        first = next(iter(tensors.values()))
        return TensorDict(tensors, batch_size=[first.shape[0]], device=device)
    if not hasattr(value, "shape"):
        raise TypeError("Warp environment observations must be a TensorDict, mapping, or tensor")
    return TensorDict({"policy": _to_device(value, device)}, batch_size=[value.shape[0]], device=device)


def _observation_from_env(env, *, device):
    return _as_obs_tensordict(env.get_observations(), device=device)


def _normalise_extras(extras: Any, env, *, device):
    """Preserve RSL extras and expose task metrics under stable TB names."""

    if extras is None:
        extras = {}
    if not isinstance(extras, dict):
        extras = {"raw": extras}
    else:
        extras = dict(extras)
    log = dict(extras.get("log") or {})
    # PiperWarpEnv emits RSL-RL's ``log`` structure only for environments whose
    # episode just completed. Never derive rates from top-level per-step flags:
    # doing so would turn a state flag into an episode metric and dilute the
    # success/lift rate across all rollout transitions.
    aliases = {
        "success_rate": ("task/success_rate", "is_success", "success_rate"),
        "lift_rate": ("task/lift_rate", "has_lifted", "lift_rate"),
        "final_goal_distance": (
            "task/final_goal_distance",
            "goal_distance",
            "final_goal_distance",
        ),
    }
    for label, candidates in aliases.items():
        if any(f"task/{label}" == key for key in log):
            continue
        found = None
        for key in candidates:
            if key in log:
                found = log[key]
                break
        if found is not None:
            log[f"task/{label}"] = _to_device(found, device)
    if log:
        extras["log"] = log
    else:
        extras.pop("log", None)
    return extras


def _render_frames(env):
    """Get four selected frames without requiring a CPU physics mirror."""

    count = min(4, int(env.num_envs))
    indices = tuple(range(count))
    method = getattr(env, "render_frames", None)
    if method is None:
        raise RuntimeError("PiperWarpEnv must expose render_frames(indices) for the non-headless preview")
    frames = method(indices)
    if frames is None:
        raise RuntimeError("PiperWarpEnv.render_frames(indices) returned no frames")
    return frames


def _make_train_cfg(args, *, num_obs: int, num_actions: int) -> dict:
    """Build the rsl_rl 5.x OnPolicyRunner configuration."""

    del num_obs, num_actions  # dimensions are inferred from TensorDict/env.
    return {
        "seed": args.seed,
        "runner_class_name": "OnPolicyRunner",
        "algorithm_class_name": "PPO",
        "num_steps_per_env": args.steps_per_env,
        "save_interval": args.save_interval,
        "logger": "tensorboard",
        "run_name": "piper_pick_place",
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "actor": {
            "class_name": "rsl_rl.models:MLPModel",
            "hidden_dims": [256, 128, 64],
            "activation": "tanh",
            "obs_normalization": True,
            "distribution_cfg": {
                "class_name": "rsl_rl.modules.distribution:GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "rsl_rl.models:MLPModel",
            "hidden_dims": [256, 128, 64],
            "activation": "tanh",
            "obs_normalization": True,
        },
        "algorithm": {
            "class_name": "rsl_rl.algorithms:PPO",
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "clip_param": 0.2,
            "gamma": 0.99,
            "lam": 0.95,
            "value_loss_coef": 1.0,
            "entropy_coef": 0.01,
            "learning_rate": 3.0e-4,
            "max_grad_norm": 1.0,
            "optimizer": "adam",
            "use_clipped_value_loss": True,
            "schedule": "adaptive",
            "desired_kl": 0.01,
            "normalize_advantage_per_mini_batch": False,
            "rnd_cfg": None,
            "symmetry_cfg": None,
            "share_cnn_encoders": False,
        },
    }


def _checkpoint_schema(
    args,
    env,
    obs,
    *,
    iteration: int,
    train_cfg: dict,
    total_steps: int = 0,
    lifetime_total_steps: int = 0,
) -> dict:
    policy = obs.get("policy")
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "task": "piper_pick_place",
        "algorithm": "rsl_rl.PPO",
        "rsl_rl_config": "v5",
        "num_obs": int(policy.shape[-1]),
        "num_actions": int(env.num_actions),
        "steps_per_env": int(args.steps_per_env),
        "iteration": int(iteration),
        "actor_hidden_dims": list(train_cfg["actor"]["hidden_dims"]),
        "critic_hidden_dims": list(train_cfg["critic"]["hidden_dims"]),
        "actor_activation": str(train_cfg["actor"]["activation"]),
        "critic_activation": str(train_cfg["critic"]["activation"]),
        # This is transitions collected in the run that wrote this file. It
        # lets a resumed summary distinguish new work from lifetime progress.
        "total_steps": int(total_steps),
        "lifetime_total_steps": int(lifetime_total_steps),
    }


def _load_resume_schema(path: Path, args, env, obs, train_cfg: dict, *, torch):
    if path.suffix.lower() != ".pt":
        raise ValueError(f"--resume must point to an RSL-RL .pt checkpoint, not {path.name!r}")
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    schema = payload.get("piper_schema") if isinstance(payload, dict) else None
    if not isinstance(schema, dict):
        raise ValueError("Resume checkpoint has no piper_schema; SB3/legacy .pt files are incompatible")
    expected = _checkpoint_schema(args, env, obs, iteration=0, train_cfg=train_cfg)
    for key in (
        "schema_version",
        "task",
        "algorithm",
        "rsl_rl_config",
        "num_obs",
        "num_actions",
        "actor_hidden_dims",
        "critic_hidden_dims",
        "actor_activation",
        "critic_activation",
    ):
        if key in ("actor_activation", "critic_activation") and key not in schema:
            raise ValueError(
                f"Incompatible resume checkpoint: missing {key}; "
                "the checkpoint predates the tanh MLP schema, so start a fresh run"
            )
        if schema.get(key) != expected.get(key):
            raise ValueError(
                f"Incompatible resume checkpoint field {key!r}: "
                f"expected {expected.get(key)!r}, got {schema.get(key)!r}"
            )
    if not isinstance(payload.get("actor_state_dict"), dict) or not isinstance(payload.get("critic_state_dict"), dict):
        raise ValueError("Resume checkpoint is missing RSL-RL actor/critic state dictionaries")
    return payload


def _save_checkpoint(
    runner,
    path: Path,
    args,
    env,
    obs,
    train_cfg: dict,
    *,
    torch,
    iteration: int,
    total_steps: int = 0,
    lifetime_total_steps: int | None = None,
    infos=None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = runner.alg.save()
    payload["iter"] = int(iteration)
    checkpoint_infos = dict(infos or {})
    if lifetime_total_steps is None:
        lifetime_total_steps = total_steps
    checkpoint_infos["total_steps"] = int(total_steps)
    checkpoint_infos["lifetime_total_steps"] = int(lifetime_total_steps)
    payload["infos"] = checkpoint_infos
    payload["piper_schema"] = _checkpoint_schema(
        args,
        env,
        obs,
        iteration=iteration,
        train_cfg=train_cfg,
        total_steps=total_steps,
        lifetime_total_steps=lifetime_total_steps,
    )
    # Avoid leaving a truncated ``model.pt`` if Ctrl+C/disk failure occurs
    # during serialization. The temporary file stays beside the destination so
    # os.replace is atomic on the mounted project filesystem.
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _reset_environment(env, seed: int):
    # PiperWarpEnv follows the explicit GPU vector-environment contract:
    # reset(seed=None) mutates state and get_observations() returns the
    # resulting TensorDict.
    env.reset(seed=seed)


def _close_safely(resource):
    if resource is None:
        return
    close = getattr(resource, "close", None)
    if close is not None:
        try:
            close()
        except Exception as exc:
            print(f"Warning: cleanup failed for {type(resource).__name__}: {exc}", file=sys.stderr)


def _train_iterations(
    runner,
    env,
    args,
    obs,
    *,
    viewer=None,
    torch,
    start_iteration: int,
    state: dict,
    on_checkpoint=None,
):
    """RSL-RL's rollout/update loop with a render-and-stop hook per step.

    ``state`` is updated before/after every environment transition so an
    interrupt in the middle of a rollout still has an accurate observation and
    transition count available to the caller for checkpointing.
    """

    from rsl_rl.utils import check_nan

    alg = runner.alg
    logger = runner.logger
    alg.train_mode()
    logger.init_logging_writer()
    device = torch.device(args.device)
    for iteration in range(start_iteration, start_iteration + args.iterations):
        collect_started = time.monotonic()
        for _ in range(args.steps_per_env):
            if state.get("stop_requested", False):
                break
            if viewer is not None and not viewer.poll():
                state["stop_requested"] = True
                state["window_stop_requested"] = True
                break
            with torch.inference_mode():
                actions = alg.act(obs)
                obs, rewards, dones, infos = env.step(actions.to(env.device))
                obs = _as_obs_tensordict(obs, device=device)
                rewards = _to_device(rewards, device)
                dones = _to_device(dones, device)
                infos = _normalise_extras(infos, env, device=device)
                check_nan(obs, rewards, dones)
                alg.process_env_step(obs, rewards, dones, infos)
            logger.process_env_step(rewards, dones, infos)
            state["latest_obs"] = obs
            state["total_steps"] += int(env.num_envs)
            if viewer is not None and time.monotonic() - viewer.last_draw >= viewer.interval:
                try:
                    if not viewer.maybe_draw(_render_frames(env), state["total_steps"], str(device)):
                        state["stop_requested"] = True
                        state["window_stop_requested"] = True
                        break
                except (RuntimeError, ValueError) as exc:
                    print(f"Preview stopped ({exc}); training continues headless.", file=sys.stderr, flush=True)
                    # Keep the GLFW library initialized while Warp's optional
                    # MuJoCo renderer remains alive. Final cleanup closes the
                    # env renderer first, then this owned window/context.
                    viewer.disable()
        if state.get("stop_requested", False):
            break
        collect_time = time.monotonic() - collect_started
        with torch.inference_mode():
            alg.compute_returns(obs)
        learn_started = time.monotonic()
        loss_dict = alg.update()
        learn_time = time.monotonic() - learn_started
        logger.log(
            it=iteration,
            start_it=start_iteration,
            total_it=start_iteration + args.iterations,
            collect_time=collect_time,
            learn_time=learn_time,
            loss_dict=loss_dict,
            learning_rate=alg.learning_rate,
            action_std=alg.get_policy().output_std,
            rnd_weight=None,
        )
        # RSL-RL's built-in runner records the last completed zero-based
        # iteration. Keep this value current for both periodic and interrupted
        # checkpoints.
        runner.current_learning_iteration = iteration
        state["last_iteration"] = iteration
        if (iteration + 1) % args.save_interval == 0:
            if on_checkpoint is not None:
                on_checkpoint(iteration, obs, state["total_steps"])
    state["completed"] = not state.get("stop_requested", False)


def main(argv=None) -> int:
    args = parse_args(argv)
    # Importing torch here keeps argument/help validation usable outside the
    # managed CUDA image while making the runtime requirement explicit.
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "The RSL-RL launcher requires the managed GPU image (PyTorch, MuJoCo Warp, and rsl_rl)."
        ) from exc
    if args.device != "cuda":
        raise RuntimeError("CPU execution is unsupported: this task requires CUDA MuJoCo Warp + RSL-RL")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Run scripts/run_piper_pick_place.sh with the NVIDIA Podman device; "
            "CPU fallback is intentionally disabled."
        )

    def _handle_termination(_signum, _frame):
        # Convert container stop/SIGTERM into the same checkpointing path as
        # Ctrl+C. Podman --init still reaps any child that exits unexpectedly.
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _handle_termination)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    try:
        from scripts.piper_warp_env import PiperWarpEnv
    except ImportError as exc:
        raise RuntimeError("scripts.piper_warp_env.PiperWarpEnv is required for this launcher") from exc
    try:
        from rsl_rl.runners import OnPolicyRunner
    except ImportError as exc:
        raise RuntimeError("rsl_rl 5.x is required; rebuild/use the managed piper-rsl GPU image") from exc

    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    tb_process = None
    env = viewer = runner = None
    train_cfg = None
    status = "error"
    latest_obs = None
    prior_total_steps = 0
    checkpoint_saved = False
    checkpoint_error = None
    state = {
        "latest_obs": None,
        "total_steps": 0,
        "last_iteration": -1,
        "stop_requested": False,
        "window_stop_requested": False,
        "completed": False,
    }
    preview_frames = 0
    try:
        env = PiperWarpEnv(
            num_envs=args.num_envs,
            device=args.device,
            seed=args.seed,
            start_mode=args.start_mode,
        )
        _reset_environment(env, args.seed)
        latest_obs = _observation_from_env(env, device=torch.device(args.device))
        state["latest_obs"] = latest_obs
        if "policy" not in latest_obs:
            raise ValueError(f"PiperWarpEnv observations must include a 'policy' group, got {list(latest_obs.keys())}")
        num_obs = int(latest_obs["policy"].shape[-1])
        num_actions = int(env.num_actions)
        train_cfg = _make_train_cfg(args, num_obs=num_obs, num_actions=num_actions)
        train_cfg_for_file = deepcopy(train_cfg)
        config = vars(args).copy()
        config.update(
            {
                "task": "piper_pick_place",
                "algorithm": "rsl_rl.PPO",
                "physics": "MuJoCo Warp (GPU)",
                "num_obs": num_obs,
                "num_actions": num_actions,
                "num_envs": env.num_envs,
                "device": str(env.device),
                "train_cfg": train_cfg_for_file,
                "output_dir": str(args.output_dir),
                "resume": str(args.resume) if args.resume else None,
            }
        )
        (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"Run directory: {run_dir.resolve()}", flush=True)
        print(f"{env.num_envs} GPU Warp environments -> one RSL-RL PPO policy on CUDA", flush=True)
        if not args.no_tensorboard:
            tb_process = start_tensorboard(args.output_dir, args.tensorboard_port)

        if not args.headless:
            from scripts.viewer_grid import GridWindow, PreviewController

            viewer = PreviewController(GridWindow(), args.fps, total_envs=env.num_envs)
            if not viewer.maybe_draw(_render_frames(env), 0, str(env.device)):
                state["stop_requested"] = True
                state["window_stop_requested"] = True

        runner_cfg = deepcopy(train_cfg)
        runner = OnPolicyRunner(env, runner_cfg, log_dir=str(run_dir / "tensorboard"), device=args.device)
        start_iteration = 0
        if args.resume is not None:
            resume_payload = _load_resume_schema(args.resume, args, env, latest_obs, train_cfg, torch=torch)
            resume_infos = resume_payload.get("infos") or {}
            prior_total_steps = int(
                resume_infos.get(
                    "lifetime_total_steps",
                    resume_payload.get("piper_schema", {}).get(
                        "lifetime_total_steps",
                        resume_infos.get("total_steps", resume_payload.get("piper_schema", {}).get("total_steps", 0)),
                    ),
                )
            )
            runner.load(str(args.resume), map_location=args.device)
            # rsl_rl stores the last completed zero-based iteration. Continue
            # at the next one so a resumed run never silently duplicates it.
            start_iteration = int(runner.current_learning_iteration) + 1
            runner.current_learning_iteration = start_iteration - 1
            state["last_iteration"] = start_iteration - 1
            print(f"Resumed RSL-RL checkpoint: {args.resume} (next iteration {start_iteration})", flush=True)

        def save_periodic(iteration, obs, _total_steps):
            _save_checkpoint(
                runner,
                run_dir / f"model_{iteration}.pt",
                args,
                env,
                obs,
                train_cfg,
                torch=torch,
                iteration=iteration,
                total_steps=_total_steps,
                lifetime_total_steps=prior_total_steps + _total_steps,
            )

        if not state["stop_requested"]:
            _train_iterations(
                runner,
                env,
                args,
                latest_obs,
                viewer=viewer,
                torch=torch,
                start_iteration=start_iteration,
                state=state,
                on_checkpoint=save_periodic,
            )
        latest_obs = state["latest_obs"]
        status = "stopped" if state["window_stop_requested"] else ("completed" if state["completed"] else "interrupted")
    except KeyboardInterrupt:
        status = "interrupted"
        if state.get("latest_obs") is not None:
            latest_obs = state["latest_obs"]
        print("Interrupted; saving the current RSL-RL checkpoint.", flush=True)
    except BaseException:
        status = "error"
        raise
    finally:
        # Save the latest complete PPO state even when the user interrupted
        # between updates. A partial rollout is intentionally not optimized.
        if runner is not None and latest_obs is not None and train_cfg is not None:
            try:
                _save_checkpoint(
                    runner,
                    run_dir / "model.pt",
                    args,
                    env,
                    latest_obs,
                    train_cfg,
                    torch=torch,
                    iteration=int(state.get("last_iteration", -1)),
                    total_steps=int(state.get("total_steps", 0)),
                    lifetime_total_steps=prior_total_steps + int(state.get("total_steps", 0)),
                )
                checkpoint_saved = True
                if viewer is not None and viewer.last_grid is not None:
                    import imageio.v3 as iio

                    iio.imwrite(run_dir / "preview.png", viewer.last_grid)
            except Exception as exc:
                checkpoint_error = exc
                status = "error"
                print(f"Warning: final checkpoint save failed: {exc}", file=sys.stderr, flush=True)
        preview_frames = viewer.window.frames_drawn if viewer is not None else 0
        try:
            if runner is not None and getattr(runner, "logger", None) is not None:
                logger = runner.logger
                # Logger.stop_logging_writer intentionally handles only W&B /
                # Neptune in rsl_rl 5.0.1; TensorBoard's writer needs an
                # explicit flush/close before its server is terminated.
                if getattr(logger, "writer", None) is not None:
                    try:
                        logger.writer.flush()
                    finally:
                        logger.writer.close()
                    logger.stop_logging_writer()
        finally:
            try:
                # PiperWarpEnv owns the optional MuJoCo renderer; close it
                # before GridWindow calls glfw.terminate().
                _close_safely(env)
            finally:
                try:
                    if viewer is not None:
                        viewer.close()
                finally:
                    _terminate_process_group(tb_process)
        summary = {
            "status": status,
            "task": "piper_pick_place",
            "algorithm": "rsl_rl.PPO",
            "physics": "MuJoCo Warp (GPU)",
            "device": args.device,
            "num_envs": args.num_envs,
            "steps_per_env": args.steps_per_env,
            "iterations_requested": args.iterations,
            "last_iteration": int(state.get("last_iteration", -1)),
            "total_timesteps": int(state.get("total_steps", 0)),
            "previous_total_timesteps": prior_total_steps,
            "lifetime_total_timesteps": prior_total_steps + int(state.get("total_steps", 0)),
            "tensorboard": not args.no_tensorboard,
            "tensorboard_port": args.tensorboard_port if not args.no_tensorboard else None,
            "preview_frames": preview_frames,
            "checkpoint_saved": checkpoint_saved,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        if status != "error" and checkpoint_saved:
            print(f"Saved: {run_dir / 'model.pt'} ({status})", flush=True)
    if checkpoint_error is not None:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
