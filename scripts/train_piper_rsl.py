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
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any


CHECKPOINT_SCHEMA = 3
ACTION_DISTRIBUTION_CONTRACT = "tanh_squashed_gaussian_v1"
DEFAULT_OUTPUT_DIR = Path("runs/piper_pick_place")
DEFAULT_STEPS_PER_ENV = 64
DEFAULT_ITERATIONS = 10_000
DEFAULT_TENSORBOARD_PORT = 6006
EXPECTED_REWARD_VERSION = 3
EXPECTED_PHYSICS_TIMESTEP = 0.002
EXPECTED_CONTROL_HZ = 50.0
EXPECTED_CONTROL_SUBSTEPS = 10
EXPECTED_EPISODE_STEPS = 600


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


def _reward_contract(env) -> tuple[int, float]:
    """Return and validate the Piper environment's reward contract.

    The reward version is part of the policy/checkpoint contract.  Refusing an
    older environment here prevents a run from silently mixing reward-v1/v2
    transitions with the reward-v3 policy metadata.
    """

    missing = [
        name
        for name in ("reward_version", "recontact_penalty")
        if not hasattr(env, name)
    ]
    if missing:
        raise RuntimeError(
            "PiperWarpEnv is missing reward contract field(s) "
            f"{', '.join(missing)}; use the reward-v3 environment and start a fresh run"
        )
    reward_version = int(env.reward_version)
    recontact_penalty = float(env.recontact_penalty)
    if reward_version != EXPECTED_REWARD_VERSION:
        raise RuntimeError(
            f"Unsupported Piper reward_version={reward_version}; expected "
            f"{EXPECTED_REWARD_VERSION}. Start a fresh run with the reward-v3 environment."
        )
    if not math.isfinite(recontact_penalty) or recontact_penalty < 0.0:
        raise RuntimeError(
            f"Unsupported Piper recontact_penalty={recontact_penalty!r}; expected a "
            "finite non-negative value. Start a fresh run with the reward-v3 environment."
        )
    return reward_version, recontact_penalty


def _training_config_class():
    """Import the light-weight training config without breaking direct scripts."""

    # ``python scripts/train_piper_rsl.py --help`` has no repository root on
    # every Python installation's import path.  Keep this import lazy and
    # provide the sibling-module fallback used by that invocation.
    try:
        from scripts.piper_training_config import PiperTrainingConfig
    except ModuleNotFoundError as exc:
        if exc.name not in {"scripts", "scripts.piper_training_config"}:
            raise
        from piper_training_config import PiperTrainingConfig
    return PiperTrainingConfig


def _make_training_config(args):
    """Build the immutable environment config represented by CLI toggles."""

    config_class = _training_config_class()
    return config_class(
        domain_randomization=not args.no_domain_randomization,
        sensor_noise=not args.no_sensor_noise,
        curriculum=not args.no_curriculum,
    )


def _jsonable(value):
    """Convert config/dataclass values into stable JSON/checkpoint values."""

    if hasattr(value, "to_dict") and callable(value.to_dict):
        value = value.to_dict()
    elif hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict

        value = asdict(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        return float(value)
    raise TypeError(f"Unsupported training config value {type(value).__name__}")


def _training_config_dict(args, env=None):
    if env is None:
        config = _make_training_config(args)
    else:
        config = getattr(env, "training_config", None)
        if config is None or not callable(getattr(config, "to_dict", None)):
            raise RuntimeError("PiperWarpEnv must expose training_config.to_dict() for checkpoint metadata")
    return _jsonable(config)


def _env_cfg(env) -> dict:
    config = getattr(env, "cfg", None)
    if isinstance(config, dict):
        return _jsonable(config)
    return {}


def _timing_contract(env) -> dict:
    """Return canonical physical/control timing metadata for schema checks."""

    model = getattr(env, "model", None)
    opt = getattr(model, "opt", None)
    if opt is None or not hasattr(opt, "timestep"):
        raise RuntimeError("PiperWarpEnv must expose env.model.opt.timestep for the timing contract")
    if not hasattr(env, "frame_skip") or not hasattr(env, "max_episode_length"):
        raise RuntimeError("PiperWarpEnv must expose frame_skip and max_episode_length for the timing contract")
    timestep = float(opt.timestep)
    control_substeps = int(env.frame_skip)
    control_hz = 1.0 / (timestep * control_substeps)
    episode_steps = int(env.max_episode_length)
    episode_seconds = episode_steps / control_hz
    return {
        "physics_timestep": timestep,
        "control_hz": control_hz,
        "control_substeps": control_substeps,
        "episode_steps": episode_steps,
        "episode_seconds": episode_seconds,
    }


def _validate_timing_contract(env) -> dict:
    timing = _timing_contract(env)
    expected = {
        "physics_timestep": EXPECTED_PHYSICS_TIMESTEP,
        "control_hz": EXPECTED_CONTROL_HZ,
        "control_substeps": EXPECTED_CONTROL_SUBSTEPS,
        "episode_steps": EXPECTED_EPISODE_STEPS,
        "episode_seconds": EXPECTED_EPISODE_STEPS / EXPECTED_CONTROL_HZ,
    }
    for key, value in expected.items():
        actual = timing[key]
        if isinstance(value, float):
            matches = math.isclose(float(actual), value, rel_tol=0.0, abs_tol=1e-9)
        else:
            matches = actual == value
        if not matches:
            raise RuntimeError(
                f"Unsupported Piper timing {key}={actual!r}; expected {value!r}. "
                "The schema-2 trainer requires 0.002 s x 10 at 50 Hz and 600 ticks."
            )
    return timing


def _observation_contract(env, obs) -> dict:
    policy = obs.get("policy") if hasattr(obs, "get") else None
    critic = obs.get("critic") if hasattr(obs, "get") else None
    if policy is None:
        raise ValueError("Piper observations must include a 'policy' group")
    if critic is None:
        raise ValueError("Piper observations must include a separate 'critic' group")
    env_cfg = getattr(env, "cfg", None)
    if not isinstance(env_cfg, dict):
        raise RuntimeError("PiperWarpEnv must expose cfg with observation_version and action_semantics")
    for key in ("observation_version", "action_semantics"):
        if key not in env_cfg:
            raise RuntimeError(f"PiperWarpEnv cfg is missing required field {key!r}")
    obs_groups = env_cfg.get("obs_groups", {"actor": ["policy"], "critic": ["critic"]})
    if not isinstance(obs_groups, dict) or set(obs_groups) != {"actor", "critic"}:
        raise RuntimeError("PiperWarpEnv cfg obs_groups must contain actor and critic lists")
    normalized_groups = {}
    for name in ("actor", "critic"):
        groups = obs_groups[name]
        if not isinstance(groups, (list, tuple)) or not groups or not all(isinstance(group, str) for group in groups):
            raise RuntimeError(f"PiperWarpEnv cfg obs_groups[{name!r}] must be a non-empty list of names")
        missing = [group for group in groups if group not in obs]
        if missing:
            raise ValueError(f"Piper observations are missing {name} group(s): {missing}")
        normalized_groups[name] = list(groups)
    actor_obs_dim = sum(int(obs[group].shape[-1]) for group in normalized_groups["actor"])
    critic_obs_dim = sum(int(obs[group].shape[-1]) for group in normalized_groups["critic"])
    vision_config = env_cfg.get("vision_config")
    if vision_config is not None:
        vision_config = _jsonable(vision_config)
    has_vision = "vision" in normalized_groups["actor"]
    if has_vision != (vision_config is not None):
        raise RuntimeError("Piper RGB observations and vision_config metadata must be enabled together")
    return {
        "observation_version": str(env_cfg["observation_version"]),
        "num_obs": int(policy.shape[-1]),
        "num_critic_obs": int(critic.shape[-1]),
        "actor_obs_dim": actor_obs_dim,
        "critic_obs_dim": critic_obs_dim,
        "obs_groups": normalized_groups,
        "action_semantics": str(env_cfg["action_semantics"]),
        "vision_config": vision_config,
    }


def _env_training_steps(env) -> int:
    if not hasattr(env, "training_steps"):
        raise RuntimeError("PiperWarpEnv must expose training_steps for checkpoint progression")
    value = env.training_steps
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    value = int(value)
    if value < 0:
        raise ValueError(f"Piper training_steps must be non-negative, got {value}")
    return value


def _set_env_training_steps(env, count: int) -> None:
    count = int(count)
    if count < 0:
        raise ValueError(f"Piper training_steps must be non-negative, got {count}")
    setter = getattr(env, "set_training_steps", None)
    if not callable(setter):
        raise ValueError(
            "Resume checkpoint requires PiperWarpEnv.set_training_steps(count) "
            "to restore curriculum progression"
        )
    setter(count)


def _normalization_state_present(state_dict) -> bool:
    if not isinstance(state_dict, dict):
        return False
    return any("normalizer" in str(key).lower() for key in state_dict)


def _check_normalization_state(payload: dict, train_cfg: dict, runner=None) -> None:
    """Verify actual RSL actor/critic state dicts carry empirical statistics."""

    actor_state = payload.get("actor_state_dict")
    critic_state = payload.get("critic_state_dict")
    if not isinstance(actor_state, dict) or not isinstance(critic_state, dict):
        raise ValueError("RSL-RL checkpoint must contain actor_state_dict and critic_state_dict mappings")
    checks = (
        ("actor", actor_state, train_cfg.get("actor", {}).get("obs_normalization", False)),
        ("critic", critic_state, train_cfg.get("critic", {}).get("obs_normalization", False)),
    )
    for name, state, enabled in checks:
        if not enabled:
            continue
        if not _normalization_state_present(state):
            raise ValueError(
                f"RSL {name} state_dict is missing empirical observation-normalization statistics"
            )


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
        choices=("curriculum", "above_cube", "home"),
        default="curriculum",
        help="Piper reset policy: curriculum, near-cube above_cube, or fixed home pose",
    )
    parser.add_argument(
        "--no-domain-randomization",
        action="store_true",
        help="disable per-episode physical parameter randomization",
    )
    parser.add_argument(
        "--no-sensor-noise",
        action="store_true",
        help="disable actor sensor noise, zero offsets, and observation delay",
    )
    parser.add_argument(
        "--no-curriculum",
        action="store_true",
        help="hold the curriculum at its final difficulty stage",
    )
    parser.add_argument(
        "--rgb",
        action="store_true",
        help="enable the optional frozen ResNet RGB policy input alongside proprioception",
    )
    parser.add_argument(
        "--rgb-encoder-checkpoint",
        type=Path,
        help="offline RGB encoder weights for a fresh run (resume loads weights from the .pt checkpoint)",
    )
    parser.add_argument(
        "--rgb-camera-fps",
        type=positive_float,
        default=30.0,
        help="simulated RGB capture rate in Hz (default: 30)",
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
    if args.rgb_encoder_checkpoint is not None and not args.rgb:
        parser.error("--rgb-encoder-checkpoint requires --rgb")
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

    The Docker wrapper publishes this container port only on host
    ``127.0.0.1``.  No process is reused or killed when the requested host port
    is already occupied; the wrapper rejects that case before Docker starts.
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


def _make_train_cfg(args, *, num_obs: int, num_actions: int, obs_groups: dict | None = None) -> dict:
    """Build the rsl_rl 5.x OnPolicyRunner configuration."""

    del num_obs, num_actions  # dimensions are inferred from TensorDict/env.
    if obs_groups is None:
        obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    else:
        obs_groups = {name: list(groups) for name, groups in obs_groups.items()}
    return {
        "seed": args.seed,
        "runner_class_name": "OnPolicyRunner",
        "algorithm_class_name": "PPO",
        "num_steps_per_env": args.steps_per_env,
        "save_interval": args.save_interval,
        "logger": "tensorboard",
        "run_name": "piper_pick_place",
        "obs_groups": obs_groups,
        "actor": {
            "class_name": "rsl_rl.models:MLPModel",
            "hidden_dims": [256, 128, 64],
            "activation": "tanh",
            "obs_normalization": True,
            "distribution_cfg": {
                "class_name": "scripts.piper_action_distribution:TanhGaussianDistribution",
                "init_std": 1.0,
                "std_type": "log",
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
            # Keep the potential-shaping discount and PPO trace on the same
            # physical horizon after moving to the 50 Hz control tick.
            "gamma": math.sqrt(0.99),
            "lam": math.sqrt(0.95),
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
    training_steps: int | None = None,
) -> dict:
    observation = _observation_contract(env, obs)
    reward_version, recontact_penalty = _reward_contract(env)
    if training_steps is None:
        training_steps = _env_training_steps(env)
    normalization = {
        "actor": bool(train_cfg["actor"].get("obs_normalization", False)),
        "critic": bool(train_cfg["critic"].get("obs_normalization", False)),
    }
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "task": "piper_pick_place",
        "algorithm": "rsl_rl.PPO",
        "rsl_rl_config": "v5",
        "start_mode": str(args.start_mode),
        "num_obs": observation["num_obs"],
        "num_critic_obs": observation["num_critic_obs"],
        "actor_obs_dim": observation["actor_obs_dim"],
        "critic_obs_dim": observation["critic_obs_dim"],
        "num_actions": int(env.num_actions),
        "action_distribution": ACTION_DISTRIBUTION_CONTRACT,
        "reward_version": reward_version,
        "recontact_penalty": recontact_penalty,
        "steps_per_env": int(args.steps_per_env),
        "iteration": int(iteration),
        "actor_hidden_dims": list(train_cfg["actor"]["hidden_dims"]),
        "critic_hidden_dims": list(train_cfg["critic"]["hidden_dims"]),
        "actor_activation": str(train_cfg["actor"]["activation"]),
        "critic_activation": str(train_cfg["critic"]["activation"]),
        "observation_version": observation["observation_version"],
        "obs_groups": observation["obs_groups"],
        "normalization": normalization,
        "actor_obs_normalization": normalization["actor"],
        "critic_obs_normalization": normalization["critic"],
        "gamma": float(train_cfg["algorithm"]["gamma"]),
        "lam": float(train_cfg["algorithm"]["lam"]),
        "action_semantics": observation["action_semantics"],
        "vision_config": observation["vision_config"],
        "timing": _timing_contract(env),
        "training_config": _training_config_dict(args, env),
        # This is a vector-environment tick count, independent of the number
        # of worlds and PPO rollout batch size.  It is restored on resume.
        "training_steps": int(training_steps),
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
        "start_mode",
        "num_obs",
        "num_critic_obs",
        "actor_obs_dim",
        "critic_obs_dim",
        "num_actions",
        "action_distribution",
        "steps_per_env",
        "actor_hidden_dims",
        "critic_hidden_dims",
        "actor_activation",
        "critic_activation",
        "reward_version",
        "recontact_penalty",
        "observation_version",
        "obs_groups",
        "normalization",
        "actor_obs_normalization",
        "critic_obs_normalization",
        "gamma",
        "lam",
        "action_semantics",
        "vision_config",
        "timing",
        "training_config",
    ):
        if key not in schema:
            explanation = "the checkpoint predates the schema-2 timing/observation contract"
            if key in {"actor_activation", "critic_activation"}:
                explanation = "the checkpoint predates the tanh MLP schema"
            elif key in {"reward_version", "recontact_penalty"}:
                explanation = "the checkpoint predates the reward-v3 contract"
            raise ValueError(
                f"Incompatible resume checkpoint: missing {key}; {explanation}; "
                "start a fresh run"
            )
        if schema.get(key) != expected.get(key):
            raise ValueError(
                f"Incompatible resume checkpoint field {key!r}: "
                f"expected {expected.get(key)!r}, got {schema.get(key)!r}; "
                "start a fresh run"
            )
    if "training_steps" not in schema:
        raise ValueError(
            "Incompatible resume checkpoint: missing training_steps; "
            "curriculum progression cannot be restored; start a fresh run"
        )
    try:
        if int(schema["training_steps"]) < 0:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ValueError("Incompatible resume checkpoint: training_steps must be a non-negative integer") from exc
    if not isinstance(payload.get("actor_state_dict"), dict) or not isinstance(payload.get("critic_state_dict"), dict):
        raise ValueError("Resume checkpoint is missing RSL-RL actor/critic state dictionaries")
    if expected["vision_config"] is not None:
        if not isinstance(payload.get("vision_encoder_state_dict"), dict):
            raise ValueError(
                "RGB resume checkpoint is missing vision_encoder_state_dict; "
                "RGB and state-only checkpoints are incompatible"
            )
    _check_normalization_state(payload, train_cfg)
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
    payload = dict(runner.alg.save())
    checkpoint_state = getattr(env, "checkpoint_state", None)
    if callable(checkpoint_state):
        extra = checkpoint_state()
        if not isinstance(extra, dict):
            raise ValueError("Piper environment checkpoint_state() must return a mapping")
        payload.update(extra)
    _check_normalization_state(payload, train_cfg, runner=runner)
    current_training_steps = _env_training_steps(env)
    payload["iter"] = int(iteration)
    checkpoint_infos = dict(infos or {})
    if lifetime_total_steps is None:
        lifetime_total_steps = total_steps
    checkpoint_infos["total_steps"] = int(total_steps)
    checkpoint_infos["lifetime_total_steps"] = int(lifetime_total_steps)
    checkpoint_infos["training_steps"] = current_training_steps
    payload["infos"] = checkpoint_infos
    payload["training_steps"] = current_training_steps
    payload["piper_schema"] = _checkpoint_schema(
        args,
        env,
        obs,
        iteration=iteration,
        train_cfg=train_cfg,
        total_steps=total_steps,
        lifetime_total_steps=lifetime_total_steps,
        training_steps=current_training_steps,
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


def _format_duration(seconds: float) -> str:
    """Format elapsed durations without dropping whole days."""

    seconds = max(0, int(seconds))
    days, seconds = divmod(seconds, 24 * 60 * 60)
    hours, seconds = divmod(seconds, 60 * 60)
    minutes, seconds = divmod(seconds, 60)
    clock = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{days}d {clock}" if days else clock


def _log_iteration_with_day_aware_eta(logger, *, iteration: int, start_iteration: int, total_iterations: int, **kwargs):
    """Call rsl_rl's logger while retaining days in its console durations.

    rsl_rl 5.0.1 formats both values through ``time.strftime('%H:%M:%S')``;
    that wraps after 24 hours.  Capturing only the already-formatted console
    text lets the launcher correct the display without modifying the installed
    package or changing TensorBoard scalars.
    """

    rendered = io.StringIO()
    with redirect_stdout(rendered):
        logger.log(
            it=iteration,
            start_it=start_iteration,
            total_it=total_iterations,
            **kwargs,
        )
    output = rendered.getvalue()
    if not output:
        return
    elapsed = getattr(logger, "tot_time", None)
    if elapsed is None:
        print(output, end="")
        return
    done_iterations = iteration + 1 - start_iteration
    remaining_iterations = total_iterations - start_iteration - done_iterations
    eta = float(elapsed) / done_iterations * max(0, remaining_iterations) if done_iterations > 0 else 0.0
    replacements = {
        "Time elapsed:": _format_duration(float(elapsed)),
        "ETA:": _format_duration(eta),
    }
    lines = output.splitlines(keepends=True)
    for index, line in enumerate(lines):
        for label, value in replacements.items():
            marker = label
            position = line.find(marker)
            if position < 0:
                continue
            ending = "\n" if line.endswith("\n") else ""
            lines[index] = line[: position + len(marker)] + f" {value}" + ending
            break
    print("".join(lines), end="")


def _summarize_action_diagnostics(
    action_sum,
    action_square_sum,
    action_count: int,
    temporal_square_sum,
    temporal_count: int,
):
    """Return bounded policy-target moments without pooling dimensions.

    These values describe normalized policy outputs before actuator target
    slew limiting; they are not physical actuator velocity statistics.
    """

    count = max(1, int(action_count))
    mean = action_sum / count
    variance = (action_square_sum / count - mean.square()).clamp_min(0.0)
    return {
        "action_mean": float(mean.mean().item()),
        "action_std": float(variance.sqrt().mean().item()),
        "action_near_boundary_denominator": count,
        "temporal_action_rms": math.sqrt(float(temporal_square_sum.item()) / max(1, int(temporal_count))),
        "temporal_action_denominator": int(temporal_count),
    }


def _rolling_episode_diagnostics(window):
    """Compute success/lift rates over the explicitly bounded update window."""

    completed = sum(int(item[0]) for item in window)
    success = sum(int(item[1]) for item in window)
    lifted = sum(int(item[2]) for item in window)
    clean_success = sum(int(item[3]) for item in window)
    return {
        "completed": completed,
        "success": success,
        "lifted": lifted,
        "clean_success": clean_success,
        "success_rate": success / max(1, completed),
        "clean_success_rate": clean_success / max(1, completed),
        "lift_rate": lifted / max(1, completed),
    }


def _flush_pending_episode_counts(state):
    """Commit a rollout's terminal counts exactly once for interruption safety."""

    pending = state.get("_pending_episode_counts")
    if pending is None or state.get("_pending_counts_committed", False):
        return False
    completed, successes, lifted, anomalies, clean_successes = (
        int(value.item()) if hasattr(value, "item") else int(value)
        for value in pending
    )
    state["completed_episodes"] = int(state.get("completed_episodes", 0)) + completed
    state["success_episodes"] = int(state.get("success_episodes", 0)) + successes
    state["lifted_episodes"] = int(state.get("lifted_episodes", 0)) + lifted
    state["success_contact_anomalies"] = int(state.get("success_contact_anomalies", 0)) + anomalies
    state["clean_success_episodes"] = int(state.get("clean_success_episodes", 0)) + clean_successes
    state["partial_rollout"] = {
        "completed_episodes": completed,
        "success_episodes": successes,
        "lifted_episodes": lifted,
        "success_contact_anomalies": anomalies,
        "clean_success_episodes": clean_successes,
    }
    state["_pending_counts_committed"] = True
    return True


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

    def commit_episode_counts(completed: int, successes: int, lifted: int, anomalies: int = 0, clean_successes: int = 0):
        state["completed_episodes"] = int(state.get("completed_episodes", 0)) + int(completed)
        state["success_episodes"] = int(state.get("success_episodes", 0)) + int(successes)
        state["lifted_episodes"] = int(state.get("lifted_episodes", 0)) + int(lifted)
        state["success_contact_anomalies"] = int(state.get("success_contact_anomalies", 0)) + int(anomalies)
        state["clean_success_episodes"] = int(state.get("clean_success_episodes", 0)) + int(clean_successes)
        episode_window = list(state.get("episode_window", []))
        episode_window.append((int(completed), int(successes), int(lifted), int(clean_successes)))
        state["episode_window"] = episode_window[-20:]
        state["_pending_counts_committed"] = True
        return _rolling_episode_diagnostics(state["episode_window"])

    for iteration in range(start_iteration, start_iteration + args.iterations):
        collect_started = time.monotonic()
        action_sum = torch.zeros(int(env.num_actions), dtype=torch.float64, device=device)
        action_square_sum = torch.zeros(int(env.num_actions), dtype=torch.float64, device=device)
        action_near_boundary = torch.zeros((), dtype=torch.int64, device=device)
        action_count = 0
        temporal_square_sum = torch.zeros((), dtype=torch.float64, device=device)
        temporal_count = 0
        previous_actions = None
        previous_dones = None
        completed_count = torch.zeros((), dtype=torch.int64, device=device)
        success_count = torch.zeros((), dtype=torch.int64, device=device)
        lifted_count = torch.zeros((), dtype=torch.int64, device=device)
        success_contact_anomaly_count = torch.zeros((), dtype=torch.int64, device=device)
        clean_success_count = torch.zeros((), dtype=torch.int64, device=device)
        state["_pending_episode_counts"] = (
            completed_count,
            success_count,
            lifted_count,
            success_contact_anomaly_count,
            clean_success_count,
        )
        state["_pending_counts_committed"] = False
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
                # Bounded normalized policy-target actions, before the
                # environment's physical actuator slew limiter.
                executed_actions = actions.detach()
                executed_float = executed_actions.to(torch.float64)
                action_sum += executed_float.sum(dim=0)
                action_square_sum += torch.square(executed_float).sum(dim=0)
                action_near_boundary += (executed_actions.abs() >= 0.98).sum().to(torch.int64)
                action_count += int(executed_actions.shape[0])
                if previous_actions is not None:
                    valid = ~previous_dones
                    if valid.any():
                        temporal_square_sum += torch.square(
                            executed_float[valid] - previous_actions[valid]
                        ).sum()
                        temporal_count += int(valid.sum().item()) * int(env.num_actions)
                obs = _as_obs_tensordict(obs, device=device)
                rewards = _to_device(rewards, device)
                dones = _to_device(dones, device)
                infos = _normalise_extras(infos, env, device=device)
                done_mask = dones.to(dtype=torch.bool)
                completed_count += done_mask.to(torch.int64).sum()
                success_count += (infos.get("is_success", torch.zeros_like(done_mask)) & done_mask).to(torch.int64).sum()
                lifted_count += (infos.get("has_lifted", torch.zeros_like(done_mask)) & done_mask).to(torch.int64).sum()
                success_contact_anomaly_count += (
                    # ``recontacted`` is episode-history state, so this
                    # diagnostic is intentionally stricter than instantaneous
                    # contact at the terminal step.
                    infos.get("is_success", torch.zeros_like(done_mask))
                    & done_mask
                    & (
                        ~infos.get("has_placed", torch.zeros_like(done_mask))
                        | infos.get("recontacted", torch.zeros_like(done_mask))
                    )
                ).to(torch.int64).sum()
                clean_success_count += (
                    infos.get("is_success", torch.zeros_like(done_mask))
                    & done_mask
                    & infos.get("has_placed", torch.zeros_like(done_mask))
                    & ~infos.get("recontacted", torch.zeros_like(done_mask))
                ).to(torch.int64).sum()
                state["_pending_episode_counts"] = (
                    completed_count,
                    success_count,
                    lifted_count,
                    success_contact_anomaly_count,
                    clean_success_count,
                )
                previous_actions = executed_float
                previous_dones = done_mask
                check_nan(obs, rewards, dones)
                alg.process_env_step(obs, rewards, dones, infos)
            logger.process_env_step(rewards, dones, infos)
            state["latest_obs"] = obs
            state["total_steps"] += int(env.num_envs)
            state["training_steps"] = _env_training_steps(env)
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
            # Preserve terminal counts from a rollout interrupted by a window
            # close before PPO update/checkpoint handling returns control.
            _flush_pending_episode_counts(state)
            break
        collect_time = time.monotonic() - collect_started
        with torch.inference_mode():
            alg.compute_returns(obs)
        learn_started = time.monotonic()
        loss_dict = alg.update()
        learn_time = time.monotonic() - learn_started
        _log_iteration_with_day_aware_eta(
            logger,
            iteration=iteration,
            start_iteration=start_iteration,
            total_iterations=start_iteration + args.iterations,
            collect_time=collect_time,
            learn_time=learn_time,
            loss_dict=loss_dict,
            learning_rate=alg.learning_rate,
            action_std=alg.get_policy().output_std,
            rnd_weight=None,
        )
        completed = int(completed_count.item())
        successes = int(success_count.item())
        lifted = int(lifted_count.item())
        success_contact_anomalies = int(success_contact_anomaly_count.item())
        clean_successes = int(clean_success_count.item())
        # Commit once per complete PPO update. The early-stop path above
        # commits the partial rollout before returning to checkpoint logic.
        rolling = commit_episode_counts(completed, successes, lifted, success_contact_anomalies, clean_successes)
        action_diagnostics = _summarize_action_diagnostics(
            action_sum,
            action_square_sum,
            action_count,
            temporal_square_sum,
            temporal_count,
        )
        writer = getattr(logger, "writer", None)
        if writer is not None:
            action_denominator = max(1, action_count * int(env.num_actions))
            writer.add_scalar("Perf/policy_action_near_boundary_fraction", float(action_near_boundary.item()) / action_denominator, iteration)
            writer.add_scalar("Perf/policy_action_mean", action_diagnostics["action_mean"], iteration)
            writer.add_scalar("Perf/policy_action_std", action_diagnostics["action_std"], iteration)
            writer.add_scalar("Perf/policy_temporal_action_rms", action_diagnostics["temporal_action_rms"], iteration)
            writer.add_scalar("Perf/policy_temporal_action_denominator", action_diagnostics["temporal_action_denominator"], iteration)
            writer.add_scalar("Episode/completed_count", completed, iteration)
            writer.add_scalar("Episode/success_count", successes, iteration)
            writer.add_scalar("Episode/lifted_count", lifted, iteration)
            writer.add_scalar("Episode/success_contact_anomaly_count", success_contact_anomalies, iteration)
            writer.add_scalar("Episode/clean_success_count", clean_successes, iteration)
            writer.add_scalar("Episode/rolling_completed_count", rolling["completed"], iteration)
            writer.add_scalar("Episode/rolling_success_denominator", rolling["completed"], iteration)
            writer.add_scalar("Episode/rolling_success_rate", rolling["success_rate"], iteration)
            writer.add_scalar("Episode/rolling_clean_success_denominator", rolling["completed"], iteration)
            writer.add_scalar("Episode/rolling_clean_success_rate", rolling["clean_success_rate"], iteration)
            writer.add_scalar("Episode/rolling_lift_rate", rolling["lift_rate"], iteration)
        print(
            "Episode metrics (last 20 updates): "
            f"completed={rolling['completed']} success={rolling['success']} "
            f"clean_success={rolling['clean_success']} lifted={rolling['lifted']} "
            f"success_rate={rolling['success_rate']:.4f} "
            f"clean_success_rate={rolling['clean_success_rate']:.4f} "
            f"lift_rate={rolling['lift_rate']:.4f}",
            flush=True,
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
            "CUDA is unavailable. Run scripts/run_piper_pick_place.sh with Docker GPU support (--gpus all); "
            "CPU fallback is intentionally disabled."
        )

    def _handle_termination(_signum, _frame):
        # Convert container stop/SIGTERM into the same checkpointing path as
        # Ctrl+C. Docker --init still reaps any child that exits unexpectedly.
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _handle_termination)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    try:
        from scripts.piper_warp_env import PiperWarpEnv
    except ImportError as exc:
        raise RuntimeError("scripts.piper_warp_env.PiperWarpEnv is required for this launcher") from exc
    PiperRGBEnv = None
    if args.rgb:
        try:
            from scripts.piper_rgb_env import PiperRGBEnv
        except ImportError as exc:
            raise RuntimeError("--rgb requires scripts.piper_rgb_env.PiperRGBEnv") from exc
    try:
        from rsl_rl.runners import OnPolicyRunner
    except ImportError as exc:
        raise RuntimeError("rsl_rl 5.x is required; rebuild/use the managed piper-rsl GPU image") from exc

    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    tb_process = None
    env = viewer = runner = None
    train_cfg = None
    training_config = None
    status = "error"
    latest_obs = None
    prior_total_steps = 0
    checkpoint_saved = False
    checkpoint_error = None
    # An incompatible or malformed resume must not fall through to the
    # finally block and create a misleading fresh checkpoint.  This becomes
    # true only after schema validation and runner.load both succeed.
    resume_checkpoint_ready = args.resume is None
    state = {
        "latest_obs": None,
        "total_steps": 0,
        "training_steps": 0,
        "last_iteration": -1,
        "stop_requested": False,
        "window_stop_requested": False,
        "completed": False,
        "completed_episodes": 0,
        "success_episodes": 0,
        "lifted_episodes": 0,
        "success_contact_anomalies": 0,
        "clean_success_episodes": 0,
        "episode_window": [],
    }
    preview_frames = 0
    resume_payload_hint = None
    try:
        training_config = _make_training_config(args)
        if args.rgb and args.resume is not None:
            if args.resume.suffix.lower() != ".pt" or not args.resume.is_file():
                raise ValueError("RGB resume requires an existing RSL-RL .pt checkpoint")
            # Load the CPU checkpoint before constructing the encoder.  The
            # RGB wrapper uses its frozen state dict offline, so a resume can
            # never silently download/randomly initialize a different encoder.
            resume_payload_hint = torch.load(args.resume, map_location="cpu", weights_only=False)
        env = PiperWarpEnv(
            num_envs=args.num_envs,
            device=args.device,
            seed=args.seed,
            start_mode=args.start_mode,
            training_config=training_config,
        )
        if args.rgb:
            env = PiperRGBEnv(
                env,
                encoder_checkpoint=resume_payload_hint,
                encoder_weights=args.rgb_encoder_checkpoint,
                camera_fps=args.rgb_camera_fps,
            )
        timing = _validate_timing_contract(env)
        _reset_environment(env, args.seed)
        latest_obs = _observation_from_env(env, device=torch.device(args.device))
        state["latest_obs"] = latest_obs
        state["training_steps"] = _env_training_steps(env)
        if "policy" not in latest_obs:
            raise ValueError(f"PiperWarpEnv observations must include a 'policy' group, got {list(latest_obs.keys())}")
        if "critic" not in latest_obs:
            raise ValueError(
                "PiperWarpEnv observations must include separate clean 'critic' and noisy 'policy' groups"
            )
        num_obs = int(latest_obs["policy"].shape[-1])
        num_critic_obs = int(latest_obs["critic"].shape[-1])
        declared_num_obs = getattr(env, "num_observations", None)
        if declared_num_obs is None or int(declared_num_obs) != num_obs:
            raise RuntimeError(
                "PiperWarpEnv observation contract mismatch: "
                f"declared num_observations={declared_num_obs!r}, returned policy width={num_obs}"
            )
        num_actions = int(env.num_actions)
        reward_version, recontact_penalty = _reward_contract(env)
        observation_contract = _observation_contract(env, latest_obs)
        train_cfg = _make_train_cfg(
            args,
            num_obs=num_obs,
            num_actions=num_actions,
            obs_groups=observation_contract["obs_groups"],
        )
        shaping_gamma = getattr(env, "shaping_gamma", None)
        if shaping_gamma is None or not math.isclose(
            float(shaping_gamma), train_cfg["algorithm"]["gamma"], rel_tol=0.0, abs_tol=1e-9
        ):
            raise RuntimeError(
                "Piper shaping_gamma and PPO gamma disagree; both must use sqrt(0.99) "
                "for the 50 Hz physical horizon"
            )
        training_config_dict = _training_config_dict(args, env)
        env_cfg = _env_cfg(env)
        env_cfg.update(
            {
                "training_config": training_config_dict,
                "timing": timing,
                "observation_version": observation_contract["observation_version"],
                "action_semantics": observation_contract["action_semantics"],
                "vision_config": observation_contract["vision_config"],
                "rgb": args.rgb,
            }
        )
        train_cfg_for_file = deepcopy(train_cfg)
        config = vars(args).copy()
        config.update(
            {
                "task": "piper_pick_place",
                "algorithm": "rsl_rl.PPO",
                "physics": "MuJoCo Warp (GPU)",
                "num_obs": num_obs,
                "num_critic_obs": num_critic_obs,
                "actor_obs_dim": observation_contract["actor_obs_dim"],
                "critic_obs_dim": observation_contract["critic_obs_dim"],
                "num_actions": num_actions,
                "reward_version": reward_version,
                "recontact_penalty": recontact_penalty,
                "num_envs": env.num_envs,
                "device": str(env.device),
                "train_cfg": train_cfg_for_file,
                "training_config": training_config_dict,
                "env_cfg": env_cfg,
                "timing": timing,
                "observation_version": observation_contract["observation_version"],
                "action_semantics": observation_contract["action_semantics"],
                "vision_config": observation_contract["vision_config"],
                "rgb": args.rgb,
                "output_dir": str(args.output_dir),
                "resume": str(args.resume) if args.resume else None,
            }
        )
        (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8")
        (run_dir / "env.cfg").write_text(json.dumps(env_cfg, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"Run directory: {run_dir.resolve()}", flush=True)
        print(f"{env.num_envs} GPU Warp environments -> one RSL-RL PPO policy on CUDA", flush=True)
        if not args.no_tensorboard:
            tb_process = start_tensorboard(args.output_dir, args.tensorboard_port)

        if not args.headless:
            from scripts.viewer_grid import GridWindow, PreviewController

            if args.rgb:
                # The RGB policy renderer owns MuJoCo's EGL backend.  Keep the
                # visible GLFW preview in a child process so EGL and GLX never
                # contend for one MuJoCo process-global context.
                viewer = PreviewController.isolated(
                    args.fps,
                    total_envs=env.num_envs,
                    software=os.environ.get("PIPER_PREVIEW_SOFTWARE") == "1",
                    wslg=os.environ.get("PIPER_PREVIEW_WSLG") == "1",
                )
            else:
                viewer = PreviewController(GridWindow(), args.fps, total_envs=env.num_envs)
            if not viewer.maybe_draw(_render_frames(env), 0, str(env.device)):
                state["stop_requested"] = True
                state["window_stop_requested"] = True

        runner_cfg = deepcopy(train_cfg)
        runner = OnPolicyRunner(env, runner_cfg, log_dir=str(run_dir / "tensorboard"), device=args.device)
        start_iteration = 0
        if args.resume is not None:
            resume_payload = _load_resume_schema(args.resume, args, env, latest_obs, train_cfg, torch=torch)
            resume_schema = resume_payload["piper_schema"]
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
            _set_env_training_steps(env, int(resume_schema["training_steps"]))
            # Loading a checkpoint initializes policy/critic state first.  The
            # environment then resets at the restored curriculum tick and the
            # runner starts from the resulting observation, including any
            # sensor-history state owned by the environment.
            _reset_environment(env, args.seed)
            latest_obs = _observation_from_env(env, device=torch.device(args.device))
            state["latest_obs"] = latest_obs
            state["training_steps"] = _env_training_steps(env)
            # rsl_rl stores the last completed zero-based iteration. Continue
            # at the next one so a resumed run never silently duplicates it.
            start_iteration = int(runner.current_learning_iteration) + 1
            runner.current_learning_iteration = start_iteration - 1
            state["last_iteration"] = start_iteration - 1
            resume_checkpoint_ready = True
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
        # A KeyboardInterrupt can arrive inside a rollout before the normal
        # iteration commit; preserve those terminal counts in the summary.
        _flush_pending_episode_counts(state)
        # Save the latest complete PPO state even when the user interrupted
        # between updates. A partial rollout is intentionally not optimized.
        if resume_checkpoint_ready and runner is not None and latest_obs is not None and train_cfg is not None:
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
        preview_frames = viewer.frames_drawn if viewer is not None else 0
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
        summary_training_steps = int(state.get("training_steps", 0))
        summary_rgb_timing = None
        if env is not None:
            try:
                summary_training_steps = _env_training_steps(env)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                # Preserve the original training failure if an environment
                # died before it could expose its progress counter.
                pass
            try:
                timing = getattr(env, "render_timing", None)
                if isinstance(timing, dict):
                    summary_rgb_timing = _jsonable(timing)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                summary_rgb_timing = None
        summary = {
            "status": status,
            "task": "piper_pick_place",
            "algorithm": "rsl_rl.PPO",
            "physics": "MuJoCo Warp (GPU)",
            "rgb": bool(args.rgb),
            "device": args.device,
            "num_envs": args.num_envs,
            "steps_per_env": args.steps_per_env,
            "iterations_requested": args.iterations,
            "last_iteration": int(state.get("last_iteration", -1)),
            "total_timesteps": int(state.get("total_steps", 0)),
            "previous_total_timesteps": prior_total_steps,
            "lifetime_total_timesteps": prior_total_steps + int(state.get("total_steps", 0)),
            "training_steps": summary_training_steps,
            "tensorboard": not args.no_tensorboard,
            "tensorboard_port": args.tensorboard_port if not args.no_tensorboard else None,
            "preview_frames": preview_frames,
            "checkpoint_saved": checkpoint_saved,
            "rgb_timing": summary_rgb_timing,
            "episode_metrics": {
                "completed_episodes": int(state.get("completed_episodes", 0)),
                "success_episodes": int(state.get("success_episodes", 0)),
                "lifted_episodes": int(state.get("lifted_episodes", 0)),
                "success_contact_anomalies": int(state.get("success_contact_anomalies", 0)),
                "clean_success_episodes": int(state.get("clean_success_episodes", 0)),
                "partial_rollout": state.get("partial_rollout"),
                "rolling": _rolling_episode_diagnostics(state.get("episode_window", [])),
            },
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        if status != "error" and checkpoint_saved:
            print(f"Saved: {run_dir / 'model.pt'} ({status})", flush=True)
    if checkpoint_error is not None:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
