"""Optional RGB policy wrapper for the GPU Piper vector environment.

The base :class:`PiperWarpEnv` remains the source of physics, rewards, resets,
and the clean critic observation.  This wrapper owns only a low-resolution
MuJoCo renderer and a frozen RGB encoder.  It samples images at 30 Hz while
the policy continues to step at 50 Hz, reusing the latest feature map and
reporting its age in seconds.

The encoder module is intentionally imported lazily.  The concrete
``RGBFeatureEncoder`` and ``create_rgb_feature_encoder`` API keeps the
optional vision dependency out of state-only help and training paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


class PiperRGBEnv:
    """Wrap a working ``PiperWarpEnv`` with frozen RGB policy features."""

    num_actions = 7
    num_observations = 36
    num_critic_observations = 63
    observation_version = "rgb_resnetv2_front10_v1"

    # The base state is deliberately sliced to exclude exact cube, velocity,
    # contact, lift, and placement fields from the RGB actor.  ``vision`` is a
    # separate CHW 256x4x4 feature map flattened only at the MLP boundary.
    _POLICY_SLICES = (
        slice(0, 17),    # joint/finger proprioception and gripper target
        slice(36, 39),   # known goal command
        slice(42, 44),   # known goal half-size
        slice(49, 55),   # applied arm command
        slice(56, 63),   # previous raw action
    )
    vision_feature_shape = (256, 4, 4)
    vision_feature_dim = 4 * 4 * 256

    def __init__(
        self,
        base_env,
        *,
        encoder=None,
        encoder_checkpoint: dict | None = None,
        encoder_weights: Path | str | None = None,
        camera_fps: float = 30.0,
        image_size: tuple[int, int] = (128, 128),
    ) -> None:
        if camera_fps <= 0:
            raise ValueError("camera_fps must be positive")
        if camera_fps > 50.0:
            raise ValueError("camera_fps cannot exceed the 50 Hz policy clock")
        if len(image_size) != 2 or any(int(value) <= 0 for value in image_size):
            raise ValueError("image_size must contain two positive dimensions")
        self.base_env = base_env
        self.num_envs = int(base_env.num_envs)
        self.num_actions = int(base_env.num_actions)
        if self.num_actions != 7:
            raise ValueError("RGB Piper policy requires the seven direct joint-target actions")
        self.device = base_env.device
        self.model = base_env.model
        self.frame_skip = int(base_env.frame_skip)
        self.max_episode_length = int(base_env.max_episode_length)
        self.num_critic_observations = int(base_env.num_observations)
        self.training_config = base_env.training_config
        self.training_steps = int(base_env.training_steps)
        self.reward_version = int(base_env.reward_version)
        self.recontact_penalty = float(base_env.recontact_penalty)
        self.shaping_gamma = float(base_env.shaping_gamma)
        self.start_mode = base_env.start_mode
        self.camera_fps = float(camera_fps)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.control_hz = 1.0 / (float(self.model.opt.timestep) * self.frame_skip)
        self.control_period = 1.0 / self.control_hz
        self._capture_phase = 0.0
        self._renderer = None
        self._render_data = None
        self._camera = None
        self._render_option = None
        self._closed = False

        checkpoint_vision = self._checkpoint_vision_config(encoder_checkpoint)
        self.encoder = encoder if encoder is not None else self._build_encoder(
            encoder_weights=encoder_weights,
            checkpoint_state=(encoder_checkpoint or {}).get("vision_encoder_state_dict")
            if encoder_checkpoint is not None
            else None,
            offline=encoder_checkpoint is not None,
        )
        to_device = getattr(self.encoder, "to", None)
        if callable(to_device):
            to_device(self.device)
        self._freeze_encoder()
        self.vision_config = self._make_vision_config(checkpoint_vision)
        if checkpoint_vision is not None and self.vision_config != checkpoint_vision:
            raise ValueError(
                "RGB checkpoint encoder metadata does not match the constructed encoder; "
                "refusing to resume with a different feature contract"
            )
        if encoder_checkpoint is not None:
            state = encoder_checkpoint.get("vision_encoder_state_dict")
            if not isinstance(state, dict):
                raise ValueError(
                    "RGB resume requires vision_encoder_state_dict; state-only checkpoints are incompatible"
                )
            loader = getattr(self.encoder, "load_state_dict", None)
            if not callable(loader):
                raise ValueError("RGB encoder cannot load frozen checkpoint weights")
            loader(state, strict=True)
            self._freeze_encoder()

        self.cfg = dict(getattr(base_env, "cfg", {}))
        self.cfg.update(
            {
                "num_obs": self.num_observations,
                "num_critic_obs": self.num_critic_observations,
                "num_actions": self.num_actions,
                "action_semantics": self.cfg.get("action_semantics", "absolute_joint_targets_v1"),
                "observation_version": self.observation_version,
                "obs_groups": {"actor": ["policy", "vision"], "critic": ["critic"]},
                "vision_config": self.vision_config,
                "camera_fps": self.camera_fps,
                "camera_image_size": list(self.image_size),
            }
        )

        # Torch is available in the managed image; import it only after the
        # optional wrapper is selected so host --help remains dependency-free.
        import torch

        self._torch = torch
        self._vision_features = torch.zeros(
            (self.num_envs, self.vision_feature_dim), dtype=torch.float32, device=self.device
        )
        self._frame_age = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

    @staticmethod
    def _checkpoint_vision_config(checkpoint):
        if checkpoint is None:
            return None
        if not isinstance(checkpoint, dict):
            raise ValueError("RGB encoder checkpoint must be a mapping")
        schema = checkpoint.get("piper_schema")
        if not isinstance(schema, dict):
            raise ValueError("RGB checkpoint has no piper_schema")
        metadata = schema.get("vision_config")
        if metadata is None:
            raise ValueError("RGB resume requires a checkpoint with vision_config metadata")
        if not isinstance(metadata, dict):
            raise ValueError("RGB checkpoint vision_config must be a mapping")
        return metadata

    def _build_encoder(self, *, encoder_weights, checkpoint_state, offline: bool):
        try:
            from scripts.piper_rgb_encoder import (
                RGBFeatureEncoder,
                create_rgb_feature_encoder,
            )
        except ImportError as exc:
            raise RuntimeError("RGB mode requires scripts.piper_rgb_encoder") from exc
        if checkpoint_state is not None:
            # This constructor accepts checkpoint_state directly and therefore
            # never creates/downloads a temporary random or pretrained model.
            return RGBFeatureEncoder(pretrained=False, checkpoint_state=checkpoint_state)
        if encoder_weights is not None:
            return create_rgb_feature_encoder(
                pretrained=False,
                checkpoint_path=encoder_weights,
            )
        if offline:
            raise RuntimeError("RGB resume requires frozen encoder state in the RSL checkpoint")
        return create_rgb_feature_encoder(pretrained=True)

    def _freeze_encoder(self) -> None:
        train = getattr(self.encoder, "eval", None)
        if callable(train):
            train()
        parameters = getattr(self.encoder, "parameters", None)
        if callable(parameters):
            for parameter in parameters():
                parameter.requires_grad_(False)

    def _make_vision_config(self, checkpoint_config):
        metadata = getattr(self.encoder, "metadata", None)
        if callable(metadata):
            metadata = metadata()
        if not isinstance(metadata, dict):
            raise ValueError("RGBFeatureEncoder.metadata must be a mapping")
        required = (
            "checkpoint_version",
            "model_name",
            "image_size",
            "feature_dim",
            "front_end",
            "front_stage_depth",
            "front_output_channels",
            "pool_size",
            "feature_shape",
            "normalization_mean",
            "normalization_std",
        )
        missing = [key for key in required if key not in metadata]
        if missing:
            raise ValueError(f"RGB encoder metadata is missing required fields: {missing}")
        result = dict(metadata)
        image_size = result.get("image_size", self.image_size)
        if isinstance(image_size, int):
            image_size = [image_size, image_size]
        else:
            image_size = list(image_size)
        result.update(
            {
                "truncation": "front10",
                "feature_shape": list(self.vision_feature_shape),
                "feature_dim": int(result["feature_dim"]),
                "image_size": image_size,
                "camera_fps": self.camera_fps,
                "camera_name": "policy_rgb",
                "preprocessing": "RGBFeatureEncoder.preprocess",
            }
        )
        if tuple(result["feature_shape"]) != self.vision_feature_shape:
            raise ValueError("RGB encoder must produce a CHW 256x4x4 feature map")
        if result["feature_dim"] != self.vision_feature_dim:
            raise ValueError("RGB encoder feature_dim must be 4096")
        if tuple(result["image_size"]) != self.image_size:
            raise ValueError("RGB encoder image_size differs from the renderer contract")
        if checkpoint_config is not None:
            # The checkpoint metadata is authoritative for resume validation;
            # compare the complete mapping after normalizing tuple/list forms.
            result = self._jsonable(result)
        return result

    @staticmethod
    def _jsonable(value):
        if isinstance(value, dict):
            return {str(key): PiperRGBEnv._jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [PiperRGBEnv._jsonable(item) for item in value]
        if isinstance(value, (str, bool, int)) or value is None:
            return value
        if isinstance(value, float):
            return float(value)
        raise TypeError(f"Unsupported RGB metadata value {type(value).__name__}")

    def _ensure_renderer(self):
        if self._renderer is not None:
            return
        mujoco = self.base_env._mujoco
        width, height = self.image_size
        self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        self._render_data = mujoco.MjData(self.model)
        self._camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self._camera)
        self._camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self._camera.fixedcamid = int(self.model.camera("policy_rgb").id)
        self._render_option = mujoco.MjvOption()

    def _render_rgb(self, indices: Iterable[int]):
        self._ensure_renderer()
        selected = [int(index) for index in indices]
        if any(index < 0 or index >= self.num_envs for index in selected):
            raise IndexError("RGB render index outside the batched environment")
        self.base_env._wp.synchronize()
        frames = []
        for world_id in selected:
            self.base_env._mjw.get_data_into(
                self._render_data,
                self.model,
                self.base_env._warp_data,
                world_id=world_id,
            )
            self._renderer.update_scene(
                self._render_data,
                camera=self._camera,
                scene_option=self._render_option,
            )
            frames.append(self._renderer.render().copy())
        return frames

    def _encode(self, frames):
        import numpy as np

        if not frames:
            return self._torch.empty((0, self.vision_feature_dim), device=self.device)
        batch = np.stack(frames, axis=0)
        encode = getattr(self.encoder, "encode", None)
        if callable(encode):
            features = encode(batch)
        else:
            tensor = self._torch.as_tensor(batch, device=self.device)
            features = self.encoder(tensor)
        if not isinstance(features, self._torch.Tensor):
            features = self._torch.as_tensor(features, device=self.device)
        features = features.to(device=self.device, dtype=self._torch.float32)
        if features.ndim == 4 and tuple(features.shape[1:]) == (4, 4, 256):
            features = features.permute(0, 3, 1, 2)
        if features.ndim == 4 and tuple(features.shape[1:]) != self.vision_feature_shape:
            raise ValueError(f"RGB encoder returned unsupported feature shape {tuple(features.shape)}")
        features = features.reshape(features.shape[0], -1)
        if features.shape[-1] != self.vision_feature_dim:
            raise ValueError(
                f"RGB encoder returned {features.shape[-1]} features; expected {self.vision_feature_dim}"
            )
        return features.detach()

    def _capture(self, indices):
        indices = [int(index) for index in indices]
        if not indices:
            return
        features = self._encode(self._render_rgb(indices))
        index_tensor = self._torch.as_tensor(indices, dtype=self._torch.long, device=self.device)
        self._vision_features[index_tensor] = features
        self._frame_age[index_tensor] = 0.0

    def _wrap_observation(self, observation):
        from tensordict import TensorDict

        base_policy = observation["policy"]
        critic = observation["critic"]
        proprio = self._torch.cat([base_policy[:, part] for part in self._POLICY_SLICES], dim=-1)
        policy = self._torch.cat([proprio, self._frame_age.unsqueeze(1)], dim=-1)
        if policy.shape[-1] != self.num_observations:
            raise RuntimeError(f"RGB policy observation has width {policy.shape[-1]}, expected {self.num_observations}")
        return TensorDict(
            {"policy": policy, "vision": self._vision_features.clone(), "critic": critic.clone()},
            batch_size=[self.num_envs],
            device=self.device,
        )

    def reset(self, seed: int | None = None):
        if self._closed:
            raise RuntimeError("PiperRGBEnv is closed")
        observation = self.base_env.reset(seed=seed)
        self._capture_phase = 0.0
        self._frame_age.zero_()
        self._capture(range(self.num_envs))
        self.training_steps = int(self.base_env.training_steps)
        self.cfg["training_steps"] = self.training_steps
        self.cfg["curriculum_stage"] = self.base_env.curriculum_stage
        return self._wrap_observation(observation)

    def get_observations(self):
        return self._wrap_observation(self.base_env.get_observations())

    def step(self, actions):
        observation, rewards, dones, infos = self.base_env.step(actions)
        self._capture_phase += self.camera_fps
        self._frame_age += self.control_period
        due = self._capture_phase >= self.control_hz
        if due:
            self._capture_phase -= self.control_hz
            indices = list(range(self.num_envs))
        else:
            indices = []
        done_indices = self._torch.nonzero(dones.to(dtype=self._torch.bool), as_tuple=False).flatten().tolist()
        # A partial reset must receive a fresh image even when the periodic
        # camera phase is not due; this prevents prior-episode feature leakage.
        self._capture(sorted(set(indices).union(done_indices)))
        self.training_steps = int(self.base_env.training_steps)
        self.cfg["training_steps"] = self.training_steps
        self.cfg["curriculum_stage"] = self.base_env.curriculum_stage
        return self._wrap_observation(observation), rewards, dones, infos

    def checkpoint_state(self) -> dict:
        state_dict = getattr(self.encoder, "state_dict", None)
        if not callable(state_dict):
            raise ValueError("RGB encoder must expose state_dict() for checkpointing")
        frozen_state = {}
        for key, value in state_dict().items():
            detach = getattr(value, "detach", None)
            if callable(detach):
                value = detach().cpu()
            frozen_state[key] = value
        return {
            "vision_encoder_state_dict": frozen_state,
            "vision_config": self.vision_config,
        }

    def set_training_steps(self, count: int):
        result = self.base_env.set_training_steps(count)
        self.training_steps = int(self.base_env.training_steps)
        self.cfg["training_steps"] = self.training_steps
        self.cfg["curriculum_stage"] = result
        return result

    def render_frames(self, indices=(0, 1, 2, 3)):
        return self.base_env.render_frames(indices)

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        close = getattr(self.encoder, "close", None)
        if callable(close):
            close()
        self.base_env.close()

    def __getattr__(self, name: str) -> Any:
        # Keep the vector-environment API and reward/debug fields transparent
        # while retaining ownership of RGB renderer/encoder resources here.
        if name == "base_env":
            raise AttributeError(name)
        return getattr(self.base_env, name)
