"""Optional RGB policy wrapper for the GPU Piper vector environment.

The base :class:`PiperWarpEnv` remains the source of physics, rewards, resets,
and the clean critic observation.  This wrapper owns a 640x480 MuJoCo wrist
camera renderer and a frozen RGB encoder.  It samples images at 30 Hz while
the policy continues to step at 50 Hz, center-crops the raw frame to the
existing 128x128 encoder input, reuses the latest feature map, and reports its
age in seconds.

The encoder module is intentionally imported lazily.  The concrete
``RGBFeatureEncoder`` and ``create_rgb_feature_encoder`` API keeps the
optional vision dependency out of state-only help and training paths.
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Iterable


class PiperRGBEnv:
    """Wrap a working ``PiperWarpEnv`` with frozen RGB policy features."""

    num_actions = 7
    num_observations = 55
    num_critic_observations = 63
    observation_version = "rgb_resnetv2_wrist_d455_object_xyz_hidden_v2"

    # Hide absolute cube XYZ and the two direct position differences that
    # reconstruct it. Retain every other base policy field (orientation,
    # velocity, contact, lift, placement, time, commands, and action history)
    # so direct object XYZ paths are omitted while the remaining state
    # contract stays unchanged. Other retained fields may provide indirect
    # cues by design. ``vision`` is a separate CHW 256x4x4 map.
    _POLICY_SLICES = (
        slice(0, 20),    # joint/finger state, TCP, and gripper target
        slice(23, 33),   # cube orientation and velocity
        slice(36, 39),   # known goal command
        slice(42, 63),   # goal size, contact/progress, commands, action history
    )
    vision_feature_shape = (256, 4, 4)
    vision_feature_dim = 4 * 4 * 256

    # MuJoCo's visual scene update for this Piper model consumes only these
    # pose arrays.  Keeping the list explicit is important: mujoco_warp's
    # generic get_data_into() converts every q/constraint/solver array to the
    # host for each world, although the renderer never reads those arrays.
    _RENDER_SNAPSHOT_FIELDS = (
        "geom_xpos",
        "geom_xmat",
        "site_xpos",
        "site_xmat",
        "cam_xpos",
        "cam_xmat",
        "light_xpos",
        "light_xdir",
    )
    # The snapshot is deliberately specialized to the compiled Piper scene
    # used by the RGB policy.  Keep this contract explicit so a future model
    # edit cannot silently render stale/unpopulated state through the fast
    # path.  The generic MuJoCo-Warp transfer remains the fallback for other
    # scenes in callers that need one.
    _RENDER_MODEL_COUNTS = {
        "nbody": 15,
        "ngeom": 94,
        "nsite": 6,
        "ncam": 1,
        "nlight": 2,
        "nq": 15,
        "nv": 14,
        "nu": 7,
        "nmocap": 1,
        "neq": 1,
        "nflex": 0,
        "nskin": 0,
        "ntendon": 0,
        "nsensor": 0,
        "nplugin": 0,
    }
    # MuJoCo 3.10.0's MjvOption defaults.  These defaults leave visual
    # geoms/sites enabled while all renderer-decor paths that would consume
    # additional MjData fields remain disabled for this scene.
    _RENDER_DEFAULT_FLAGS = (
        False, True, False, False, False, False, False, True,
        True, False, False, False, False, True, False, False,
        False, False, False, False, False, False, True, True,
        False, True, False, True, False, False, False,
    )

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
            raise ValueError("RGB Piper policy requires the seven incremental joint-target actions")
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
        from scripts.piper_camera import POLICY_RENDER_SIZE

        self.camera_render_size = tuple(int(value) for value in POLICY_RENDER_SIZE)
        self.control_hz = 1.0 / (float(self.model.opt.timestep) * self.frame_skip)
        self.control_period = 1.0 / self.control_hz
        self._capture_phase = 0.0
        self._renderer = None
        self._render_data = None
        self._camera = None
        self._render_option = None
        self._closed = False
        self._render_timing = {
            "render_calls": 0,
            "render_frames": 0,
            "render_field_transfers": 0,
            "render_transfer_seconds": 0.0,
            "render_seconds": 0.0,
            "encode_calls": 0,
            "encode_frames": 0,
            # This is host dispatch time only.  The frozen CUDA encoder may
            # still be executing after the call returns; callers that need a
            # device-runtime measurement must synchronize or use CUDA events
            # around their benchmark rather than treating this as full GPU
            # execution time.
            "encode_host_dispatch_seconds": 0.0,
        }

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
                "RGB checkpoint camera/encoder metadata does not match the constructed "
                "wrist-D455 observation contract; refusing to resume with a different "
                "feature contract"
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
                "action_semantics": self.cfg.get("action_semantics", "incremental_joint_targets_v1"),
                "observation_version": self.observation_version,
                "obs_groups": {"actor": ["policy", "vision"], "critic": ["critic"]},
                "vision_config": self.vision_config,
                "camera_fps": self.camera_fps,
                "camera_image_size": list(self.image_size),
                "camera_render_size": list(self.camera_render_size),
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
        observation_version = schema.get("observation_version")
        if observation_version != PiperRGBEnv.observation_version:
            raise ValueError(
                "RGB checkpoint observation_version is incompatible with the current "
                f"wrist-D455 actor contract: expected {PiperRGBEnv.observation_version!r}, "
                f"got {observation_version!r}; start a fresh RGB PPO run"
            )
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
        from scripts.piper_camera import camera_metadata

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
                "camera_config": camera_metadata(),
            }
        )
        if tuple(result["feature_shape"]) != self.vision_feature_shape:
            raise ValueError("RGB encoder must produce a CHW 256x4x4 feature map")
        if result["feature_dim"] != self.vision_feature_dim:
            raise ValueError("RGB encoder feature_dim must be 4096")
        if tuple(result["image_size"]) != self.image_size:
            raise ValueError("RGB encoder image_size differs from the renderer contract")
        if tuple(result["camera_config"]["render_resolution"]) != self.camera_render_size:
            raise ValueError("D455 camera render resolution differs from the renderer contract")
        if float(result["camera_config"]["native_fov_deg"]["vertical"]) != 65.0:
            raise ValueError("D455 camera vertical field of view must remain 65 degrees")
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
        self._render_option = mujoco.MjvOption()
        # D455 RGB sees visual geometry only.  The model's group-3 collision
        # proxies are useful to physics and the free preview but must not
        # leak into the policy observation.
        self._render_option.geomgroup[3] = 0
        self._verify_render_contract(mujoco)
        width, height = self.camera_render_size
        self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        self._render_data = mujoco.MjData(self.model)
        self._camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self._camera)
        self._camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self._camera.fixedcamid = int(self.model.camera("policy_rgb").id)

    def _verify_render_contract(self, mujoco) -> None:
        """Reject model/option changes that would make the field list incomplete."""

        mismatches = []
        version = str(getattr(mujoco, "__version__", ""))
        if version != "3.10.0":
            mismatches.append(f"mujoco_version={version!r} (expected '3.10.0')")

        for name, expected in self._RENDER_MODEL_COUNTS.items():
            actual = int(getattr(self.model, name, -1))
            if actual != expected:
                mismatches.append(f"{name}={actual} (expected {expected})")

        # Resolve the named dynamic elements before creating the Renderer.  A
        # fixed policy camera and target-body-com light both require their
        # derived pose arrays in the snapshot; changing either invalidates the
        # eight-field assumption.
        try:
            camera_id = int(self.model.camera("policy_rgb").id)
            light_id = int(self.model.light("spotlight").id)
            target_body = int(self.model.body("link8").id)
            goal_body = int(self.model.body("goal_area").id)
            self.model.site("grasp_tcp")
            self.model.site("goal_fill")
        except Exception as exc:  # pragma: no cover - guarded model contract
            mismatches.append(f"required named visual element missing: {exc}")
        else:
            if camera_id != 0:
                mismatches.append(f"policy_rgb camera id={camera_id} (expected 0)")
            if int(self.model.cam_mode[camera_id]) != int(mujoco.mjtCamLight.mjCAMLIGHT_FIXED):
                mismatches.append("policy_rgb camera is not fixed")
            if light_id != 0:
                mismatches.append(f"spotlight id={light_id} (expected 0)")
            if int(self.model.light_mode[light_id]) != int(mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODYCOM):
                mismatches.append("spotlight is not targetbodycom")
            if int(self.model.light_targetbodyid[light_id]) != target_body:
                mismatches.append("spotlight target is not link8")
            if int(self.model.body_mocapid[goal_body]) != 0:
                mismatches.append("goal_area is not mocap body 0")

        flags = tuple(bool(value) for value in self._render_option.flags)
        if flags != self._RENDER_DEFAULT_FLAGS:
            mismatches.append(f"MjvOption.flags={flags!r} differs from MuJoCo 3.10 defaults")
        if tuple(int(value) for value in self._render_option.geomgroup) != (1, 1, 1, 0, 0, 0):
            mismatches.append("geomgroup is not the default with collision group 3 disabled")
        if tuple(int(value) for value in self._render_option.sitegroup) != (1, 1, 1, 0, 0, 0):
            mismatches.append("sitegroup differs from the required default")
        if tuple(int(value) for value in self._render_option.tendongroup) != (1, 1, 1, 0, 0, 0):
            mismatches.append("tendongroup differs from the required default")
        if int(self._render_option.frame) != int(mujoco.mjtFrame.mjFRAME_NONE):
            mismatches.append("MjvOption.frame is not mjFRAME_NONE")
        if int(self._render_option.label) != int(mujoco.mjtLabel.mjLABEL_NONE):
            mismatches.append("MjvOption.label is not mjLABEL_NONE")

        if mismatches:
            raise RuntimeError(
                "Piper RGB render-only snapshot contract mismatch; refusing fast path: "
                + "; ".join(mismatches)
            )

    @property
    def render_timing(self) -> dict[str, float | int]:
        """Return cumulative render/encoder transfer timings for diagnostics."""

        return dict(self._render_timing)

    def _render_snapshot(self, selected: list[int]) -> dict[str, Any]:
        """Copy only visual pose arrays for selected worlds to the host.

        ``mujoco_warp.get_data_into`` is intentionally not used here.  Its
        field-by-field ``.numpy()[world_id]`` calls copy the complete leading
        world dimension once per selected world.  The direct Torch views below
        gather each visual field once, then copy only the requested rows.
        """

        if not selected:
            return {}
        transfer_started = time.perf_counter()
        self.base_env._wp.synchronize()
        world_ids = self._torch.as_tensor(
            selected,
            dtype=self._torch.long,
            device=self.device,
        )
        data = self.base_env._warp_data
        to_torch = self.base_env._wp.to_torch
        snapshot = {}
        for field in self._RENDER_SNAPSHOT_FIELDS:
            source = getattr(data, field, None)
            if source is None:
                raise RuntimeError(f"MuJoCo-Warp data is missing required render field {field!r}")
            device_field = to_torch(source)
            if device_field.ndim == 0 or int(device_field.shape[0]) != self.num_envs:
                raise RuntimeError(
                    f"MuJoCo-Warp render field {field!r} has leading shape "
                    f"{tuple(device_field.shape)!r}; expected {self.num_envs} worlds"
                )
            snapshot[field] = (
                device_field.index_select(0, world_ids)
                .detach()
                .cpu()
                .numpy()
            )
        self._render_timing["render_field_transfers"] += len(self._RENDER_SNAPSHOT_FIELDS)
        self._render_timing["render_transfer_seconds"] += time.perf_counter() - transfer_started
        return snapshot

    def _populate_render_data(self, snapshot: dict[str, Any], row: int) -> None:
        """Populate the reusable host MjData from one visual snapshot row."""

        for field in self._RENDER_SNAPSHOT_FIELDS:
            target = getattr(self._render_data, field)
            value = snapshot[field][row]
            # MuJoCo stores matrix arrays flattened as (..., 9), while Warp's
            # Torch view exposes (..., 3, 3).  Reshape handles both forms and
            # performs one host-to-host copy into the reusable MjData.
            target[...] = value.reshape(target.shape)

    def _render_rgb(self, indices: Iterable[int]):
        self._ensure_renderer()
        selected = [int(index) for index in indices]
        if any(index < 0 or index >= self.num_envs for index in selected):
            raise IndexError("RGB render index outside the batched environment")
        if not selected:
            return []
        snapshot = self._render_snapshot(selected)
        render_started = time.perf_counter()
        frames = []
        try:
            for row in range(len(selected)):
                self._populate_render_data(snapshot, row)
                self._renderer.update_scene(
                    self._render_data,
                    camera=self._camera,
                    scene_option=self._render_option,
                )
                frames.append(self._renderer.render().copy())
        finally:
            self._render_timing["render_calls"] += 1
            self._render_timing["render_frames"] += len(frames)
            self._render_timing["render_seconds"] += time.perf_counter() - render_started
        return frames

    def _encode(self, frames):
        import numpy as np

        if not frames:
            return self._torch.empty((0, self.vision_feature_dim), device=self.device)
        batch = np.stack(frames, axis=0)
        encode_started = time.perf_counter()
        try:
            encode = getattr(self.encoder, "encode", None)
            if callable(encode):
                features = encode(batch)
            else:
                tensor = self._torch.as_tensor(batch, device=self.device)
                features = self.encoder(tensor)
        finally:
            self._render_timing["encode_calls"] += 1
            self._render_timing["encode_frames"] += len(frames)
            self._render_timing["encode_host_dispatch_seconds"] += (
                time.perf_counter() - encode_started
            )
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
