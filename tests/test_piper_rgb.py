"""Contract tests for the optional simulated-RGB ResNetV2 front-end.

The base simulator image intentionally does not install the vision extras, so
the model tests skip there.  They run in the vision image with explicit random
test weights; no test silently downloads or labels random weights as
pretrained.  The tests exercise preprocessing, frozen/eval behavior,
spatially sensitive features, and strict offline checkpoint restoration.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import unittest
import importlib.util
from copy import deepcopy

try:  # The base simulator image may not include the optional extras.
    import torch
except ImportError:  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]

try:
    import numpy as np
except ImportError:  # pragma: no cover - environment dependent
    np = None  # type: ignore[assignment]

if torch is not None and importlib.util.find_spec("timm") is not None:
    try:
        from scripts.piper_rgb_encoder import RGBFeatureEncoder
    except (ImportError, RuntimeError):  # missing timm is handled as a skip
        RGBFeatureEncoder = None  # type: ignore[assignment,misc]
else:
    RGBFeatureEncoder = None  # type: ignore[assignment,misc]

if torch is not None and np is not None and importlib.util.find_spec("tensordict") is not None:
    try:
        from scripts.piper_rgb_env import PiperRGBEnv
    except ImportError:  # pragma: no cover - source tree dependent
        PiperRGBEnv = None  # type: ignore[assignment,misc]
else:
    PiperRGBEnv = None  # type: ignore[assignment,misc]

_TorchModule = torch.nn.Module if torch is not None else object


def _vision_reason() -> str:
    if torch is None:
        return "torch is not installed"
    if RGBFeatureEncoder is None:
        return "optional timm vision dependency is not installed"
    return ""


VISION_TESTS = unittest.skipUnless(RGBFeatureEncoder is not None, _vision_reason())


@VISION_TESTS
class RGBFeatureEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(7)
        # Random weights are explicit test-only fixtures.  Production callers
        # use the default pretrained=True path or a strict saved checkpoint.
        cls.encoder = RGBFeatureEncoder(pretrained=False, test_only=True)

    def test_contract_and_frozen_eval_behavior(self) -> None:
        encoder = self.encoder
        self.assertEqual(encoder.MODEL_NAME, "resnetv2_50x1_bit.goog_in21k_ft_in1k")
        self.assertEqual(encoder.image_size, 128)
        self.assertEqual(encoder.feature_dim, 4096)
        self.assertFalse(encoder.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in encoder.parameters()))
        encoder.train()
        self.assertFalse(encoder.training)
        self.assertTrue(all(not child.training for child in encoder.children()))

    def test_center_crop_resize_and_encode_are_rgb_uint8_contract(self) -> None:
        # HWC input is intentionally accepted as a one-frame convenience.
        image = torch.zeros((160, 240, 3), dtype=torch.uint8)
        image[20:140, 50:190, 0] = 255
        preprocessed = self.encoder.preprocess(image)
        self.assertEqual(tuple(preprocessed.shape), (1, 3, 128, 128))
        self.assertEqual(preprocessed.dtype, torch.float32)
        self.assertGreaterEqual(float(preprocessed.min()), 0.0)
        self.assertLessEqual(float(preprocessed.max()), 1.0)
        features = self.encoder.encode(image)
        self.assertEqual(tuple(features.shape), (1, 4096))
        self.assertFalse(features.requires_grad)
        self.assertEqual(features.dtype, torch.float32)

        with self.assertRaises(TypeError):
            self.encoder.encode(image.to(dtype=torch.float32))
        with self.assertRaises(ValueError):
            self.encoder.encode(torch.zeros((128, 128, 4), dtype=torch.uint8))

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_preprocess_and_features_match_cpu_reference_with_bounded_drift(self) -> None:
        # 65 frames exercises the bounded device-side preprocessing chunks and
        # keeps the raw 640x480 capture contract in the parity check.
        gpu = RGBFeatureEncoder(
            pretrained=False,
            checkpoint_state=self.encoder.state_dict(),
        ).cuda()
        frames = torch.randint(0, 256, (65, 480, 640, 3), dtype=torch.uint8)
        with torch.inference_mode():
            cpu_preprocessed = self.encoder.preprocess(frames)
            gpu_preprocessed = gpu.preprocess(frames, device=torch.device("cuda")).cpu()
            # Production's old path performed CPU crop/resize, copied the
            # resulting 128x128 tensor to CUDA, then ran this same frozen GPU
            # front.  Compare that reference against device preprocessing so
            # CUDA convolution-order differences are excluded.
            reference = cpu_preprocessed.to(device="cuda")
            reference = (reference - gpu.normalization_mean) / gpu.normalization_std
            reference = gpu.pool(gpu.encoder(reference)).flatten(1).cpu()
            accelerated = gpu.encode(frames).cpu()
        preprocessing_delta = (cpu_preprocessed - gpu_preprocessed).abs()
        feature_delta = (reference - accelerated).abs()
        self.assertLessEqual(float(preprocessing_delta.max()), 3e-7)
        # The saved pretrained front shows a max delta of about 0.0051 and
        # mean delta of about 0.00044 on RTX 5090; retain a modest margin for
        # CUDA kernel/driver variation.
        self.assertLessEqual(float(feature_delta.max()), 1e-2)
        self.assertLessEqual(float(feature_delta.mean()), 1e-3)
        del gpu
        torch.cuda.empty_cache()

    def test_spatial_translation_changes_features(self) -> None:
        # Two identical patches at different positions remain in the central
        # crop.  A pooled 4x4 front must preserve enough spatial information
        # for the random test front to distinguish them.
        left = torch.zeros((192, 256, 3), dtype=torch.uint8)
        right = torch.zeros_like(left)
        left[72:120, 74:122] = torch.tensor([255, 100, 20], dtype=torch.uint8)
        right[72:120, 150:198] = torch.tensor([255, 100, 20], dtype=torch.uint8)
        left_features = self.encoder.encode(left)
        right_features = self.encoder.encode(right)
        self.assertGreater(float(torch.max(torch.abs(left_features - right_features))), 1e-6)

    def test_checkpoint_round_trip_is_offline_and_strict(self) -> None:
        encoder = self.encoder
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rgb_encoder.pt"
            encoder.save_checkpoint(path)
            restored = RGBFeatureEncoder.from_checkpoint(path)
            sample = torch.randint(0, 256, (2, 150, 220, 3), dtype=torch.uint8)
            torch.testing.assert_close(encoder.encode(sample), restored.encode(sample))
            self.assertEqual(restored.metadata, encoder.metadata)
            self.assertTrue(all(not p.requires_grad for p in restored.parameters()))
            self.assertFalse(restored.training)

            payload = encoder.checkpoint_payload()
            payload["metadata"]["image_size"] = 64
            bad_metadata = Path(temporary) / "bad_metadata.pt"
            torch.save(payload, bad_metadata)
            with self.assertRaises(ValueError):
                RGBFeatureEncoder.from_checkpoint(bad_metadata)

            bad_state = encoder.checkpoint_payload()
            bad_state["state_dict"].pop(next(iter(bad_state["state_dict"])))
            missing_key = Path(temporary) / "missing_key.pt"
            torch.save(bad_state, missing_key)
            with self.assertRaises((RuntimeError, ValueError)):
                RGBFeatureEncoder.from_checkpoint(missing_key)

    def test_random_weights_require_explicit_test_or_checkpoint_mode(self) -> None:
        with self.assertRaises(ValueError):
            RGBFeatureEncoder(pretrained=False)


@unittest.skipUnless(PiperRGBEnv is not None, "torch, NumPy, and tensordict are not installed")
class PiperRGBEnvTests(unittest.TestCase):
    """Exercise the sim-render cache without constructing a MuJoCo world."""

    class _Opt:
        timestep = 0.002

    class _Model:
        def __init__(self):
            self.opt = type("Opt", (), {"timestep": 0.002})()

    class _BaseEnv:
        def __init__(self) -> None:
            self.num_envs = 2
            self.num_actions = 7
            self.device = torch.device("cpu")
            self.model = PiperRGBEnvTests._Model()
            self.frame_skip = 10
            self.max_episode_length = 600
            self.num_observations = 63
            self.training_config = object()
            self.training_steps = 0
            self.curriculum_stage = 0
            self.reward_version = 3
            self.recontact_penalty = 0.1
            self.shaping_gamma = 0.99
            self.start_mode = "home"
            self.cfg = {"action_semantics": "incremental_joint_targets_v1"}
            self.next_dones = torch.zeros(2, dtype=torch.bool)
            self.base_policy = torch.arange(63, dtype=torch.float32).repeat(2, 1)

        def _observation(self):
            return {
                "policy": self.base_policy.clone(),
                "critic": (self.base_policy + 1000).clone(),
            }

        def reset(self, seed=None):
            del seed
            self.training_steps = 0
            return self._observation()

        def get_observations(self):
            return self._observation()

        def step(self, actions):
            del actions
            self.training_steps += 1
            dones = self.next_dones.clone()
            self.next_dones.zero_()
            return self._observation(), torch.zeros(2), dones, {}

        def set_training_steps(self, count):
            self.training_steps = int(count)
            return 0

        def close(self):
            return None

    class _Encoder(_TorchModule):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.ones(1))

        @property
        def metadata(self):
            return {
                "checkpoint_version": 1,
                "model_name": "resnetv2_50x1_bit.goog_in21k_ft_in1k",
                "image_size": 128,
                "feature_dim": 4096,
                "front_end": "stem+stages.0",
                "front_stage_depth": 3,
                "front_output_channels": 256,
                "pool_size": [4, 4],
                "feature_shape": [256, 4, 4],
                "normalization_mean": [0.5, 0.5, 0.5],
                "normalization_std": [0.5, 0.5, 0.5],
            }

        def encode(self, frames):
            return torch.ones((len(frames), 4096), dtype=torch.float32)

    def setUp(self) -> None:
        self.base = self._BaseEnv()
        self.wrapper = PiperRGBEnv(
            self.base,
            encoder=self._Encoder(),
            camera_fps=30.0,
        )
        self.capture_calls = []

        def capture(indices):
            indices = [int(index) for index in indices]
            if not indices:
                return
            self.capture_calls.append(indices)
            index = torch.as_tensor(indices, dtype=torch.long)
            self.wrapper._vision_features[index] = 1.0
            self.wrapper._frame_age[index] = 0.0

        # Rendering itself belongs to the GPU MuJoCo integration.  This test
        # replaces only the image producer while exercising the real cache,
        # age, observation grouping, and partial-reset logic.
        self.wrapper._capture = capture

    def tearDown(self) -> None:
        self.wrapper.close()

    def test_30hz_cache_reused_between_50hz_policy_ticks(self) -> None:
        observation = self.wrapper.reset(seed=3)
        self.assertEqual(self.capture_calls, [[0, 1]])
        self.assertEqual(tuple(observation["policy"].shape), (2, 55))
        self.assertEqual(tuple(observation["vision"].shape), (2, 4096))
        self.wrapper.step(torch.zeros((2, 7)))
        self.assertEqual(len(self.capture_calls), 1)
        self.assertTrue(torch.allclose(self.wrapper._frame_age, torch.full((2,), 0.02)))
        self.wrapper.step(torch.zeros((2, 7)))
        self.assertEqual(len(self.capture_calls), 2)
        self.assertEqual(self.capture_calls[-1], [0, 1])
        self.assertTrue(torch.allclose(self.wrapper._frame_age, torch.zeros(2)))
        self.wrapper.step(torch.zeros((2, 7)))
        self.assertEqual(len(self.capture_calls), 2)
        self.assertTrue(torch.allclose(self.wrapper._frame_age, torch.full((2,), 0.02)))
        self.wrapper.step(torch.zeros((2, 7)))
        self.assertEqual(len(self.capture_calls), 3)
        self.assertEqual(self.capture_calls[-1], [0, 1])
        self.assertTrue(torch.allclose(self.wrapper._frame_age, torch.zeros(2)))
        self.wrapper.step(torch.zeros((2, 7)))
        self.assertEqual(len(self.capture_calls), 4)
        self.assertEqual(self.capture_calls[-1], [0, 1])
        self.assertTrue(torch.allclose(self.wrapper._frame_age, torch.zeros(2)))

    def test_partial_reset_refreshes_only_done_world_feature(self) -> None:
        self.wrapper.reset(seed=4)
        self.wrapper._capture_phase = 0.0
        self.wrapper._frame_age[:] = 0.4
        self.base.next_dones[0] = True
        self.wrapper.step(torch.zeros((2, 7)))
        self.assertEqual(self.capture_calls[-1], [0])
        self.assertAlmostEqual(float(self.wrapper._frame_age[0]), 0.0)
        self.assertAlmostEqual(float(self.wrapper._frame_age[1]), 0.42, places=6)

    def test_actor_policy_excludes_direct_cube_position_paths(self) -> None:
        first = self.base._observation()
        second = self.base._observation()
        # Absolute cube XYZ and both direct position differences are hidden;
        # orientation, velocity, contact, lift, placement, time, commands,
        # and action history remain in the actor policy state.
        hidden = (slice(20, 23), slice(33, 36), slice(39, 42))
        for part in hidden:
            second["policy"][:, part] += 9000.0
        first_policy = self.wrapper._wrap_observation(first)["policy"]
        second_policy = self.wrapper._wrap_observation(second)["policy"]
        self.assertTrue(torch.equal(first_policy, second_policy))

        retained = self.base._observation()
        retained["policy"][:, 23:33] += 7.0
        retained_policy = self.wrapper._wrap_observation(retained)["policy"]
        self.assertFalse(torch.equal(first_policy, retained_policy))
        self.assertTrue(torch.equal(first_policy[:, 54], self.wrapper._frame_age))
        self.assertEqual(tuple(self.wrapper._wrap_observation(first)["critic"].shape), (2, 63))

    def test_policy_state_matches_retained_base_fields_in_order(self) -> None:
        observation = self.base._observation()
        wrapped = self.wrapper._wrap_observation(observation)
        expected = torch.cat(
            [
                observation["policy"][:, 0:20],
                observation["policy"][:, 23:33],
                observation["policy"][:, 36:39],
                observation["policy"][:, 42:63],
                self.wrapper._frame_age.unsqueeze(1),
            ],
            dim=-1,
        )
        self.assertTrue(torch.equal(wrapped["policy"], expected))
        self.assertEqual(self.wrapper.observation_version, "rgb_resnetv2_wrist_d455_object_xyz_hidden_v2")

    def test_old_rgb_observation_version_is_rejected_and_current_resumes(self) -> None:
        current_config = deepcopy(self.wrapper.vision_config)
        current_checkpoint = {
            "piper_schema": {
                "observation_version": self.wrapper.observation_version,
                "vision_config": current_config,
            },
            "vision_encoder_state_dict": self.wrapper.encoder.state_dict(),
        }
        resumed = PiperRGBEnv(
            self.base,
            encoder=self._Encoder(),
            encoder_checkpoint=current_checkpoint,
            camera_fps=30.0,
        )
        try:
            self.assertEqual(resumed.num_observations, 55)
            self.assertEqual(resumed.observation_version, self.wrapper.observation_version)
        finally:
            resumed.close()
        old_checkpoint = {
            "piper_schema": {
                "observation_version": "rgb_resnetv2_wrist_d455_v1",
                "vision_config": current_config,
            },
            "vision_encoder_state_dict": self.wrapper.encoder.state_dict(),
        }
        with self.assertRaisesRegex(ValueError, "observation_version"):
            PiperRGBEnv(
                self.base,
                encoder=self._Encoder(),
                encoder_checkpoint=old_checkpoint,
                camera_fps=30.0,
            )

    def test_external_view_checkpoint_is_rejected_for_wrist_d455(self) -> None:
        old_config = deepcopy(self.wrapper.vision_config)
        # The old external-view format had no camera_config field at all.
        # Keep the actor observation version current so this specifically
        # exercises the camera contract mismatch.
        old_config.pop("camera_config", None)
        checkpoint = {
            "piper_schema": {
                "observation_version": self.wrapper.observation_version,
                "vision_config": old_config,
            },
            "vision_encoder_state_dict": self.wrapper.encoder.state_dict(),
        }
        with self.assertRaisesRegex(ValueError, "different feature contract"):
            PiperRGBEnv(
                self.base,
                encoder=self._Encoder(),
                encoder_checkpoint=checkpoint,
                camera_fps=30.0,
            )

    def test_camera_metadata_distinguishes_raw_capture_from_encoder_input(self) -> None:
        config = self.wrapper.vision_config["camera_config"]
        self.assertEqual(config["render_resolution"], [640, 480])
        self.assertEqual(config["encoder_image_size"], [128, 128])
        self.assertFalse(config["render_aspect_preserved"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
