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
            self.reward_version = 2
            self.recontact_penalty = 0.1
            self.shaping_gamma = 0.99
            self.start_mode = "home"
            self.cfg = {"action_semantics": "absolute_joint_targets_v1"}
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
        self.assertEqual(tuple(observation["vision"].shape), (2, 4096))
        self.wrapper.step(torch.zeros((2, 7)))
        self.assertEqual(len(self.capture_calls), 1)
        self.assertTrue(torch.allclose(self.wrapper._frame_age, torch.full((2,), 0.02)))
        self.wrapper.step(torch.zeros((2, 7)))
        self.assertEqual(len(self.capture_calls), 2)
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

    def test_actor_policy_does_not_copy_contact_or_cube_truth(self) -> None:
        first = self.base._observation()
        second = self.base._observation()
        # These ranges are the cube/contact/lift fields excluded by the
        # wrapper's explicit actor slices.  The critic remains unchanged and
        # still receives the full clean state from the base environment.
        second["policy"][:, 17:36] += 9000.0
        second["policy"][:, 39:42] += 9000.0
        second["policy"][:, 44:49] += 9000.0
        first_policy = self.wrapper._wrap_observation(first)["policy"]
        second_policy = self.wrapper._wrap_observation(second)["policy"]
        self.assertTrue(torch.equal(first_policy, second_policy))
        self.assertEqual(tuple(self.wrapper._wrap_observation(first)["critic"].shape), (2, 63))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
