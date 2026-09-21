"""Frozen RGB features for the optional Piper simulated-RGB observation path.

The encoder deliberately owns only image preprocessing and a frozen front of
the official ``timm`` ResNetV2 BiT model.  It does not know about MuJoCo,
RealSense, robot commands, or PPO.  The default constructor loads the
pretrained weights and propagates download/load failures; an offline random
model is available only through the explicit ``test_only=True`` opt-in or a
saved checkpoint state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


MODEL_NAME = "resnetv2_50x1_bit.goog_in21k_ft_in1k"
IMAGE_SIZE = 128
FEATURE_DIM = 256 * 4 * 4
CHECKPOINT_VERSION = 1
_FRONT_STAGE_DEPTH = 3
_FRONT_OUTPUT_CHANNELS = 256
_POOL_SIZE = (4, 4)
# The pooled tensor is CHW before flattening.  The wrapper accepts the
# flattened 4096-D result and records this source layout in its metadata.
_FEATURE_SHAPE = (256, 4, 4)
_MEAN = (0.5, 0.5, 0.5)
_STD = (0.5, 0.5, 0.5)


def _load_timm():
    """Import timm lazily so the base simulation path remains optional."""

    try:
        import timm
    except ImportError as exc:  # pragma: no cover - exercised without vision extras
        raise RuntimeError(
            "RGBFeatureEncoder requires the optional vision dependencies; "
            "install requirements-vision.txt"
        ) from exc
    return timm


def _load_torch_checkpoint(path: Path) -> Mapping[str, Any]:
    """Load a checkpoint on CPU across supported torch versions."""

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch versions before the weights_only keyword
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError(f"RGB encoder checkpoint must contain a mapping, got {type(payload)!r}")
    return payload


class RGBFeatureEncoder(nn.Module):
    """Frozen ResNetV2 BiT front-end returning 4096 spatial features.

    Parameters
    ----------
    pretrained:
        Load ``MODEL_NAME`` from timm/Hugging Face.  Failures are surfaced to
        the caller so an unavailable download cannot silently create random
        features.
    checkpoint_state:
        A state dictionary previously produced by :meth:`state_dict`.  This
        path constructs the architecture without a network request and loads
        it strictly.
    test_only:
        Explicitly permit random weights for deterministic unit tests.  This
        flag is intentionally required when ``pretrained=False`` without a
        checkpoint state.
    """

    MODEL_NAME = MODEL_NAME
    image_size = IMAGE_SIZE
    feature_dim = FEATURE_DIM
    feature_shape = _FEATURE_SHAPE
    checkpoint_version = CHECKPOINT_VERSION

    def __init__(
        self,
        *,
        pretrained: bool = True,
        checkpoint_state: Mapping[str, Tensor] | None = None,
        test_only: bool = False,
    ) -> None:
        super().__init__()
        if checkpoint_state is not None and pretrained:
            raise ValueError("pass either pretrained=True or checkpoint_state, not both")
        if not pretrained and checkpoint_state is None and not test_only:
            raise ValueError(
                "pretrained=False requires checkpoint_state or the explicit test_only=True flag"
            )

        timm = _load_timm()
        try:
            backbone = timm.create_model(
                self.MODEL_NAME,
                pretrained=checkpoint_state is None and pretrained,
            )
        except Exception as exc:
            if pretrained and checkpoint_state is None:
                raise RuntimeError(
                    f"could not load pretrained RGB encoder {self.MODEL_NAME!r}; "
                    "provide a compatible offline checkpoint or fix the model cache/network"
                ) from exc
            raise

        # ResNetV2-50x1 has a 7x7 stem followed by stage 0 with three
        # pre-activation bottleneck blocks.  Keeping the stage as a module,
        # rather than recreating selected convolutions, preserves the learned
        # residual and projection shortcut parameters.
        if len(backbone.stages[0].blocks) != _FRONT_STAGE_DEPTH:
            raise RuntimeError(
                "unexpected ResNetV2 stage-0 depth: "
                f"{len(backbone.stages[0].blocks)} != {_FRONT_STAGE_DEPTH}"
            )
        self.encoder = nn.Sequential(backbone.stem, backbone.stages[0])
        self.pool = nn.AdaptiveAvgPool2d(_POOL_SIZE)
        self.register_buffer(
            "normalization_mean",
            torch.tensor(_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=True,
        )
        self.register_buffer(
            "normalization_std",
            torch.tensor(_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=True,
        )

        if checkpoint_state is not None:
            incompatible = self.load_state_dict(dict(checkpoint_state), strict=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise ValueError(
                    "RGB encoder checkpoint state does not exactly match the frozen front: "
                    f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
                )

        self._freeze()

    def _freeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "RGBFeatureEncoder":
        """Keep the frozen ImageNet front in evaluation mode permanently.

        A parent policy module may call ``train()`` recursively each update.
        BatchNorm/statistical behavior must remain the pretrained evaluation
        behavior, so this front intentionally ignores that request.
        """

        super().train(False)
        return self

    @property
    def metadata(self) -> dict[str, Any]:
        """Architecture and preprocessing contract persisted with checkpoints."""

        return {
            "checkpoint_version": self.checkpoint_version,
            "model_name": self.MODEL_NAME,
            "image_size": self.image_size,
            "feature_dim": self.feature_dim,
            "front_end": "stem+stages.0",
            "front_stage_depth": _FRONT_STAGE_DEPTH,
            "front_output_channels": _FRONT_OUTPUT_CHANNELS,
            "pool_size": list(_POOL_SIZE),
            "feature_shape": list(_FEATURE_SHAPE),
            "normalization_mean": list(_MEAN),
            "normalization_std": list(_STD),
        }

    @classmethod
    def _validate_metadata(cls, metadata: Mapping[str, Any]) -> None:
        expected = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "model_name": MODEL_NAME,
            "image_size": IMAGE_SIZE,
            "feature_dim": FEATURE_DIM,
            "front_end": "stem+stages.0",
            "front_stage_depth": _FRONT_STAGE_DEPTH,
            "front_output_channels": _FRONT_OUTPUT_CHANNELS,
            "pool_size": list(_POOL_SIZE),
            "feature_shape": list(_FEATURE_SHAPE),
            "normalization_mean": list(_MEAN),
            "normalization_std": list(_STD),
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(
                    f"incompatible RGB encoder checkpoint metadata for {key!r}: "
                    f"expected {value!r}, got {metadata.get(key)!r}"
                )

    def preprocess(self, rgb: Tensor) -> Tensor:
        """Center-crop an HWC uint8 RGB batch and resize it to 128×128.

        A single ``[H, W, 3]`` frame is accepted as a convenience and is
        promoted to a batch.  The return value is NCHW float in ``[0, 1]``;
        model normalization is applied by :meth:`encode`.
        """

        if not isinstance(rgb, Tensor):
            rgb = torch.as_tensor(rgb)
        if rgb.dtype != torch.uint8:
            raise TypeError(f"RGB input must be uint8, got {rgb.dtype}")
        if rgb.ndim == 3:
            rgb = rgb.unsqueeze(0)
        if rgb.ndim != 4 or rgb.shape[-1] != 3:
            raise ValueError(
                "RGB input must have shape [H, W, 3] or [N, H, W, 3], "
                f"got {tuple(rgb.shape)}"
            )
        height, width = int(rgb.shape[1]), int(rgb.shape[2])
        if height < 1 or width < 1:
            raise ValueError("RGB input height and width must be positive")
        side = min(height, width)
        top = (height - side) // 2
        left = (width - side) // 2
        cropped = rgb[:, top : top + side, left : left + side]
        tensor = cropped.permute(0, 3, 1, 2).to(dtype=torch.float32) / 255.0
        return F.interpolate(
            tensor,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

    @torch.inference_mode()
    def encode(self, rgb: Tensor) -> Tensor:
        """Encode uint8 RGB frame(s) into frozen 4096-D CUDA/CPU features."""

        tensor = self.preprocess(rgb)
        parameter = next(self.parameters())
        tensor = tensor.to(device=parameter.device)
        tensor = (tensor - self.normalization_mean) / self.normalization_std
        features = self.pool(self.encoder(tensor))
        return features.flatten(1)

    def forward(self, rgb: Tensor) -> Tensor:
        return self.encode(rgb)

    def checkpoint_payload(self) -> dict[str, Any]:
        """Return a CPU-safe payload for an offline strict checkpoint."""

        return {
            "metadata": self.metadata,
            "state_dict": {key: value.detach().cpu() for key, value in self.state_dict().items()},
        }

    def save_checkpoint(self, path: str | Path) -> Path:
        """Save front weights, normalization buffers, and metadata."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_payload(), destination)
        return destination

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "RGBFeatureEncoder":
        """Restore a saved front without contacting timm/Hugging Face."""

        source = Path(path)
        payload = _load_torch_checkpoint(source)
        metadata = payload.get("metadata")
        state_dict = payload.get("state_dict")
        if not isinstance(metadata, Mapping) or not isinstance(state_dict, Mapping):
            raise ValueError("RGB encoder checkpoint must contain metadata and state_dict mappings")
        cls._validate_metadata(metadata)
        return cls(pretrained=False, checkpoint_state=state_dict)


def create_rgb_feature_encoder(
    *,
    pretrained: bool = True,
    checkpoint_path: str | Path | None = None,
    test_only: bool = False,
) -> RGBFeatureEncoder:
    """Construct the frozen RGB encoder or restore its offline checkpoint."""

    if checkpoint_path is not None:
        if pretrained:
            # A path is an explicit offline source and therefore takes
            # precedence over the network-backed default.
            pretrained = False
        return RGBFeatureEncoder.from_checkpoint(checkpoint_path)
    return RGBFeatureEncoder(pretrained=pretrained, test_only=test_only)


__all__ = [
    "CHECKPOINT_VERSION",
    "FEATURE_DIM",
    "IMAGE_SIZE",
    "MODEL_NAME",
    "RGBFeatureEncoder",
    "create_rgb_feature_encoder",
]
