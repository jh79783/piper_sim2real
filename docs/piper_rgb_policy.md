# Optional simulated RGB policy input

The RGB path trains from RGB images rendered by MuJoCo. It does not require a
connected D455, `pyrealsense2`, a camera driver, or a robot SDK. The optional
vision image keeps the regular GPU simulation image and adds the pinned `timm`
dependency:

```bash
podman build -f Containerfile.vision -t localhost/piper-rsl:vision .
```

The base image must provide the matching PyTorch/CUDA stack. Fresh pretrained
construction needs access to the model cache or the network; a download or
model-load failure is reported rather than replaced by random weights.

Run a headless simulated-RGB training job from the repository root:

```bash
bash scripts/run_piper_pick_place.sh --rgb --headless \
  --num-envs 8 --no-tensorboard
```

With `--rgb`, the launcher selects `localhost/piper-rsl:vision` by default;
state-only runs continue to use `localhost/piper-rsl:gpu`. Set
`PIPER_RSL_IMAGE` to override either choice explicitly.

An existing compatible RSL-RL checkpoint can be resumed with the same `--rgb`
mode and `--resume /workspace/runs/piper_pick_place/<timestamp>/model.pt`. RGB and state-only
checkpoints have different observation contracts and are rejected when mixed.

## Encoder contract

`RGBFeatureEncoder` in [`scripts/piper_rgb_encoder.py`](../scripts/piper_rgb_encoder.py)
uses the official timm model
`resnetv2_50x1_bit.goog_in21k_ft_in1k`. Its front is the model stem followed by
`stages[0]`, which contains three preactivation bottleneck blocks. That keeps
the learned residual paths and the first block's projection shortcut intact:
there are ten main-path convolution layers (one stem plus three per block),
and the projection shortcut is an additional convolution. The implementation
does not rebuild a random ResNet or cut a residual block in half. The source
and pretrained model metadata are available in the
[official timm implementation](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/resnetv2.py)
and the [published model card](https://huggingface.co/timm/resnetv2_50x1_bit.goog_in21k_ft_in1k).

The public construction and tensor contract are:

```python
from scripts.piper_rgb_encoder import RGBFeatureEncoder, create_rgb_feature_encoder

encoder = create_rgb_feature_encoder()  # pretrained=True by default
features = encoder.encode(rgb_uint8)    # [N,H,W,3] or one [H,W,3] frame
assert tuple(features.shape[-1:]) == (4096,)
```

`rgb_uint8` is HWC RGB8 data. The encoder center-crops the shorter image side,
resizes to 128×128, converts to float in `[0, 1]`, and applies the pretrained
model normalization `(0.5, 0.5, 0.5)` for both mean and standard deviation.
The front produces 256 channels, adaptive-pools to 4×4, and flattens to 4096
features. The pooled map is retained in CHW `(256, 4, 4)` form before
flattening, so image position is still represented; there is no
global-mean-only projection.

The encoder is frozen and forced to evaluation mode, including when a parent
policy calls `.train()`. `encode` runs without autograd and returns a tensor on
the encoder's device. `pretrained=False` is accepted only with an explicit
`test_only=True` random fixture or a strict `checkpoint_state`.

## Policy integration

When `--rgb` is selected, the RGB environment wrapper renders the current
MuJoCo scene and adds the flattened 4096-D feature group to the actor input.
The renderer runs at the configured simulated camera rate (30 Hz by default)
while the control clock remains 50 Hz. Intermediate policy ticks reuse the
latest rendered feature and expose its age so the policy can learn the timing
difference. The critic keeps its clean state observation under the existing
actor/critic observation contract.

The rendered image is a simulator observation. It is not evidence that the
same viewpoint, lighting, camera intrinsics, or color response has been
matched to a physical D455. A later hardware capture path will need camera
calibration and explicit RGB synchronization before deployment; this change
does not send CAN, motor, or robot commands.

## Offline weights and validation

The encoder checkpoint stores its state dictionary, normalization buffers, and
metadata together. Restoration validates the model name, front-stage depth,
feature shape, crop size, pool size, and normalization before loading the state
strictly:

```python
encoder.save_checkpoint("artifacts/rgb_front.pt")
restored = RGBFeatureEncoder.from_checkpoint("artifacts/rgb_front.pt")
```

Use random weights only for local shape/translation tests:

```python
test_encoder = RGBFeatureEncoder(pretrained=False, test_only=True)
```

The vision tests run without a camera:

```bash
python3 -m unittest tests.test_piper_rgb -v
```

The test-only model is deliberately not a sim-to-real claim. A pretrained
download, rendered training run, and eventual physical-camera validation are
separate checks.
