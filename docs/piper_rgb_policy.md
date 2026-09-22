# Optional simulated RGB policy input

The RGB path trains from RGB images rendered by MuJoCo. It does not require a
connected D455, `pyrealsense2`, a camera driver, or a robot SDK. The policy
camera is a rigid child of Piper's `link6`, so its view moves with the wrist;
the state-only path and the free external preview remain unchanged. The
optional vision image keeps the regular GPU simulation image and adds the
pinned `timm` dependency:

```bash
docker build -f Dockerfile.vision -t piper-rsl:vision .
```

Build `piper-rl:gpu` and then `piper-rsl:gpu` as described in the
[README](../README.md) before building the vision image. The base image
provides the matching PyTorch/CUDA stack. Fresh pretrained
construction needs access to the model cache or the network; a download or
model-load failure is reported rather than replaced by random weights.

Run a headless simulated-RGB training job from the repository root:

```bash
bash scripts/run_piper_pick_place.sh --rgb --headless \
  --num-envs 8 --no-tensorboard
```

With `--rgb`, the launcher selects `piper-rsl:vision` by default;
state-only runs continue to use `piper-rsl:gpu`. Set
`PIPER_RSL_IMAGE` to override either choice explicitly.

The Docker launcher exposes the NVIDIA graphics driver capabilities, and the
vision image registers the NVIDIA EGL vendor for headless GPU rendering.

Headless RGB runs use the vision image's EGL backend. For a GUI run, the
launcher uses the host `DISPLAY` value and mounts the matching X11 socket plus
a temporary read-only authority file containing only that display's cookie;
it does not call `xhost`. This supports native displays such as `:10` as well
as WSLg `:0`. MuJoCo selects one OpenGL backend per Python process, so GUI
runs keep the RGB policy renderer on EGL and put the visible GLFW preview in a
small child process: `PIPER_RENDERER=gpu` uses the native/WSLg accelerated
GLFW path when available, while `PIPER_RENDERER=software` selects software
GLFW only for that preview child. State-only GUI runs retain a single GLFW
process. Use `--headless` when no preview display is needed.

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
MuJoCo scene from the wrist-mounted Intel RealSense D455 approximation and adds
the flattened 4096-D feature group to the actor input. Raw policy frames are
`640×480×3` at the configured simulated camera rate (30 Hz by default). The
encoder center-crops the shorter side and resizes the result to its existing
`128×128` input. The control clock remains 50 Hz; intermediate policy ticks
reuse the latest rendered feature and expose its age so the policy can learn
the timing difference. The RGB actor's numeric state is 55-D: it retains 54 of
the 63 base policy fields plus the RGB frame age. It excludes only absolute cube
XYZ (`20:23`), cube-minus-TCP (`33:36`), and goal-minus-cube (`39:42`); cube
orientation/velocity, contact, lift, placement, time, commands, and previous
action fields remain. Those three position groups are the direct XYZ paths;
retained fields may still provide indirect object cues by design. The actor input is therefore 55 + 4096 = 4151-D,
while the critic remains the full clean 63-D observation. The observation
contract is `rgb_resnetv2_wrist_d455_object_xyz_hidden_v2`; older RGB PPO
checkpoints are rejected and require a fresh actor, while their standalone
frozen encoder weights remain reusable.

## Performance notes

The RGB renderer transfers the eight visual pose arrays needed by this Piper
scene (`geom`, `site`, `camera`, and `light` poses) once per capture, gathers
the requested worlds, and fills a reusable host `MjData`. It does not call the
generic full-state `get_data_into` path for policy frames. Raw 640×480 RGB8
frames still arrive at 30 Hz, while crop, resize, and normalization now run on
the encoder device in bounded 64-frame chunks. The frozen ResNet front, its
128×128 input, 4096-D feature shape, normalization, and checkpoint metadata
contract are unchanged.

On an RTX 5090 with 512 environments and 64 rollout ticks, collection fell
from 265.7–277.2 seconds in the previous run to 11.3 seconds (2,866
transitions/second); the PPO update took 0.122 seconds. The run's cumulative
`render_timing` counters are persisted in `summary.json`: transfer and render
times cover all captures, while `encode_host_dispatch_seconds` measures host
dispatch time around the frozen encoder call and does not claim to isolate GPU
kernel time. The launcher formats elapsed time and ETA with days once a run
exceeds 24 hours, so long-run estimates do not wrap at midnight.

The camera body dimensions and nominal RGB frame offsets come from the pinned
[official RealSense D455 xacro](https://github.com/realsenseai/realsense-ros/blob/9a11121700cb4780e273e34141f6402fe184321d/realsense2_description/urdf/_d455.urdf.xacro#L29-L159).
The provisional wrist bracket pose follows AgileX's D435 camera reference
([source](https://github.com/agilexrobotics/piper_isaac_sim/blob/8e1f88fdb7afca49c40e9a0c1c01cc588e86f0d2/piper_description/urdf/piper_description_v100_realsense_camera.urdf#L367-L373));
it is documented as an uncalibrated D455 mounting assumption and does not claim
that the D435 bracket physically fits the longer D455. The ROS optical-frame
axes are converted to MuJoCo's camera axes explicitly. The 65° vertical FOV is
used for a 4:3 simulation pinhole at 640×480. RealSense's native RGB profile is
1280×800 and its 640×480 profile is an ISP central crop, so this simulation
does not claim calibrated hardware intrinsics; see the
[official librealsense crop path](https://github.com/realsenseai/librealsense/blob/e15c5d6bb1563e778d116f682aeefffbae2daedc/src/ds/d400/d400-private.cpp#L304-L376).

Camera geometry, frame conversion, source commits, FOV assumptions, raw render
size, and encoder size are persisted in the RGB `vision_config` checkpoint
metadata. The observation version is wrist-D455-specific, so an older
external-view RGB checkpoint is rejected during resume rather than silently
reused.

The rendered image is a simulator observation. It is not evidence that the
same viewpoint, lighting, calibrated camera intrinsics, or color response has
been matched to a physical D455. A later hardware capture path will need a
D455-specific bracket survey, camera calibration, and explicit RGB
synchronization before deployment; this change does not send CAN, motor, or
robot commands.

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
