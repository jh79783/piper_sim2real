#!/usr/bin/env python3
"""Validate CUDA Torch + native MuJoCo-Warp on a tiny model.

This is intentionally independent of the project environments.  It checks
the lowest-level contract used by the direct backend:

    host MjModel -> mujoco_warp.put_model -> make_data -> step

Run it inside the RSL image with a GPU exposed, for example:

    python scripts/check_gpu_backend.py --worlds 4 --steps 3

MuJoCo's host model is only used to compile/upload the static model.  The
state arrays and all physics steps are allocated/launched on ``cuda:0``.
"""

from __future__ import annotations

import argparse

import mujoco
import mujoco_warp as mjw
import torch
import warp as wp


MODEL_XML = """
<mujoco model="warp_smoke">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="ball" pos="0 0 0.35">
      <freejoint/>
      <geom name="ball_geom" type="sphere" size="0.05" mass="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _device_name(array) -> str:
    """Return a stable device spelling across supported Warp versions."""

    return str(array.device)


def run(worlds: int, steps: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Torch CUDA is unavailable; expose an NVIDIA GPU to the container")
    if not wp.is_cuda_available():
        raise RuntimeError("Warp CUDA is unavailable; check the NVIDIA driver and Docker GPU support (--gpus all)")
    if worlds < 1 or steps < 1:
        raise ValueError("worlds and steps must be positive")

    torch_device = torch.device("cuda:0")
    torch_probe = torch.ones(1, device=torch_device)
    print(f"torch={torch.__version__} cuda={torch.version.cuda} device={torch.cuda.get_device_name(0)}")
    print(f"warp={wp.__version__} cuda_devices={[str(d) for d in wp.get_cuda_devices()]}")
    print(f"torch_probe_device={torch_probe.device}")

    host_model = mujoco.MjModel.from_xml_string(MODEL_XML)
    # MJWarp allocates all wp.array fields on the active Warp device.  Keeping
    # both construction and stepping in this scope avoids accidental CPU data.
    with wp.ScopedDevice("cuda:0"):
        model = mjw.put_model(host_model)
        data = mjw.make_data(host_model, nworld=worlds, nconmax=8, njmax=32)
        for name in ("qpos", "qvel", "ctrl"):
            array = getattr(data, name)
            device = _device_name(array)
            if not device.startswith("cuda"):
                raise AssertionError(f"data.{name} allocated on {device}, expected CUDA")
        qpos_before = data.qpos.numpy().copy()
        for _ in range(steps):
            mjw.step(model, data)
        wp.synchronize()
        qpos_after = data.qpos.numpy()

    if qpos_after.shape != (worlds, host_model.nq):
        raise AssertionError(f"unexpected qpos shape: {qpos_after.shape}")
    if not torch.isfinite(torch.as_tensor(qpos_after)).all():
        raise AssertionError("MuJoCo-Warp produced non-finite qpos")
    if (qpos_before == qpos_after).all():
        raise AssertionError("qpos did not change; GPU step may not have run")
    print(
        f"PASS: put_model/make_data/step on CUDA ({worlds} worlds, {steps} steps); "
        f"qpos_z={qpos_after[:, 2].tolist()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", type=int, default=4)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    run(args.worlds, args.steps)


if __name__ == "__main__":
    main()
