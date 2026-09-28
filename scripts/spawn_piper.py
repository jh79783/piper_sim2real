"""Spawn the Piper arm in Isaac Sim and sweep its joints.

    python scripts/spawn_piper.py                      # GUI on the host display
    python scripts/spawn_piper.py --headless --steps 300
    python scripts/spawn_piper.py --headless --snapshot runs/spawn_piper.png
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--steps", type=int, default=None, help="Stop after N steps (default: until the window closes).")
parser.add_argument("--snapshot", type=str, default=None, help="Save a PNG of the scene at the last step.")
parser.add_argument("--no_motion", action="store_true", help="Hold the zero pose instead of sweeping.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.snapshot:
    args.enable_cameras = True
if args.headless and args.steps is None:
    args.steps = 300

app = AppLauncher(args).app

from pathlib import Path  # noqa: E402

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402

from piper_sim2real.assets import ARM_JOINTS, FINGER_JOINTS, FINGER_OPEN, PIPER_CFG  # noqa: E402

EYE = (0.9, 0.9, 0.6)
TARGET = (0.0, 0.0, 0.25)


def design_scene():
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/ground", ground)
    light = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))
    light.func("/World/light", light)
    return Articulation(PIPER_CFG.replace(prim_path="/World/Piper"))


def make_camera():
    from isaaclab.sensors import Camera, CameraCfg

    camera = Camera(
        CameraCfg(
            prim_path="/World/snapshot_cam",
            height=720,
            width=1280,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.01, 20.0)),
        )
    )
    return camera


def describe(robot: Articulation):
    limits = robot.data.soft_joint_pos_limits[0].cpu()
    print(f"[piper] bodies: {robot.body_names}")
    for i, name in enumerate(robot.joint_names):
        print(f"[piper] {name}: limits [{limits[i, 0]:+.3f}, {limits[i, 1]:+.3f}]")


def sweep_targets(robot: Articulation, arm_ids, finger_ids, t: float) -> torch.Tensor:
    """Oscillate each arm joint over the middle of its range, open/close the gripper."""
    target = robot.data.default_joint_pos.clone()
    limits = robot.data.soft_joint_pos_limits[0]
    for k, j in enumerate(arm_ids):
        lower, upper = limits[j]
        mid = 0.5 * (lower + upper)
        # Joints whose zero sits at a limit (joint2, joint3) sweep from zero
        # into their range instead of around the midpoint.
        span = 0.25 * (upper - lower)
        phase = math.sin(2.0 * math.pi * 0.25 * t + k)
        if lower >= 0.0:
            target[:, j] = lower + (upper - lower) * 0.2 * (1.0 + phase)
        elif upper <= 0.0:
            target[:, j] = upper - (upper - lower) * 0.2 * (1.0 + phase)
        else:
            target[:, j] = mid + span * phase
    grip = 0.5 * FINGER_OPEN * (1.0 + math.sin(2.0 * math.pi * 0.5 * t))
    target[:, finger_ids[0]] = grip
    target[:, finger_ids[1]] = -grip
    return target


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args.device))
    sim.set_camera_view(EYE, TARGET)
    robot = design_scene()
    camera = make_camera() if args.snapshot else None
    sim.reset()
    if camera is not None:
        camera.set_world_poses_from_view(
            torch.tensor([EYE], device=sim.device), torch.tensor([TARGET], device=sim.device)
        )

    describe(robot)
    arm_ids, _ = robot.find_joints(ARM_JOINTS, preserve_order=True)
    finger_ids, _ = robot.find_joints(FINGER_JOINTS, preserve_order=True)

    dt = sim.get_physics_dt()
    step = 0
    while app.is_running() and (args.steps is None or step < args.steps):
        if not args.no_motion:
            robot.set_joint_position_target(sweep_targets(robot, arm_ids, finger_ids, step * dt))
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)
        if camera is not None:
            camera.update(dt)
        step += 1

    q = robot.data.joint_pos[0].cpu().numpy().round(3)
    ee = robot.data.body_pos_w[0, robot.find_bodies("gripper_base")[0][0]].cpu().numpy().round(3)
    print(f"[piper] stepped {step} steps, joint_pos={q.tolist()}, gripper_base_pos={ee.tolist()}")

    if camera is not None:
        from PIL import Image

        out = Path(args.snapshot)
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(camera.data.output["rgb"][0, ..., :3].cpu().numpy()).save(out)
        print(f"[piper] snapshot: {out}")

    # SimulationApp.close() otherwise blocks in Isaac Lab's timeline-stop callback.
    sim._disable_app_control_on_stop_handle = True


if __name__ == "__main__":
    main()
    app.close()
