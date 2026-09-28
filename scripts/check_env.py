"""Headless smoke test: boot Isaac Sim through Isaac Lab and step a scene on the GPU."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=200)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args = parser.parse_args()

app = AppLauncher(args).app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args.device))
ground = sim_utils.GroundPlaneCfg()
ground.func("/World/ground", ground)
cube = sim_utils.CuboidCfg(
    size=(0.1, 0.1, 0.1),
    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
    collision_props=sim_utils.CollisionPropertiesCfg(),
)
cube.func("/World/cube", cube, translation=(0.0, 0.0, 1.0))
sim.reset()
for _ in range(args.steps):
    sim.step()

print(f"[check_env] torch {torch.__version__}, cuda={torch.cuda.is_available()}, "
      f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}")
print(f"[check_env] sim device={sim.device}, stepped {args.steps} steps OK")
# Without this, SimulationApp.close() fires Isaac Lab's timeline-stop callback,
# which keeps rendering until the timeline plays again and never returns.
sim._disable_app_control_on_stop_handle = True
app.close()
