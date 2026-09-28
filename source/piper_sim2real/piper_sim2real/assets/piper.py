"""Isaac Lab articulation config for the AgileX Piper arm with its parallel gripper.

The asset is the official AgileX URDF (``assets/piper_description``), imported
at spawn time by the Isaac Sim URDF importer.  Joint limits and link inertias
come from that URDF.

Drive gains below are placeholders, not Piper measurements.  The real arm runs
its own gravity-compensated position loop, so gravity is disabled on the links
and the PD drives only have to track targets.  Real2sim identification must
replace these values before any hardware claim.

The two fingers are independent prismatic joints (``joint7`` in [0, 0.035],
``joint8`` in [-0.035, 0]); PhysX has no mimic constraint, so callers command
``joint7 = g`` and ``joint8 = -g``.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.converters import UrdfConverterCfg

from piper_sim2real.paths import PIPER_URDF, PIPER_USD_DIR

ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]
FINGER_JOINTS = ["joint7", "joint8"]
FINGER_OPEN = 0.035
EE_BODY = "gripper_base"

PIPER_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(PIPER_URDF),
        usd_dir=str(PIPER_USD_DIR),
        fix_base=True,
        # Keep gripper_base as a separate body so link names match
        # the upstream URDF used by the real-robot stack.
        merge_fixed_joints=False,
        self_collision=False,
        collider_type="convex_hull",
        # Actuator configs below overwrite the drive gains on spawn.
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={".*": 0.0},
    ),
    actuators={
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-3]"],
            stiffness=80.0,
            damping=5.0,
            effort_limit_sim=100.0,
            velocity_limit_sim=5.0,
            armature=0.005,
        ),
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["joint4"],
            stiffness=40.0,
            damping=5.0,
            effort_limit_sim=100.0,
            velocity_limit_sim=5.0,
            armature=0.005,
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["joint[5-6]"],
            stiffness=10.0,
            damping=1.5,
            effort_limit_sim=100.0,
            velocity_limit_sim=3.0,
            armature=0.005,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=FINGER_JOINTS,
            stiffness=400.0,
            damping=20.0,
            effort_limit_sim=10.0,
            velocity_limit_sim=1.0,
        ),
    },
)
