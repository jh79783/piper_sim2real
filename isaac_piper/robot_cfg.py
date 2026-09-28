"""Isaac Lab articulation config for the AgileX Piper arm.

Drive gains, force limits, joint friction and armature are copied from the
Menagerie ``piper.xml`` actuators/defaults so the Isaac articulation has the
same nominal low-level controller as the MuJoCo baseline:

* ``joint1-3``: kp 80, kv 5; ``joint4``: kp 40, kv 5; ``joint5-6``: kp 10, kv 1.5
* arm force range +-100 N*m, joint frictionloss 0.3 N*m, armature 0.005
* finger drive kp 40, kv 5, force range +-10 N, no frictionloss

MuJoCo couples ``joint8 = -joint7`` with an equality constraint.  PhysX has
no joint equality, so both finger joints get the same drive and the env sends
``+g`` / ``-g`` targets.  ``gravcomp="1"`` becomes ``disable_gravity=True``.
None of these values are Piper measurements; real2sim identification must
replace them before any hardware claim.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from isaac_piper.kinematics import HOME_JOINTS
from isaac_piper.paths import PIPER_USD

ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]
FINGER_JOINTS = ["joint7", "joint8"]
FINGER_OPEN = 0.035

PIPER_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(PIPER_USD),
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
        joint_pos={
            **{name: value for name, value in zip(ARM_JOINTS, HOME_JOINTS)},
            "joint7": FINGER_OPEN,
            "joint8": -FINGER_OPEN,
        },
    ),
    actuators={
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-3]"],
            stiffness=80.0,
            damping=5.0,
            effort_limit_sim=100.0,
            friction=0.3,
            armature=0.005,
        ),
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["joint4"],
            stiffness=40.0,
            damping=5.0,
            effort_limit_sim=100.0,
            friction=0.3,
            armature=0.005,
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["joint[5-6]"],
            stiffness=10.0,
            damping=1.5,
            effort_limit_sim=100.0,
            friction=0.3,
            armature=0.005,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=FINGER_JOINTS,
            stiffness=40.0,
            damping=5.0,
            effort_limit_sim=10.0,
            friction=0.0,
            armature=0.005,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
