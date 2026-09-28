"""Configuration of the Isaac Sim Piper pick-and-place task (state policy).

The policy-facing contract is the one established by the earlier MuJoCo-Warp
task (see ``README.md``): 50 Hz control, 600-tick (12 s)
episodes, a 7-D normalized incremental joint/gripper target action, a 63-D
observation with a noisy/delayed actor view and a clean critic view, reward-v3
and the 16-tick stable-placement success gate.  Only the simulator behind it
changed.
"""

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from isaac_piper.robot_cfg import PIPER_CFG

CUBE_HALF_SIZE = 0.02
GOAL_HALF_SIZE = (0.06, 0.05)
# Fitted collision offsets: the 2.5 mm finger pads and the 40 mm cube need a
# contact offset well below PhysX's default (2 cm) to avoid phantom contacts.
CONTACT_OFFSET = 0.004


@configclass
class CurriculumStageCfg:
    step: int = MISSING
    randomization_scale: float = MISSING
    layout_scale: float = MISSING
    home_probability: float = MISSING
    action_rate_weight: float = MISSING


@configclass
class RandomizationCfg:
    """Final-stage ranges; each curriculum stage multiplies them by its scale."""

    domain_randomization: bool = True
    sensor_noise: bool = True
    # Per-episode model uncertainty (uniform, relative unless noted).
    link_mass_fraction: float = 0.10
    cube_mass_fraction: float = 0.25
    inertia_fraction: float = 0.10
    link_com_range: float = 0.002  # m, per axis
    cube_com_range: float = 0.002  # m, per axis
    friction_fraction: float = 0.25
    # Friction samples are quantized so PhysX sees a bounded material set.
    friction_buckets: int = 64
    # Uniform measurement error half-ranges (raw SI, before normalization).
    joint_position_noise: float = 0.001
    joint_velocity_noise: float = 0.25
    joint_zero_offset: float = 0.005
    finger_position_noise: float = 0.0002
    finger_velocity_noise: float = 0.002
    position_noise: float = 0.002
    orientation_noise: float = 0.01
    linear_velocity_noise: float = 0.02
    angular_velocity_noise: float = 0.05
    max_sensor_delay_steps: int = 2


CONTACT_LINKS = tuple(f"link{i}" for i in range(1, 9))


def _cube_contact(link: str) -> ContactSensorCfg:
    return ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/base_link/{link}",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
        update_period=0.0,
        history_length=0,
    )


@configclass
class PiperPickPlaceSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = PIPER_CFG.replace(
        spawn=PIPER_CFG.spawn.replace(
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=CONTACT_OFFSET, rest_offset=0.0)
        )
    )
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(2 * CUBE_HALF_SIZE,) * 3,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.04),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=CONTACT_OFFSET, rest_offset=0.0),
            # MuJoCo combines geom friction with max(); keep that rule here.
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.3, dynamic_friction=1.3, friction_combine_mode="max"
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.35, 0.08)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.32, -0.10, CUBE_HALF_SIZE + 0.0005)),
    )
    table: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.80, 0.72, 0.06),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=CONTACT_OFFSET, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, friction_combine_mode="max"
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.20, 0.25, 0.30)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.25, 0.0, -0.03)),
    )
    # A collider box instead of GroundPlaneCfg keeps the scene free of
    # Nucleus downloads, so training works offline.
    floor: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Floor",
        spawn=sim_utils.CuboidCfg(
            size=(2.0, 2.0, 0.04),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.14, 0.16)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.25, 0.0, -0.08)),
    )
    # Every moving robot link reports normal force against the cube only.
    # PhysX filtered contacts are one-to-many, hence one sensor per link.
    contact_link1: ContactSensorCfg = _cube_contact("link1")
    contact_link2: ContactSensorCfg = _cube_contact("link2")
    contact_link3: ContactSensorCfg = _cube_contact("link3")
    contact_link4: ContactSensorCfg = _cube_contact("link4")
    contact_link5: ContactSensorCfg = _cube_contact("link5")
    contact_link6: ContactSensorCfg = _cube_contact("link6")
    contact_link7: ContactSensorCfg = _cube_contact("link7")
    contact_link8: ContactSensorCfg = _cube_contact("link8")
    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
    )


@configclass
class PiperPickPlaceEnvCfg(DirectRLEnvCfg):
    # 5 ms PhysX steps x 4 = one 20 ms (50 Hz) policy tick.  The MuJoCo path
    # used 2 ms x 10; the policy period, not the solver step, is the contract.
    decimation: int = 4
    episode_length_s: float = 12.0
    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=4,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, friction_combine_mode="max"
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_found_lost_aggregate_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**22,
        ),
    )
    scene: PiperPickPlaceSceneCfg = PiperPickPlaceSceneCfg(num_envs=4096, env_spacing=1.5, replicate_physics=True)

    action_space: int = 7
    observation_space: int = 63
    state_space: int = 0

    # Action contract (normalized increments, clamped to [-1, 1]).
    joint_target_rate: float = 0.035  # rad per tick at |a| = 1
    gripper_target_rate: float = 0.004  # m per tick at |a| = 1
    joint_limit_margin: float = 0.002

    # Task and reward-v3.
    success_seconds: float = 0.32
    shaping_gamma: float = 0.99**0.5
    time_penalty: float = 0.005
    success_bonus: float = 20.0
    failure_penalty: float = 5.0
    recontact_penalty: float = 0.2
    contact_force_threshold: float = 0.01  # N, normal force for "in contact"
    start_mode: str = "curriculum"  # curriculum | above_cube | home
    curriculum: bool = True
    stages: list[CurriculumStageCfg] = [
        CurriculumStageCfg(step=0, randomization_scale=0.2, layout_scale=0.55, home_probability=0.0,
                           action_rate_weight=0.001),
        CurriculumStageCfg(step=32_000, randomization_scale=0.5, layout_scale=0.75, home_probability=0.25,
                           action_rate_weight=0.002),
        CurriculumStageCfg(step=64_000, randomization_scale=0.75, layout_scale=0.9, home_probability=0.5,
                           action_rate_weight=0.005),
        CurriculumStageCfg(step=96_000, randomization_scale=1.0, layout_scale=1.0, home_probability=0.8,
                           action_rate_weight=0.01),
    ]
    randomization: RandomizationCfg = RandomizationCfg()

    goal_marker: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/PiperGoal",
        markers={
            "goal": sim_utils.CuboidCfg(
                size=(2 * GOAL_HALF_SIZE[0], 2 * GOAL_HALF_SIZE[1], 0.002),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.9, 0.3), opacity=0.35),
            )
        },
    )

    def __post_init__(self):
        if abs(self.sim.dt * self.decimation - 0.02) > 1e-9:
            raise ValueError("sim.dt * decimation must equal the 20 ms (50 Hz) policy period")
        if self.start_mode not in ("curriculum", "above_cube", "home"):
            raise ValueError("start_mode must be curriculum, above_cube or home")
        if not self.stages or self.stages[0].step != 0:
            raise ValueError("curriculum must start at step zero")
        if any(b.step <= a.step for a, b in zip(self.stages, self.stages[1:])):
            raise ValueError("curriculum steps must increase strictly")
        if not 0 <= self.randomization.max_sensor_delay_steps <= 2:
            raise ValueError("max_sensor_delay_steps must be between zero and two ticks")
