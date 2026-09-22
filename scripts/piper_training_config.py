"""Serializable training and sim-to-real randomization defaults for Piper.

The values in this module describe the simulation contract.  In particular,
``training_steps`` means calls to the vector environment (one 50 Hz tick),
independent of the number of worlds, PPO batches, or optimizer updates.  The
launcher owns checkpoint restore and calls the small ``set_training_steps``
API exposed by :mod:`piper_warp_env`.
"""

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class CurriculumStage:
    step: int
    randomization_scale: float
    layout_scale: float
    home_probability: float
    action_rate_weight: float


@dataclass(frozen=True)
class PiperTrainingConfig:
    domain_randomization: bool = True
    sensor_noise: bool = True
    curriculum: bool = True
    # MuJoCo advances at 2 ms.  The policy emits one incremental joint-target
    # command every 20 ms (50 Hz), hence ten physics substeps per transition.
    physics_timestep: float = 0.002
    control_hz: int = 50
    decimation: int = 10
    episode_seconds: float = 12.0
    success_seconds: float = 0.32
    # Incremental target action contract. Six arm values are normalized target
    # increments and the seventh is a normalized gripper-target increment.
    num_actions: int = 7
    joint_target_rate: float = 0.035
    gripper_target_rate: float = 0.004
    time_penalty: float = 0.005
    # Per-episode model uncertainty at the final curriculum stage.
    link_mass_fraction: float = 0.10
    cube_mass_fraction: float = 0.25
    inertia_fraction: float = 0.10
    link_com_range: float = 0.002
    cube_com_range: float = 0.002
    friction_fraction: float = 0.25
    # Uniform measurement error bounds, before statistical normalization.
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
    # Count vector-env steps, independent of world count or PPO batch size.
    stages: tuple[CurriculumStage, ...] = field(default_factory=lambda: (
        CurriculumStage(0, 0.2, 0.55, 0.0, 0.001),
        CurriculumStage(32_000, 0.5, 0.75, 0.25, 0.002),
        CurriculumStage(64_000, 0.75, 0.9, 0.5, 0.005),
        CurriculumStage(96_000, 1.0, 1.0, 0.8, 0.01),
    ))

    def __post_init__(self):
        for name in (
            "link_mass_fraction",
            "cube_mass_fraction",
            "inertia_fraction",
            "friction_fraction",
        ):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        scalar_fields = asdict(self)
        for name, value in scalar_fields.items():
            if isinstance(value, (int, float)) and (
                not math.isfinite(value) or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.control_hz != 50:
            raise ValueError("control_hz must be 50 for the Piper 20 ms contract")
        if self.decimation != 10:
            raise ValueError("decimation must be 10 for 2 ms physics")
        if not math.isclose(self.physics_timestep * self.decimation, 1.0 / self.control_hz):
            raise ValueError("physics_timestep * decimation must equal one control period")
        if self.num_actions != 7:
            raise ValueError("num_actions must be 7 (six arm targets plus gripper target)")
        if not isinstance(self.max_sensor_delay_steps, int):
            raise ValueError("max_sensor_delay_steps must be an integer")
        if not 0 <= self.max_sensor_delay_steps <= 2:
            raise ValueError("max_sensor_delay_steps must be between zero and two ticks")
        if not self.stages or self.stages[0].step != 0:
            raise ValueError("curriculum must start at step zero")
        previous = -1
        for stage in self.stages:
            if not isinstance(stage.step, int) or stage.step <= previous:
                raise ValueError("curriculum steps must increase strictly")
            previous = stage.step
            for value in (stage.randomization_scale, stage.layout_scale, stage.home_probability):
                if not 0 <= value <= 1:
                    raise ValueError("curriculum scales/probabilities must be in [0, 1]")
            if not math.isfinite(stage.action_rate_weight) or stage.action_rate_weight < 0:
                raise ValueError("action-rate weights must be finite and nonnegative")

    def stage_at(self, steps: int) -> CurriculumStage:
        if int(steps) < 0:
            raise ValueError("training steps must be nonnegative")
        if not self.curriculum:
            return self.stages[-1]
        return next(stage for stage in reversed(self.stages) if steps >= stage.step)

    def to_dict(self):
        result = asdict(self)
        # ``json.dumps`` handles tuples, but a list makes the persisted schema
        # stable across Python/dataclass versions and easier to inspect.
        result["stages"] = [asdict(stage) for stage in self.stages]
        return result

    @classmethod
    def from_dict(cls, values: Mapping[str, Any] | None):
        """Rebuild a config from a checkpoint/config JSON mapping."""

        if values is None:
            return cls()
        values = dict(values)
        raw_stages = values.get("stages")
        if raw_stages is not None:
            values["stages"] = tuple(
                item if isinstance(item, CurriculumStage) else CurriculumStage(**item)
                for item in raw_stages
            )
        fields = set(cls.__dataclass_fields__)
        unknown = sorted(set(values) - fields)
        if unknown:
            raise ValueError(f"unknown Piper training config fields: {', '.join(unknown)}")
        return cls(**values)

    @property
    def episode_steps(self) -> int:
        return int(round(self.episode_seconds * self.control_hz))

    @property
    def success_steps(self) -> int:
        return int(round(self.success_seconds * self.control_hz))
