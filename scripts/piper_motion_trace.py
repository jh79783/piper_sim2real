"""Offline Piper command/feedback trace recording and MuJoCo replay.

The file format is deliberately independent of ``piper_sdk``.  A verified
controller can pass already-converted SI values to :class:`TraceWriter` while
the SDK remains an optional, read-only integration outside this module.

Trace values use six absolute joint targets/feedback values in radians and a
full gripper opening in metres.  The training policy's normalized incremental
commands are not written here; an adapter must convert them to the absolute
target sent to the controller before recording.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import numbers
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence


TRACE_SCHEMA = "piper_motion_trace"
TRACE_SCHEMA_VERSION = 1
JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
JOINT_COUNT = len(JOINT_NAMES)
NANOSECONDS_PER_SECOND = 1_000_000_000
DEFAULT_MAX_SOURCE_AGE_S = 0.100
DEFAULT_TIME_TOLERANCE_S = 1e-8
DEFAULT_MAX_STEPS_PER_INTERVAL = 100_000
MODEL_RANGE_TOLERANCE_RAD = 1e-7
MODEL_RANGE_TOLERANCE_M = 1e-9
MODEL_FEEDBACK_RANGE_TOLERANCE_RAD = 1e-6
MODEL_FEEDBACK_RANGE_TOLERANCE_M = 1e-8


def _positive_float_arg(value: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a finite positive number") from exc
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return result


def _positive_int_arg(value: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


class MotionTraceError(ValueError):
    """A malformed, unsafe, or un-replayable motion trace."""

    def __init__(self, message: str, *, issues: Sequence[str] = ()):
        self.issues = tuple(str(issue) for issue in issues)
        detail = message
        if self.issues:
            detail += ": " + "; ".join(self.issues[:8])
            if len(self.issues) > 8:
                detail += f"; ... ({len(self.issues)} issues)"
        super().__init__(detail)


class MotionTraceRangeError(MotionTraceError):
    """A command or feedback value is outside the selected MuJoCo model."""


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise MotionTraceError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MotionTraceError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise MotionTraceError(f"{name} must be finite")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise MotionTraceError(f"{name} must be a non-negative integer")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise MotionTraceError(f"{name} must be a non-negative integer") from exc
    if not math.isfinite(float(value)) or float(value) != integer:
        raise MotionTraceError(f"{name} must be an integer")
    if integer < 0:
        raise MotionTraceError(f"{name} must be non-negative")
    return integer


def millidegree_to_rad(value: Any) -> float:
    """Convert SDK-style 0.001-degree units to radians."""

    return _finite_float(value, "millidegree") * (math.pi / 180_000.0)


def rad_to_millidegree(value: Any) -> float:
    """Convert radians to SDK-style 0.001-degree units."""

    return _finite_float(value, "radian") * (180_000.0 / math.pi)


def micrometer_full_opening_to_model_half_travel(value: Any) -> float:
    """Convert full gripper opening in micrometres to model half travel metres."""

    return _finite_float(value, "full gripper opening in micrometres") * 1e-6 / 2.0


def model_half_travel_to_micrometer_full_opening(value: Any) -> float:
    """Convert model gripper half travel metres to full opening micrometres."""

    return _finite_float(value, "model gripper half travel in metres") * 2.0 * 1e6


def full_opening_m_to_model_half_travel(value: Any) -> float:
    """Convert SI full gripper opening metres to model half travel metres."""

    return _finite_float(value, "full gripper opening in metres") / 2.0


def model_half_travel_to_full_opening_m(value: Any) -> float:
    """Convert model gripper half travel metres to SI full opening metres."""

    return _finite_float(value, "model gripper half travel in metres") * 2.0


def _finite_joint_vector(value: Any) -> tuple[float, ...] | None:
    if value is None:
        return None
    try:
        values = tuple(_finite_float(item, "joint position") for item in value)
    except TypeError as exc:
        raise MotionTraceError("joint position must be a six-element sequence") from exc
    if len(values) != JOINT_COUNT:
        raise MotionTraceError(f"joint position must contain {JOINT_COUNT} values")
    return values


def _optional_finite(value: Any, name: str) -> float | None:
    if value is None:
        return None
    return _finite_float(value, name)


def _json_loads_strict(line: str) -> Any:
    def reject_constant(value: str):
        raise MotionTraceError(f"JSON contains non-finite constant {value}")

    try:
        return json.loads(line, parse_constant=reject_constant)
    except json.JSONDecodeError as exc:
        raise MotionTraceError(f"invalid JSON at line: {exc.lineno}:{exc.colno}") from exc


def _json_dump_line(payload: Mapping[str, Any]) -> str:
    try:
        return json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
    except (TypeError, ValueError) as exc:
        raise MotionTraceError(f"trace record is not JSON-safe: {exc}") from exc


@dataclass(frozen=True)
class MotionComponent:
    joint_position_rad: tuple[float, ...] | None
    gripper_full_opening_m: float | None
    source_timestamp_ns: int | None
    valid: bool
    gripper_valid: bool
    source: str | None
    validity_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class MotionSample:
    seq: int
    t_monotonic_ns: int
    command: MotionComponent
    feedback: MotionComponent
    record_valid: bool = True
    issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class MotionTrace:
    header: Mapping[str, Any]
    samples: tuple[MotionSample, ...]
    issues: tuple[str, ...] = ()

    @property
    def errors(self) -> tuple[str, ...]:
        """Return issues that make replay unsafe."""

        errors = list(self.issues)
        for sample in self.samples:
            errors.extend(sample.issues)
            if not sample.command.valid:
                errors.append(f"seq {sample.seq}: invalid command")
        return tuple(errors)

    @property
    def invalid_sample_count(self) -> int:
        return sum(not sample.record_valid for sample in self.samples)


def _component_payload(
    *,
    joint_position_rad: Sequence[Any] | None,
    gripper_full_opening_m: Any,
    source_timestamp_ns: Any,
    valid_requested: bool,
    gripper_valid_requested: bool,
    source: str | None,
    require_source_timestamp: bool,
    validity_reasons: Sequence[str] = (),
) -> dict[str, Any]:
    reasons: list[str] = [str(reason) for reason in validity_reasons]
    try:
        joints = _finite_joint_vector(joint_position_rad)
    except MotionTraceError as exc:
        joints = None
        reasons.append(str(exc))
    try:
        gripper = _optional_finite(gripper_full_opening_m, "full gripper opening in metres")
    except MotionTraceError as exc:
        gripper = None
        reasons.append(str(exc))
    timestamp = None
    if source_timestamp_ns is not None:
        try:
            timestamp = _nonnegative_int(source_timestamp_ns, "source_timestamp_ns")
        except MotionTraceError as exc:
            reasons.append(str(exc))
    elif require_source_timestamp:
        reasons.append("missing source_timestamp_ns")
    if not valid_requested:
        reasons.append("declared invalid")
    if joints is None:
        reasons.append("missing or invalid six-joint value")
    if gripper is not None and not gripper_valid_requested:
        reasons.append("gripper value declared invalid")
    joint_valid = bool(valid_requested and joints is not None and timestamp is not None)
    gripper_valid = bool(
        gripper_valid_requested and gripper is not None and timestamp is not None
    )
    return {
        "joint_position_rad": list(joints) if joints is not None else None,
        "gripper_full_opening_m": gripper,
        "source_timestamp_ns": timestamp,
        "valid": joint_valid,
        "gripper_valid": gripper_valid,
        "source": source,
        "validity_reasons": sorted(set(reasons)),
    }


class TraceWriter:
    """Write a strict JSONL trace without importing the SDK or simulator.

    ``record`` is passive: it receives values from an already verified
    controller and never calls a motor, CAN, or SDK command API.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        metadata: Mapping[str, Any] | None = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        overwrite: bool = False,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and not overwrite:
            raise FileExistsError(f"trace already exists: {self.path}")
        self._file = self.path.open("w", encoding="utf-8")
        self._clock_ns = clock_ns
        self._seq = 0
        self._last_t: int | None = None
        values = dict(metadata or {})
        self.header = {
            "record_type": "header",
            "schema": TRACE_SCHEMA,
            "schema_version": TRACE_SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "joint_names": list(JOINT_NAMES),
            "units": {
                "time": "monotonic_ns",
                "joint_position": "rad",
                "gripper": "m_full_opening",
            },
            "command_semantics": "absolute_joint_target_si_v1",
            "capture_clock": "monotonic_ns",
            "source_clock": values.pop("source_clock", "monotonic_ns"),
            "source_timestamps_comparable": bool(values.pop("source_timestamps_comparable", True)),
            "max_source_age_s": float(values.pop("max_source_age_s", DEFAULT_MAX_SOURCE_AGE_S)),
            "initial_state_assumption": values.pop(
                "initial_state_assumption",
                "first_valid_feedback_qpos_once_then_zero_joint_velocity",
            ),
            "controller": values.get("controller", {}),
            "firmware": values.get("firmware", {}),
            "metadata": values,
        }
        if self.header["max_source_age_s"] <= 0 or not math.isfinite(self.header["max_source_age_s"]):
            self.close()
            raise MotionTraceError("max_source_age_s must be finite and positive")
        self._write(self.header)

    def _write(self, payload: Mapping[str, Any]) -> None:
        self._file.write(_json_dump_line(payload))
        self._file.flush()

    @property
    def samples_written(self) -> int:
        return self._seq

    def record(
        self,
        *,
        command_joint_position_rad: Sequence[Any] | None,
        command_gripper_full_opening_m: Any = None,
        feedback_joint_position_rad: Sequence[Any] | None = None,
        feedback_gripper_full_opening_m: Any = None,
        t_monotonic_ns: Any = None,
        command_source_timestamp_ns: Any = None,
        feedback_source_timestamp_ns: Any = None,
        command_valid: bool = True,
        feedback_valid: bool = True,
        command_gripper_valid: bool | None = None,
        feedback_gripper_valid: bool | None = None,
        command_source: str | None = None,
        feedback_source: str | None = None,
        command_validity_reasons: Sequence[str] = (),
        feedback_validity_reasons: Sequence[str] = (),
    ) -> MotionSample:
        """Append one sample; no command is sent by this method."""

        timestamp = self._clock_ns() if t_monotonic_ns is None else t_monotonic_ns
        timestamp = _nonnegative_int(timestamp, "t_monotonic_ns")
        if self._last_t is not None and timestamp <= self._last_t:
            raise MotionTraceError(
                f"out-of-order capture timestamp {timestamp} after {self._last_t}"
            )
        if command_source_timestamp_ns is None:
            command_source_timestamp_ns = timestamp
        if command_gripper_valid is None:
            command_gripper_valid = command_gripper_full_opening_m is not None
        if feedback_gripper_valid is None:
            feedback_gripper_valid = feedback_gripper_full_opening_m is not None
        command = _component_payload(
            joint_position_rad=command_joint_position_rad,
            gripper_full_opening_m=command_gripper_full_opening_m,
            source_timestamp_ns=command_source_timestamp_ns,
            valid_requested=bool(command_valid),
            gripper_valid_requested=bool(command_gripper_valid),
            source=command_source,
            require_source_timestamp=True,
            validity_reasons=command_validity_reasons,
        )
        if feedback_joint_position_rad is None and feedback_gripper_full_opening_m is None:
            # A command event can intentionally have no simultaneous feedback.
            # Keep that absence explicit as JSON null rather than turning an
            # ordinary command-only event into a malformed feedback record.
            feedback = None
        else:
            feedback = _component_payload(
                joint_position_rad=feedback_joint_position_rad,
                gripper_full_opening_m=feedback_gripper_full_opening_m,
                source_timestamp_ns=feedback_source_timestamp_ns,
                valid_requested=bool(feedback_valid),
                gripper_valid_requested=bool(feedback_gripper_valid),
                source=feedback_source,
                require_source_timestamp=True,
                validity_reasons=feedback_validity_reasons,
            )
        sample = {
            "record_type": "sample",
            "seq": self._seq,
            "t_monotonic_ns": timestamp,
            "command": command,
            "feedback": feedback,
        }
        self._write(sample)
        result = _sample_from_mapping(
            sample,
            previous=None,
            max_source_age_s=float(self.header["max_source_age_s"]),
            source_timestamps_comparable=bool(self.header["source_timestamps_comparable"]),
        )
        self._seq += 1
        self._last_t = timestamp
        return result

    def close(self) -> None:
        if getattr(self, "_file", None) is not None and not self._file.closed:
            self._file.flush()
            self._file.close()

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class Ros2JointStateRecorder:
    """Passive ROS 2 ``JointState`` adapter for the verified AgxArm stack.

    The adapter subscribes only to command and feedback topics. It never
    publishes, enables the arm, calls ``pyAgxArm``, or invokes a motion API.
    Construct the writer with ``source_timestamps_comparable=False`` when ROS
    header stamps are in a clock domain that cannot be compared to the local
    monotonic capture clock. Replay then uses capture time for ZOH while the
    report explicitly marks source-time alignment as unknown.
    """

    def __init__(
        self,
        *,
        node: Any,
        writer: TraceWriter,
        command_topic: str = "/control/move_j",
        feedback_topic: str = "/feedback/joint_states",
        qos_depth: int = 1,
        joint_names: Sequence[str] = JOINT_NAMES,
        gripper_name: str = "gripper",
        monitor_unsupported_topics: bool = True,
    ) -> None:
        if tuple(joint_names) != JOINT_NAMES:
            raise MotionTraceError(f"joint_names must be {list(JOINT_NAMES)!r}")
        if writer.header.get("source_timestamps_comparable", True):
            raise MotionTraceError(
                "ROS header stamps are not assumed comparable to monotonic capture time; "
                "construct TraceWriter with source_timestamps_comparable=False"
            )
        if int(qos_depth) <= 0:
            raise MotionTraceError("qos_depth must be positive")
        try:
            from sensor_msgs.msg import JointState
        except ImportError as exc:
            raise MotionTraceError(
                "ROS 2 recording requires sensor_msgs; offline trace validation/replay has no ROS dependency"
            ) from exc
        self.node = node
        self.writer = writer
        self.command_topic = command_topic
        self.feedback_topic = feedback_topic
        self.gripper_name = gripper_name
        self.unsupported_topics: list[str] = []
        self.setup_warnings: list[str] = []
        self._unsupported_reason: str | None = None
        self._last_feedback_source_timestamp_ns: int | None = None
        self._warned_missing_ros_clock = False
        self._latest_command: tuple[tuple[float, ...] | None, float | None, int | None] = (
            None,
            None,
            None,
        )
        self._command_subscription = node.create_subscription(
            JointState, command_topic, self._on_command, int(qos_depth)
        )
        self._feedback_subscription = node.create_subscription(
            JointState, feedback_topic, self._on_feedback, int(qos_depth)
        )
        self._unsupported_subscriptions = []
        if monitor_unsupported_topics:
            self._subscribe_unsupported(
                JointState,
                "/control/joint_states",
                qos_depth,
            )
            self._subscribe_unsupported(
                JointState,
                "/control/move_js",
                qos_depth,
            )
            try:
                from geometry_msgs.msg import PoseArray, PoseStamped
            except ImportError:
                self.setup_warnings.append("geometry_msgs unavailable; pose command topics are not monitored")
            else:
                self._subscribe_unsupported(PoseStamped, "/control/move_p", qos_depth)
                self._subscribe_unsupported(PoseStamped, "/control/move_l", qos_depth)
                self._subscribe_unsupported(PoseArray, "/control/move_c", qos_depth)
            try:
                from agx_arm_msgs.msg import MoveMITMsg
            except ImportError:
                self.setup_warnings.append("agx_arm_msgs unavailable; /control/move_mit is not monitored")
            else:
                self._subscribe_unsupported(MoveMITMsg, "/control/move_mit", qos_depth)

    def _subscribe_unsupported(self, message_type: Any, topic: str, qos_depth: int) -> None:
        self._unsupported_subscriptions.append(
            self.node.create_subscription(
                message_type,
                topic,
                lambda _message, unsupported_topic=topic: self._on_unsupported(unsupported_topic),
                int(qos_depth),
            )
        )
        self.unsupported_topics.append(topic)

    @staticmethod
    def _stamp_ns(message: Any) -> int | None:
        stamp = getattr(getattr(message, "header", None), "stamp", None)
        if stamp is None:
            return None
        try:
            seconds = int(getattr(stamp, "sec"))
            nanoseconds = int(getattr(stamp, "nanosec"))
        except (AttributeError, TypeError, ValueError):
            return None
        if seconds < 0 or nanoseconds < 0 or nanoseconds >= NANOSECONDS_PER_SECOND:
            return None
        if seconds == 0 and nanoseconds == 0:
            return None
        return seconds * NANOSECONDS_PER_SECOND + nanoseconds

    @staticmethod
    def _positions(message: Any, joint_names: Sequence[str], gripper_name: str):
        names = list(getattr(message, "name", ()))
        positions = list(getattr(message, "position", ()))
        mapping = {
            name: positions[index]
            for index, name in enumerate(names)
            if index < len(positions)
        }
        arm = None
        try:
            arm = tuple(_finite_float(mapping[name], f"ROS joint {name}") for name in joint_names)
        except (KeyError, MotionTraceError):
            arm = None
        gripper = None
        if gripper_name in mapping:
            try:
                gripper = _finite_float(mapping[gripper_name], "ROS gripper opening")
            except MotionTraceError:
                gripper = None
        return arm, gripper

    def _on_command(self, message: Any) -> None:
        joints, gripper = self._positions(message, JOINT_NAMES, self.gripper_name)
        source_timestamp = self._stamp_ns(message)
        self._latest_command = (joints, gripper, source_timestamp)
        if joints is not None:
            self._unsupported_reason = None
        # Persist every command event immediately. Feedback callbacks may be
        # sparse; caching alone would lose intermediate MOVEJ targets.
        self.writer.record(
            command_joint_position_rad=joints,
            command_gripper_full_opening_m=gripper,
            command_valid=joints is not None,
            command_gripper_valid=gripper is not None,
            command_source_timestamp_ns=source_timestamp,
            command_source="ros2:" + self.command_topic,
            feedback_joint_position_rad=None,
            feedback_valid=False,
        )

    def _on_unsupported(self, topic: str) -> None:
        # Do not guess how a Cartesian, MoveJS, or MIT command changed the
        # absolute joint target. Until a new MOVEJ command arrives, every
        # subsequent replay sample is explicitly invalid for joint ZOH.
        self._unsupported_reason = f"unsupported command topic observed: {topic}"
        self.writer.record(
            command_joint_position_rad=None,
            command_valid=False,
            command_source="ros2:" + self._unsupported_reason,
            command_validity_reasons=(self._unsupported_reason,),
            feedback_joint_position_rad=None,
            feedback_valid=False,
        )

    def _ros_clock_now_ns(self) -> int | None:
        try:
            clock = self.node.get_clock()
            now = clock.now()
            value = getattr(now, "nanoseconds", None)
            if value is not None:
                return _nonnegative_int(value, "ROS node clock nanoseconds")
            seconds_nanoseconds = getattr(now, "seconds_nanoseconds", None)
            if callable(seconds_nanoseconds):
                seconds, nanoseconds = seconds_nanoseconds()
                return _nonnegative_int(seconds, "ROS node clock seconds") * NANOSECONDS_PER_SECOND + _nonnegative_int(
                    nanoseconds, "ROS node clock nanoseconds"
                )
        except (AttributeError, MotionTraceError, TypeError, ValueError):
            pass
        if not self._warned_missing_ros_clock:
            self.setup_warnings.append(
                "ROS node clock unavailable; source freshness is reported as unknown"
            )
            self._warned_missing_ros_clock = True
        return None

    def _feedback_timestamp_valid(self, source_timestamp_ns: int | None) -> bool:
        if source_timestamp_ns is None:
            return False
        previous = self._last_feedback_source_timestamp_ns
        if previous is not None and source_timestamp_ns <= previous:
            return False
        self._last_feedback_source_timestamp_ns = source_timestamp_ns
        now = self._ros_clock_now_ns()
        if now is None:
            return True
        max_age_ns = int(float(self.writer.header["max_source_age_s"]) * NANOSECONDS_PER_SECOND)
        return abs(now - source_timestamp_ns) <= max_age_ns

    def _on_feedback(self, message: Any) -> MotionSample:
        joints, gripper = self._positions(message, JOINT_NAMES, self.gripper_name)
        command_joints, command_gripper, command_stamp = self._latest_command
        feedback_stamp = self._stamp_ns(message)
        feedback_timestamp_valid = self._feedback_timestamp_valid(feedback_stamp)
        # This callback only records observations. It does not call a ROS
        # publisher or a pyAgxArm getter/command method.
        sample = self.writer.record(
            command_joint_position_rad=command_joints,
            command_gripper_full_opening_m=command_gripper,
            command_valid=command_joints is not None and self._unsupported_reason is None,
            command_gripper_valid=command_gripper is not None,
            command_source_timestamp_ns=command_stamp,
            command_source=(
                "ros2:" + self.command_topic
                if self._unsupported_reason is None
                else "ros2:" + self.command_topic + ";" + self._unsupported_reason
            ),
            feedback_joint_position_rad=joints,
            feedback_gripper_full_opening_m=gripper,
            feedback_valid=joints is not None and feedback_timestamp_valid,
            feedback_gripper_valid=gripper is not None,
            feedback_source_timestamp_ns=feedback_stamp,
            feedback_source="ros2:" + self.feedback_topic,
            feedback_validity_reasons=(
                ()
                if feedback_timestamp_valid
                else ("missing, stale, or non-monotonic ROS feedback timestamp",)
            ),
        )
        return sample


def _component_from_mapping(
    payload: Any,
    *,
    label: str,
    sample_t_ns: int,
    max_source_age_s: float,
    source_timestamps_comparable: bool,
    previous_source_timestamp_ns: int | None,
    apply_staleness: bool,
    require_strict_source_progress: bool,
) -> tuple[MotionComponent, list[str], int | None]:
    issues: list[str] = []
    if payload is None:
        return MotionComponent(None, None, None, False, False, None, ("missing component",)), [], None
    if not isinstance(payload, Mapping):
        return MotionComponent(None, None, None, False, False, None, ("component is not an object",)), [
            f"{label}: component is not an object"
        ], None
    reasons = list(payload.get("validity_reasons") or [])
    raw_joints = payload.get("joint_position_rad")
    raw_gripper = payload.get("gripper_full_opening_m")
    try:
        joints = _finite_joint_vector(raw_joints)
    except MotionTraceError as exc:
        joints = None
        reasons.append(str(exc))
    try:
        gripper = _optional_finite(raw_gripper, f"{label}.gripper_full_opening_m")
    except MotionTraceError as exc:
        gripper = None
        reasons.append(str(exc))
    try:
        timestamp = (
            None
            if payload.get("source_timestamp_ns") is None
            else _nonnegative_int(payload.get("source_timestamp_ns"), f"{label}.source_timestamp_ns")
        )
    except MotionTraceError as exc:
        timestamp = None
        reasons.append(str(exc))
    declared_valid = payload.get("valid")
    declared_gripper_valid = payload.get("gripper_valid", False)
    if not isinstance(declared_valid, bool):
        declared_valid = False
        reasons.append("valid must be boolean")
    if not isinstance(declared_gripper_valid, bool):
        declared_gripper_valid = False
        reasons.append("gripper_valid must be boolean")
    if declared_valid and joints is None:
        reasons.append("declared valid without a finite six-joint vector")
    if declared_valid and timestamp is None:
        reasons.append("declared valid without source_timestamp_ns")
    if declared_gripper_valid and gripper is None:
        reasons.append("declared gripper_valid without finite gripper value")
    if declared_gripper_valid and timestamp is None:
        reasons.append("declared gripper_valid without source_timestamp_ns")
    if timestamp is not None:
        if previous_source_timestamp_ns is not None:
            if timestamp < previous_source_timestamp_ns:
                reasons.append("source timestamp is out of order")
            elif require_strict_source_progress and timestamp == previous_source_timestamp_ns:
                reasons.append("source timestamp is unchanged")
        if apply_staleness and source_timestamps_comparable:
            age_s = (sample_t_ns - timestamp) / NANOSECONDS_PER_SECOND
            if age_s > max_source_age_s:
                reasons.append(f"source data is stale ({age_s:.6f}s > {max_source_age_s:.6f}s)")
            if age_s < -max_source_age_s:
                reasons.append("source timestamp is too far in the future")
    if not declared_valid:
        reasons.append("declared invalid")
    valid = bool(declared_valid and joints is not None and timestamp is not None and not reasons)
    gripper_valid = bool(
        declared_gripper_valid and gripper is not None and timestamp is not None and not reasons
    )
    component = MotionComponent(
        joints,
        gripper,
        timestamp,
        valid,
        gripper_valid,
        payload.get("source") if isinstance(payload.get("source"), str) else None,
        tuple(sorted(set(str(reason) for reason in reasons))),
    )
    if component.validity_reasons:
        # Missing feedback is expected in command-only traces; it is still
        # explicitly invalid and therefore cannot be mistaken for feedback.
        issues.extend(f"{label}: {reason}" for reason in component.validity_reasons)
    return component, issues, timestamp


def _sample_from_mapping(
    payload: Mapping[str, Any],
    *,
    previous: MotionSample | None,
    max_source_age_s: float | None,
    source_timestamps_comparable: bool = True,
) -> MotionSample:
    try:
        seq = _nonnegative_int(payload.get("seq"), "seq")
        timestamp = _nonnegative_int(payload.get("t_monotonic_ns"), "t_monotonic_ns")
    except MotionTraceError as exc:
        raise MotionTraceError(f"invalid sample identity: {exc}") from exc
    issues: list[str] = []
    if previous is not None:
        if seq <= previous.seq:
            issues.append(f"seq {seq}: sequence is not strictly increasing")
        if timestamp <= previous.t_monotonic_ns:
            issues.append(f"seq {seq}: capture timestamp is not strictly increasing")
    age = DEFAULT_MAX_SOURCE_AGE_S if max_source_age_s is None else float(max_source_age_s)
    command, command_issues, command_source_timestamp = _component_from_mapping(
        payload.get("command"),
        label=f"seq {seq} command",
        sample_t_ns=timestamp,
        max_source_age_s=age,
        source_timestamps_comparable=source_timestamps_comparable,
        previous_source_timestamp_ns=(
            previous.command.source_timestamp_ns if previous is not None else None
        ),
        apply_staleness=False,
        require_strict_source_progress=False,
    )
    feedback, feedback_issues, feedback_source_timestamp = _component_from_mapping(
        payload.get("feedback"),
        label=f"seq {seq} feedback",
        sample_t_ns=timestamp,
        max_source_age_s=age,
        source_timestamps_comparable=source_timestamps_comparable,
        previous_source_timestamp_ns=(
            previous.feedback.source_timestamp_ns if previous is not None else None
        ),
        apply_staleness=True,
        require_strict_source_progress=True,
    )
    del command_source_timestamp, feedback_source_timestamp
    # A missing feedback component is an expected warning for command-only
    # captures. All other invalidity is retained explicitly on the sample.
    for issue in command_issues + feedback_issues:
        if "feedback: missing component" not in issue:
            issues.append(issue)
    record_valid = not any(
        "command" in issue or "sequence" in issue or "capture timestamp" in issue
        for issue in issues
    )
    return MotionSample(seq, timestamp, command, feedback, record_valid, tuple(issues))


def _validate_header(header: Any) -> dict[str, Any]:
    if not isinstance(header, Mapping):
        raise MotionTraceError("trace header must be an object")
    expected = {
        "record_type": "header",
        "schema": TRACE_SCHEMA,
        "schema_version": TRACE_SCHEMA_VERSION,
    }
    for key, value in expected.items():
        if header.get(key) != value:
            raise MotionTraceError(f"header {key!r} must be {value!r}")
    names = header.get("joint_names")
    if tuple(names or ()) != JOINT_NAMES:
        raise MotionTraceError(f"header joint_names must be {list(JOINT_NAMES)!r}")
    units = header.get("units")
    if not isinstance(units, Mapping) or units.get("joint_position") != "rad" or units.get("gripper") != "m_full_opening":
        raise MotionTraceError("header units must declare rad joints and full-opening metres")
    try:
        age = _finite_float(header.get("max_source_age_s", DEFAULT_MAX_SOURCE_AGE_S), "max_source_age_s")
    except MotionTraceError:
        raise
    if age <= 0:
        raise MotionTraceError("header max_source_age_s must be positive")
    if not isinstance(header.get("source_timestamps_comparable", True), bool):
        raise MotionTraceError("header source_timestamps_comparable must be boolean")
    if not isinstance(header.get("initial_state_assumption"), str):
        raise MotionTraceError("header initial_state_assumption must be a string")
    return dict(header)


def read_trace(
    path: str | Path,
    *,
    strict: bool = True,
    max_source_age_s: float | None = None,
) -> MotionTrace:
    """Read and validate JSONL without importing NumPy, MuJoCo, or the SDK."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    header: dict[str, Any] | None = None
    samples: list[MotionSample] = []
    issues: list[str] = []
    with source.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            if not raw_line.strip():
                issues.append(f"line {line_number}: blank lines are not valid records")
                continue
            try:
                payload = _json_loads_strict(raw_line)
                if not isinstance(payload, Mapping):
                    raise MotionTraceError("record must be an object")
                record_type = payload.get("record_type")
                if header is None:
                    if record_type != "header":
                        raise MotionTraceError("first record must be a header")
                    header = _validate_header(payload)
                    continue
                if record_type != "sample":
                    raise MotionTraceError("records after the header must be samples")
                comparable = bool(header.get("source_timestamps_comparable", True))
                sample = _sample_from_mapping(
                    payload,
                    previous=samples[-1] if samples else None,
                    max_source_age_s=(
                        max_source_age_s
                        if max_source_age_s is not None
                        else float(header.get("max_source_age_s", DEFAULT_MAX_SOURCE_AGE_S))
                    ),
                    source_timestamps_comparable=comparable,
                )
                samples.append(sample)
                issues.extend(f"line {line_number}: {issue}" for issue in sample.issues)
            except MotionTraceError as exc:
                issues.append(f"line {line_number}: {exc}")
                if strict:
                    raise MotionTraceError(f"invalid trace at line {line_number}", issues=issues) from exc
    if header is None:
        raise MotionTraceError("trace has no header")
    trace = MotionTrace(header, tuple(samples), tuple(issues))
    if strict and trace.errors:
        raise MotionTraceError("trace contains invalid data", issues=trace.errors)
    return trace


def _load_piper_model():
    """Build the project Piper model lazily for offline replay only."""

    try:
        import mujoco
        from scripts.piper_pick_place_env import build_model
    except ImportError as exc:
        raise MotionTraceError(
            "MuJoCo replay requires the managed simulator environment; trace validation is dependency-free"
        ) from exc
    return mujoco, build_model()


def _configure_contact_free_scene(model: Any) -> int:
    """Disable task-geometry collisions for an explicit joint-response bench."""

    robot_root = int(model.body("base_link").id)
    robot_bodies = set()
    for body_id in range(model.nbody):
        current = body_id
        while current != 0:
            if current == robot_root:
                robot_bodies.add(body_id)
                break
            current = int(model.body_parentid[current])
    non_robot = [
        geom_id
        for geom_id, body_id in enumerate(model.geom_bodyid)
        if int(body_id) not in robot_bodies
    ]
    if non_robot:
        model.geom_contype[non_robot] = 0
        model.geom_conaffinity[non_robot] = 0
    return len(non_robot)


def _model_metadata(model: Any, *, scene_mode: str, collision_disabled_geom_count: int = 0) -> dict[str, Any]:
    names = list(JOINT_NAMES)
    joint_ranges = []
    for name in names:
        joint_id = int(model.joint(name).id)
        joint_ranges.append([float(value) for value in model.jnt_range[joint_id]])
    gripper_range = [float(value) for value in model.jnt_range[int(model.joint("joint7").id)]]
    robot_bodies = ["base_link", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "link8"]
    gravcomp = {}
    for name in robot_bodies:
        body_id = int(model.body(name).id)
        gravcomp[name] = float(model.body_gravcomp[body_id])
    actuator_gain = {}
    actuator_bias = {}
    actuator_force_range = {}
    for name in (*JOINT_NAMES, "gripper"):
        actuator_id = int(model.actuator(name).id)
        actuator_gain[name] = [float(value) for value in model.actuator_gainprm[actuator_id]]
        actuator_bias[name] = [float(value) for value in model.actuator_biasprm[actuator_id]]
        actuator_force_range[name] = [float(value) for value in model.actuator_forcerange[actuator_id]]
    return {
        "backend": "mujoco_cpu_python",
        "model_source": "scripts.piper_pick_place_env.build_model",
        "scene_mode": scene_mode,
        "collision_disabled_geom_count": collision_disabled_geom_count,
        "timestep_s": float(model.opt.timestep),
        "gravity_m_s2": [float(value) for value in model.opt.gravity],
        "integrator": int(model.opt.integrator),
        "iterations": int(model.opt.iterations),
        "joint_names": names,
        "joint_ranges_rad": joint_ranges,
        "gripper_joint": "joint7",
        "gripper_model_half_travel_range_m": gripper_range,
        "gripper_full_opening_range_m": [2.0 * value for value in gripper_range],
        "robot_body_gravcomp": gravcomp,
        "actuator_gainprm": actuator_gain,
        "actuator_biasprm": actuator_bias,
        "actuator_force_ranges": actuator_force_range,
    }


def _model_ids(model: Any) -> tuple[list[int], int, int, int]:
    arm_joints = [int(model.joint(name).id) for name in JOINT_NAMES]
    arm_qpos = [int(model.jnt_qposadr[joint_id]) for joint_id in arm_joints]
    arm_actuators = [int(model.actuator(name).id) for name in JOINT_NAMES]
    gripper_joint_id = int(model.joint("joint7").id)
    gripper_qpos = int(model.jnt_qposadr[gripper_joint_id])
    gripper_actuator = int(model.actuator("gripper").id)
    return arm_qpos, gripper_qpos, arm_actuators, gripper_actuator


def _range_issues(trace: MotionTrace, model: Any) -> list[str]:
    import numpy as np

    _, _, _, _ = _model_ids(model)
    joint_ranges = np.asarray(
        [model.jnt_range[int(model.joint(name).id)] for name in JOINT_NAMES], dtype=float
    )
    gripper_range = np.asarray(model.jnt_range[int(model.joint("joint7").id)], dtype=float)
    issues: list[str] = []
    for sample in trace.samples:
        for label, component in (("command", sample.command), ("feedback", sample.feedback)):
            if component.valid and component.joint_position_rad is not None:
                for index, value in enumerate(component.joint_position_rad):
                    low, high = joint_ranges[index]
                    tolerance = (
                        MODEL_FEEDBACK_RANGE_TOLERANCE_RAD
                        if label == "feedback"
                        else MODEL_RANGE_TOLERANCE_RAD
                    )
                    if value < low - tolerance or value > high + tolerance:
                        issues.append(
                            f"seq {sample.seq} {label} {JOINT_NAMES[index]}={value:.9g} rad "
                            f"outside model range [{low:.9g}, {high:.9g}]"
                        )
            if component.gripper_valid and component.gripper_full_opening_m is not None:
                half = full_opening_m_to_model_half_travel(component.gripper_full_opening_m)
                low, high = gripper_range
                tolerance = (
                    MODEL_FEEDBACK_RANGE_TOLERANCE_M
                    if label == "feedback"
                    else MODEL_RANGE_TOLERANCE_M
                )
                if half < low - tolerance or half > high + tolerance:
                    issues.append(
                        f"seq {sample.seq} {label} full gripper opening="
                        f"{component.gripper_full_opening_m:.9g} m maps to half travel {half:.9g} m, "
                        f"outside model range [{low:.9g}, {high:.9g}]"
                    )
    return issues


def _metric(errors: Sequence[float], *, unit: str) -> dict[str, Any]:
    if not errors:
        return {"status": "unavailable", "samples": 0, "unit": unit}
    values = [abs(float(value)) for value in errors]
    return {
        "status": "ok",
        "samples": len(values),
        "unit": unit,
        "mae": sum(values) / len(values),
        "rmse": math.sqrt(sum(value * value for value in values) / len(values)),
        "max_abs": max(values),
    }


def _integrate_duration(
    mujoco_module: Any,
    model: Any,
    data: Any,
    duration_s: float,
    *,
    time_tolerance_s: float,
    max_steps_per_interval: int,
) -> tuple[int, bool]:
    """Integrate an event interval without rounding away ROS callback jitter.

    Full nominal MuJoCo steps are followed by one positive fractional step.
    The model timestep is restored immediately after that step, so replay uses
    the training model's timestep and actuator semantics for the next event.
    """

    if duration_s <= 0 or not math.isfinite(duration_s):
        raise MotionTraceError(f"replay interval must be finite and positive, got {duration_s!r}")
    nominal_timestep = float(model.opt.timestep)
    if nominal_timestep <= 0 or not math.isfinite(nominal_timestep):
        raise MotionTraceError(f"model timestep must be finite and positive, got {nominal_timestep!r}")
    full_steps = int(math.floor(duration_s / nominal_timestep))
    remainder = duration_s - full_steps * nominal_timestep
    if remainder < 0:
        remainder = 0.0
    if remainder <= time_tolerance_s:
        remainder = 0.0
    elif nominal_timestep - remainder <= time_tolerance_s:
        full_steps += 1
        remainder = 0.0
    required_steps = full_steps + (1 if remainder > 0 else 0)
    if required_steps <= 0 or required_steps > max_steps_per_interval:
        raise MotionTraceError(
            f"replay interval {duration_s:.9g}s requires {required_steps} MuJoCo steps"
        )
    for _ in range(full_steps):
        mujoco_module.mj_step(model, data)
    fractional = remainder > 0.0
    if fractional:
        try:
            model.opt.timestep = remainder
            mujoco_module.mj_step(model, data)
        finally:
            model.opt.timestep = nominal_timestep
    return required_steps, fractional


def replay_trace(
    trace: MotionTrace,
    *,
    model: Any | None = None,
    mujoco_module: Any | None = None,
    require_initial_feedback: bool = True,
    contact_free: bool = False,
    time_tolerance_s: float = DEFAULT_TIME_TOLERANCE_S,
    max_steps_per_interval: int = DEFAULT_MAX_STEPS_PER_INTERVAL,
) -> dict[str, Any]:
    """Replay absolute commands with ZOH and compare feedback without teleporting.

    The first valid joint feedback seeds ``qpos`` once. Every later state comes
    from ``mj_step``; later feedback is used only for error measurement.
    """

    if trace.errors:
        raise MotionTraceError("cannot replay an invalid trace", issues=trace.errors)
    if not trace.samples:
        raise MotionTraceError("cannot replay an empty trace")
    source_timestamps_comparable = bool(trace.header.get("source_timestamps_comparable", True))
    if mujoco_module is None or model is None:
        mujoco_module, model = _load_piper_model()
    collision_disabled_geom_count = (
        _configure_contact_free_scene(model) if contact_free else 0
    )
    scene_mode = (
        "contact_free_non_robot_geometry_collisions_disabled"
        if contact_free
        else "training_scene_with_task_geometry_contacts"
    )
    model_issues = _range_issues(trace, model)
    if model_issues:
        raise MotionTraceRangeError("trace values do not fit the selected MuJoCo model", issues=model_issues)
    try:
        import numpy as np
    except ImportError as exc:
        raise MotionTraceError("MuJoCo replay requires NumPy") from exc
    feedback_indices = [
        index for index, sample in enumerate(trace.samples) if sample.feedback.valid
    ]
    if not feedback_indices:
        if require_initial_feedback:
            raise MotionTraceError("no valid joint feedback is available to seed initial qpos")
        initial_index = 0
    else:
        initial_index = feedback_indices[0]
    if initial_index >= len(trace.samples):
        raise MotionTraceError("initial feedback index is outside the trace")
    command_indices = [
        index
        for index, sample in enumerate(trace.samples[: initial_index + 1])
        if sample.command.valid
    ]
    if not command_indices:
        raise MotionTraceError("no valid command is available at the initial feedback time")
    initial_command_index = command_indices[-1]
    arm_qpos, gripper_qpos, arm_actuators, gripper_actuator = _model_ids(model)
    data = mujoco_module.MjData(model)
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    initial_feedback = trace.samples[initial_index].feedback
    if initial_feedback.valid and initial_feedback.joint_position_rad is not None:
        data.qpos[arm_qpos] = np.asarray(initial_feedback.joint_position_rad, dtype=float)
    if initial_feedback.gripper_valid and initial_feedback.gripper_full_opening_m is not None:
        half = full_opening_m_to_model_half_travel(initial_feedback.gripper_full_opening_m)
        data.qpos[gripper_qpos] = half
        other_gripper_qpos = int(model.jnt_qposadr[int(model.joint("joint8").id)])
        data.qpos[other_gripper_qpos] = -half
    mujoco_module.mj_forward(model, data)

    initial_gripper_half = (
        full_opening_m_to_model_half_travel(initial_feedback.gripper_full_opening_m)
        if initial_feedback.gripper_valid and initial_feedback.gripper_full_opening_m is not None
        else None
    )
    if initial_gripper_half is not None:
        # If command gripper data is absent, hold the observed initial target;
        # do not let the model's default zero ctrl close the gripper.
        data.ctrl[gripper_actuator] = initial_gripper_half

    def apply_command(component: MotionComponent) -> None:
        if not component.valid or component.joint_position_rad is None:
            return
        data.ctrl[arm_actuators] = np.asarray(component.joint_position_rad, dtype=float)
        if component.gripper_valid and component.gripper_full_opening_m is not None:
            data.ctrl[gripper_actuator] = full_opening_m_to_model_half_travel(
                component.gripper_full_opening_m
            )

    apply_command(trace.samples[initial_command_index].command)
    joint_errors: dict[str, list[float]] = {name: [] for name in JOINT_NAMES}
    gripper_errors: list[float] = []
    replay_steps = 0
    fractional_intervals = 0
    integrated_duration_s = 0.0
    feedback_compared = 0
    final_simulated_joint_position = None
    final_simulated_gripper_opening = None
    for index in range(initial_index, len(trace.samples)):
        sample = trace.samples[index]
        if index > initial_index:
            previous = trace.samples[index - 1]
            delta_s = (sample.t_monotonic_ns - previous.t_monotonic_ns) / NANOSECONDS_PER_SECOND
            if delta_s <= 0:
                raise MotionTraceError(f"seq {sample.seq}: non-positive replay interval")
            step_count, fractional = _integrate_duration(
                mujoco_module,
                model,
                data,
                delta_s,
                time_tolerance_s=time_tolerance_s,
                max_steps_per_interval=max_steps_per_interval,
            )
            replay_steps += step_count
            fractional_intervals += int(fractional)
            integrated_duration_s += delta_s
        if index > initial_index and sample.feedback.valid and sample.feedback.joint_position_rad is not None:
            simulated = np.asarray(data.qpos[arm_qpos], dtype=float)
            final_simulated_joint_position = simulated.copy()
            for joint_index, name in enumerate(JOINT_NAMES):
                joint_errors[name].append(
                    float(simulated[joint_index] - sample.feedback.joint_position_rad[joint_index])
                )
            feedback_compared += 1
            if sample.feedback.gripper_valid and sample.feedback.gripper_full_opening_m is not None:
                simulated_opening = model_half_travel_to_full_opening_m(float(data.qpos[gripper_qpos]))
                final_simulated_gripper_opening = simulated_opening
                gripper_errors.append(
                    simulated_opening - sample.feedback.gripper_full_opening_m
                )
        # Commands are zero-order-held from their sample timestamp until the
        # next valid command; this assignment never writes qpos.
        apply_command(sample.command)

    return {
        "status": "ok",
        "trace_schema": trace.header.get("schema"),
        "trace_schema_version": trace.header.get("schema_version"),
        "command_semantics": trace.header.get("command_semantics"),
        "initial_state_assumption": trace.header.get("initial_state_assumption"),
        "initial_feedback_seq": trace.samples[initial_index].seq if feedback_indices else None,
        "skipped_prefix_samples": initial_index,
        "replay_steps": replay_steps,
        "fractional_intervals": fractional_intervals,
        "integrated_duration_s": integrated_duration_s,
        "feedback_samples_compared": feedback_compared,
        "feedback_teleports": 0,
        "final_simulated_joint_position_rad": (
            final_simulated_joint_position.tolist()
            if final_simulated_joint_position is not None
            else None
        ),
        "final_simulated_gripper_opening_m": final_simulated_gripper_opening,
        "zoh": True,
        "source_timestamp_alignment": (
            "comparable" if source_timestamps_comparable else "not_comparable"
        ),
        "stale_source_detection": source_timestamps_comparable,
        "model": _model_metadata(
            model,
            scene_mode=scene_mode,
            collision_disabled_geom_count=collision_disabled_geom_count,
        ),
        "joint_metrics": {
            name: _metric(errors, unit="rad") for name, errors in joint_errors.items()
        },
        "gripper_metrics": _metric(gripper_errors, unit="m_full_opening"),
    }


def _trace_summary(trace: MotionTrace) -> dict[str, Any]:
    valid_commands = sum(sample.command.valid for sample in trace.samples)
    valid_feedback = sum(sample.feedback.valid for sample in trace.samples)
    valid_gripper_feedback = sum(sample.feedback.gripper_valid for sample in trace.samples)
    return {
        "status": "ok" if not trace.errors else "invalid",
        "schema": trace.header.get("schema"),
        "schema_version": trace.header.get("schema_version"),
        "samples": len(trace.samples),
        "valid_commands": valid_commands,
        "valid_joint_feedback": valid_feedback,
        "valid_gripper_feedback": valid_gripper_feedback,
        "invalid_samples": trace.invalid_sample_count,
        "issues": list(trace.errors),
    }


def _write_json_output(payload: Mapping[str, Any], path: str | Path | None) -> None:
    text = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    if path is None:
        print(text, end="")
    else:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
        print(f"Wrote {destination}")


def _record_ros(args: argparse.Namespace) -> int:
    """Run the passive ROS2 subscriber recorder; never create a publisher."""

    try:
        import rclpy
    except ImportError as exc:
        raise MotionTraceError(
            "record-ros requires a ROS 2 environment with rclpy; use validate/replay offline"
        ) from exc
    metadata = {
        "controller": {
            "mode": args.controller_mode,
            "speed_percent_metadata": args.speed_percent,
            "publisher_rate_hz_metadata": args.publisher_rate_hz,
            "command_topic": args.command_topic,
            "feedback_topic": args.feedback_topic,
        },
        "firmware": {"version": args.firmware_version},
        "source_timestamps_comparable": False,
        "source_clock": "ros_header_stamp",
        "max_source_age_s": args.max_source_age_s,
        "initial_state_assumption": "first_valid_feedback_qpos_once_then_zero_joint_velocity",
    }
    writer = TraceWriter(args.trace, metadata=metadata, overwrite=args.overwrite)
    node = None
    recorder = None
    status = "completed"
    try:
        rclpy.init(args=None)
        node = rclpy.create_node(args.node_name)
        recorder = Ros2JointStateRecorder(
            node=node,
            writer=writer,
            command_topic=args.command_topic,
            feedback_topic=args.feedback_topic,
            qos_depth=args.qos_depth,
            monitor_unsupported_topics=True,
        )
        deadline = (
            time.monotonic() + args.duration_s if args.duration_s is not None else None
        )
        while rclpy.ok():
            if args.max_samples is not None and writer.samples_written >= args.max_samples:
                break
            if deadline is not None and time.monotonic() >= deadline:
                break
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        status = "interrupted"
    finally:
        writer.close()
        if node is not None:
            try:
                node.destroy_node()
            except (AttributeError, RuntimeError):
                pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except (AttributeError, RuntimeError):
            pass
    summary = {
        "status": status,
        "trace": str(args.trace),
        "samples_written": writer.samples_written,
        "command_topic": args.command_topic,
        "feedback_topic": args.feedback_topic,
        "unsupported_topics_monitored": recorder.unsupported_topics if recorder else [],
        "setup_warnings": recorder.setup_warnings if recorder else [],
        "publisher_count": 0,
        "motion_api_calls": 0,
    }
    _write_json_output(summary, None)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate JSONL structure and timing")
    validate.add_argument("trace", type=Path)
    validate.add_argument("--output", type=Path)
    validate.add_argument("--max-source-age-s", type=float, default=None)
    replay = subparsers.add_parser("replay", help="replay commands in the CPU Piper MuJoCo model")
    replay.add_argument("trace", type=Path)
    replay.add_argument("--output", type=Path)
    replay.add_argument("--allow-missing-feedback", action="store_true")
    replay.add_argument(
        "--contact-free",
        action="store_true",
        help="disable non-robot geometry collisions for an explicit joint-response bench",
    )
    replay.add_argument("--time-tolerance-s", type=float, default=DEFAULT_TIME_TOLERANCE_S)
    record = subparsers.add_parser(
        "record-ros",
        help="passively record ROS2 MOVEJ/feedback topics; never publishes commands",
    )
    record.add_argument("trace", type=Path)
    record.add_argument("--duration-s", type=_positive_float_arg, default=None)
    record.add_argument("--max-samples", type=_positive_int_arg, default=None)
    record.add_argument("--overwrite", action="store_true")
    record.add_argument("--node-name", default="piper_motion_trace_recorder")
    record.add_argument("--command-topic", default="/control/move_j")
    record.add_argument("--feedback-topic", default="/feedback/joint_states")
    record.add_argument("--qos-depth", type=_positive_int_arg, default=1)
    record.add_argument("--controller-mode", default="MOVEJ-only")
    record.add_argument("--speed-percent", type=float, default=None)
    record.add_argument("--publisher-rate-hz", type=float, default=None)
    record.add_argument("--firmware-version", default="unknown")
    record.add_argument("--max-source-age-s", type=_positive_float_arg, default=DEFAULT_MAX_SOURCE_AGE_S)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate":
            trace = read_trace(args.trace, strict=False, max_source_age_s=args.max_source_age_s)
            payload = _trace_summary(trace)
            _write_json_output(payload, args.output)
            return 0 if payload["status"] == "ok" else 2
        if args.command == "record-ros":
            return _record_ros(args)
        trace = read_trace(args.trace, strict=True)
        payload = replay_trace(
            trace,
            require_initial_feedback=not args.allow_missing_feedback,
            contact_free=args.contact_free,
            time_tolerance_s=args.time_tolerance_s,
        )
        _write_json_output(payload, args.output)
        return 0
    except (MotionTraceError, FileNotFoundError) as exc:
        payload = {"status": "error", "error": str(exc)}
        _write_json_output(payload, getattr(args, "output", None))
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
