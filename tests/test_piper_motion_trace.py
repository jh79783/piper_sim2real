"""Strict offline trace, unit conversion, and MuJoCo replay tests."""

import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
import io
from unittest.mock import patch

from scripts.piper_motion_trace import (
    JOINT_NAMES,
    MotionTraceError,
    MotionTraceRangeError,
    TraceWriter,
    full_opening_m_to_model_half_travel,
    millidegree_to_rad,
    micrometer_full_opening_to_model_half_travel,
    model_half_travel_to_micrometer_full_opening,
    model_half_travel_to_full_opening_m,
    rad_to_millidegree,
    read_trace,
    replay_trace,
)


class MotionTraceContractTests(unittest.TestCase):
    def test_unit_converters_use_full_opening_and_millidegrees(self):
        import math

        self.assertAlmostEqual(millidegree_to_rad(180_000), math.pi)
        self.assertAlmostEqual(rad_to_millidegree(math.pi), 180_000.0)
        self.assertAlmostEqual(micrometer_full_opening_to_model_half_travel(70_000), 0.035)
        self.assertAlmostEqual(model_half_travel_to_micrometer_full_opening(0.035), 70_000.0)
        self.assertAlmostEqual(full_opening_m_to_model_half_travel(0.07), 0.035)
        self.assertAlmostEqual(model_half_travel_to_full_opening_m(0.035), 0.07)

    def test_writer_persists_metadata_and_passive_si_validity(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace.jsonl"
            with TraceWriter(
                path,
                metadata={
                    "controller": {"name": "verified-test-controller"},
                    "firmware": {"version": "fixture"},
                },
            ) as writer:
                writer.record(
                    command_joint_position_rad=[0.0] * 6,
                    command_gripper_full_opening_m=0.07,
                    feedback_joint_position_rad=[0.0] * 6,
                    feedback_gripper_full_opening_m=0.07,
                    t_monotonic_ns=1_000_000_000,
                    feedback_source_timestamp_ns=1_000_000_000,
                )
            trace = read_trace(path)
            self.assertEqual(trace.header["schema_version"], 1)
            self.assertEqual(trace.header["controller"]["name"], "verified-test-controller")
            self.assertTrue(trace.samples[0].command.valid)
            self.assertTrue(trace.samples[0].feedback.valid)
            self.assertTrue(trace.samples[0].feedback.gripper_valid)
            self.assertEqual(trace.samples[0].feedback.joint_position_rad, (0.0,) * 6)

    def test_writer_rejects_out_of_order_capture(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace.jsonl"
            with TraceWriter(path) as writer:
                writer.record(command_joint_position_rad=[0.0] * 6, t_monotonic_ns=10)
                with self.assertRaises(MotionTraceError):
                    writer.record(command_joint_position_rad=[0.0] * 6, t_monotonic_ns=10)

    def test_nonfinite_and_stale_feedback_never_becomes_valid(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace.jsonl"
            with TraceWriter(path, metadata={"max_source_age_s": 0.01}) as writer:
                writer.record(
                    command_joint_position_rad=[0.0] * 6,
                    feedback_joint_position_rad=[0.0] * 6,
                    t_monotonic_ns=1_000_000_000,
                    feedback_source_timestamp_ns=0,
                )
            trace = read_trace(path, strict=False)
            self.assertFalse(trace.samples[0].feedback.valid)
            self.assertTrue(any("stale" in issue for issue in trace.errors))

            lines = path.read_text(encoding="utf-8").splitlines()
            sample = json.loads(lines[1])
            sample["command"]["joint_position_rad"][0] = "NaN"
            path.write_text(lines[0] + "\n" + json.dumps(sample) + "\n", encoding="utf-8")
            with self.assertRaises(MotionTraceError):
                read_trace(path, strict=True)

    def test_ros_adapter_helpers_do_not_require_ros_import(self):
        from scripts.piper_motion_trace import Ros2JointStateRecorder

        class Stamp:
            sec = 2
            nanosec = 3

        class Header:
            stamp = Stamp()

        class Message:
            header = Header()
            name = [*JOINT_NAMES, "gripper"]
            position = [0.0] * 6 + [0.07]

        self.assertEqual(Ros2JointStateRecorder._stamp_ns(Message()), 2_000_000_003)
        joints, gripper = Ros2JointStateRecorder._positions(Message(), JOINT_NAMES, "gripper")
        self.assertEqual(joints, (0.0,) * 6)
        self.assertEqual(gripper, 0.07)

    def test_ros_recorder_is_passive_and_invalidates_unsupported_segments(self):
        from scripts.piper_motion_trace import Ros2JointStateRecorder

        class FakeNode:
            def __init__(self):
                self.subscriptions = []

            def create_subscription(self, message_type, topic, callback, depth):
                self.subscriptions.append((message_type, topic, callback, depth))
                return object()

        class JointState:
            def __init__(self):
                self.name = [*JOINT_NAMES, "gripper"]
                self.position = [0.0] * 6 + [0.07]
                self.header = types.SimpleNamespace(
                    stamp=types.SimpleNamespace(sec=2, nanosec=3)
                )

        sensor_msgs = types.ModuleType("sensor_msgs")
        sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")
        sensor_msgs_msg.JointState = JointState
        sensor_msgs.msg = sensor_msgs_msg
        geometry_msgs = types.ModuleType("geometry_msgs")
        geometry_msgs_msg = types.ModuleType("geometry_msgs.msg")
        geometry_msgs_msg.PoseArray = type("PoseArray", (), {})
        geometry_msgs_msg.PoseStamped = type("PoseStamped", (), {})
        geometry_msgs.msg = geometry_msgs_msg
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ros.jsonl"
            with TraceWriter(
                path,
                metadata={"source_timestamps_comparable": False},
            ) as writer:
                with patch.dict(
                    sys.modules,
                    {
                        "sensor_msgs": sensor_msgs,
                        "sensor_msgs.msg": sensor_msgs_msg,
                        "geometry_msgs": geometry_msgs,
                        "geometry_msgs.msg": geometry_msgs_msg,
                    },
                ):
                    node = FakeNode()
                    recorder = Ros2JointStateRecorder(node=node, writer=writer)
                    topics = {topic for _, topic, _, _ in node.subscriptions}
                    self.assertIn("/control/move_j", topics)
                    self.assertIn("/feedback/joint_states", topics)
                    self.assertIn("/control/move_js", topics)
                    recorder._on_command(JointState())
                    recorder._on_unsupported("/control/move_js")
                    sample = recorder._on_feedback(JointState())
                    self.assertFalse(sample.command.valid)
                    self.assertTrue(sample.feedback.valid)
                    self.assertIn("/control/move_js", recorder.unsupported_topics)
            trace = read_trace(path, strict=False)
            self.assertTrue(trace.samples[0].command.valid)
            self.assertFalse(trace.samples[-1].command.valid)
            self.assertTrue(any("declared invalid" in issue for issue in trace.errors))

    def test_record_ros_cli_stub_only_subscribes_and_writes_header(self):
        from scripts.piper_motion_trace import main

        class FakeNode:
            def __init__(self):
                self.subscriptions = []

            def create_subscription(self, message_type, topic, callback, depth):
                self.subscriptions.append((message_type, topic, callback, depth))
                return object()

            def destroy_node(self):
                return None

        fake_node = FakeNode()
        fake_rclpy = types.ModuleType("rclpy")
        fake_rclpy.init = lambda args=None: None
        fake_rclpy.create_node = lambda name: fake_node
        fake_rclpy.ok = lambda: True
        fake_rclpy.spin_once = lambda node, timeout_sec=0.1: None
        fake_rclpy.shutdown = lambda: None

        class JointState:
            pass

        sensor_msgs = types.ModuleType("sensor_msgs")
        sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")
        sensor_msgs_msg.JointState = JointState
        sensor_msgs.msg = sensor_msgs_msg
        geometry_msgs = types.ModuleType("geometry_msgs")
        geometry_msgs_msg = types.ModuleType("geometry_msgs.msg")
        geometry_msgs_msg.PoseArray = type("PoseArray", (), {})
        geometry_msgs_msg.PoseStamped = type("PoseStamped", (), {})
        geometry_msgs.msg = geometry_msgs_msg
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "recorded.jsonl"
            output = io.StringIO()
            with patch.dict(
                sys.modules,
                {
                    "rclpy": fake_rclpy,
                    "sensor_msgs": sensor_msgs,
                    "sensor_msgs.msg": sensor_msgs_msg,
                    "geometry_msgs": geometry_msgs,
                    "geometry_msgs.msg": geometry_msgs_msg,
                },
            ), redirect_stdout(output):
                self.assertEqual(
                    main(["record-ros", str(path), "--duration-s", "0.001"]),
                    0,
                )
            summary = json.loads(output.getvalue())
            self.assertEqual(summary["publisher_count"], 0)
            self.assertEqual(summary["motion_api_calls"], 0)
            self.assertEqual(summary["samples_written"], 0)
            trace = read_trace(path)
            self.assertEqual(trace.header["command_semantics"], "absolute_joint_target_si_v1")


class MotionTraceReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mujoco

        from scripts.piper_pick_place_env import build_model

        cls.mujoco = mujoco
        cls.model = build_model()

    def _fixture_trace(self, path: Path, *, perturb_feedback: bool = False):
        import numpy as np

        model = self.model
        data = self.mujoco.MjData(model)
        data.qpos[:] = model.qpos0
        arm_qpos = [int(model.jnt_qposadr[int(model.joint(name).id)]) for name in JOINT_NAMES]
        arm_actuators = [int(model.actuator(name).id) for name in JOINT_NAMES]
        gripper_qpos = int(model.jnt_qposadr[int(model.joint("joint7").id)])
        command = [float(data.qpos[index]) for index in arm_qpos]
        opening = float(data.qpos[gripper_qpos]) * 2.0
        data.ctrl[arm_actuators] = np.asarray(command)
        data.ctrl[int(model.actuator("gripper").id)] = float(data.qpos[gripper_qpos])
        self.mujoco.mj_forward(model, data)
        with TraceWriter(path, metadata={"model": "piper_training_build_model"}) as writer:
            timestamp = 1_000_000_000
            for index in range(4):
                feedback = [float(data.qpos[position]) for position in arm_qpos]
                if perturb_feedback and index == 1:
                    feedback[0] += 0.05
                writer.record(
                    command_joint_position_rad=command,
                    command_gripper_full_opening_m=opening,
                    feedback_joint_position_rad=feedback,
                    feedback_gripper_full_opening_m=opening,
                    t_monotonic_ns=timestamp,
                    feedback_source_timestamp_ns=timestamp,
                )
                self.mujoco.mj_step(model, data)
                timestamp += int(round(float(model.opt.timestep) * 1_000_000_000))

    def test_replay_uses_zoh_and_scores_feedback_after_one_initial_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.jsonl"
            self._fixture_trace(path)
            trace = read_trace(path)
            report = replay_trace(trace, model=self.model, mujoco_module=self.mujoco)
            self.assertTrue(report["zoh"])
            self.assertEqual(report["feedback_teleports"], 0)
            self.assertEqual(report["replay_steps"], 3)
            self.assertEqual(report["feedback_samples_compared"], 3)
            self.assertLess(report["joint_metrics"]["joint1"]["max_abs"], 1e-10)
            self.assertLess(report["gripper_metrics"]["max_abs"], 1e-10)
            self.assertEqual(report["model"]["backend"], "mujoco_cpu_python")
            self.assertEqual(report["model"]["timestep_s"], 0.002)
            self.assertEqual(
                report["initial_state_assumption"],
                "first_valid_feedback_qpos_once_then_zero_joint_velocity",
            )
            self.assertTrue(all(value == 1.0 for value in report["model"]["robot_body_gravcomp"].values()))

    def test_later_feedback_is_measurement_only_not_qpos_teleport(self):
        with tempfile.TemporaryDirectory() as temporary:
            baseline_path = Path(temporary) / "baseline.jsonl"
            perturbed_path = Path(temporary) / "perturbed.jsonl"
            self._fixture_trace(baseline_path)
            self._fixture_trace(perturbed_path, perturb_feedback=True)
            baseline = replay_trace(read_trace(baseline_path), model=self.model, mujoco_module=self.mujoco)
            perturbed = replay_trace(read_trace(perturbed_path), model=self.model, mujoco_module=self.mujoco)
            self.assertEqual(
                baseline["final_simulated_joint_position_rad"],
                perturbed["final_simulated_joint_position_rad"],
            )
            self.assertGreater(perturbed["joint_metrics"]["joint1"]["max_abs"], 0.04)

    def test_range_mismatch_is_rejected_without_clipping(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "range.jsonl"
            high = float(self.model.jnt_range[int(self.model.joint("joint1").id), 1]) + 1.0
            with TraceWriter(path) as writer:
                writer.record(
                    command_joint_position_rad=[high, 0.0, 0.0, 0.0, 0.0, 0.0],
                    feedback_joint_position_rad=[0.0] * 6,
                    t_monotonic_ns=1_000_000_000,
                    feedback_source_timestamp_ns=1_000_000_000,
                )
            with self.assertRaises(MotionTraceRangeError) as raised:
                replay_trace(read_trace(path), model=self.model, mujoco_module=self.mujoco)
            self.assertIn("joint1", str(raised.exception))
            self.assertIn("outside model range", str(raised.exception))

    def test_cli_validate_and_replay_write_json_reports(self):
        from scripts.piper_motion_trace import main

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.jsonl"
            validation = Path(temporary) / "validation.json"
            replay = Path(temporary) / "replay.json"
            self._fixture_trace(path)
            self.assertEqual(main(["validate", str(path), "--output", str(validation)]), 0)
            self.assertEqual(main(["replay", str(path), "--output", str(replay)]), 0)
            self.assertEqual(json.loads(validation.read_text())["status"], "ok")
            replay_report = json.loads(replay.read_text())
            self.assertEqual(replay_report["status"], "ok")
            self.assertIn("joint_metrics", replay_report)

    def test_command_zoh_survives_irregular_async_event_timing(self):
        import numpy as np

        from scripts.piper_motion_trace import _integrate_duration

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "async.jsonl"
            model = self.model
            data = self.mujoco.MjData(model)
            data.qpos[:] = model.qpos0
            arm_qpos = [int(model.jnt_qposadr[int(model.joint(name).id)]) for name in JOINT_NAMES]
            arm_actuators = [int(model.actuator(name).id) for name in JOINT_NAMES]
            gripper_qpos = int(model.jnt_qposadr[int(model.joint("joint7").id)])
            command0 = [float(data.qpos[position]) for position in arm_qpos]
            command1 = list(command0)
            command1[0] = 0.1
            opening = float(data.qpos[gripper_qpos]) * 2.0
            data.ctrl[arm_actuators] = np.asarray(command0)
            data.ctrl[int(model.actuator("gripper").id)] = float(data.qpos[gripper_qpos])
            self.mujoco.mj_forward(model, data)
            t0 = 1_000_000_000
            t1 = t0 + 10_300_000
            t2 = t1 + 120_700_000
            with TraceWriter(path, metadata={"max_source_age_s": 0.001}) as writer:
                writer.record(
                    command_joint_position_rad=command0,
                    command_gripper_full_opening_m=opening,
                    feedback_joint_position_rad=[float(data.qpos[position]) for position in arm_qpos],
                    feedback_gripper_full_opening_m=opening,
                    t_monotonic_ns=t0,
                    feedback_source_timestamp_ns=t0,
                )
                _integrate_duration(
                    self.mujoco, model, data, 0.0103,
                    time_tolerance_s=1e-10, max_steps_per_interval=1000,
                )
                writer.record(
                    command_joint_position_rad=command1,
                    command_gripper_full_opening_m=opening,
                    command_valid=True,
                    feedback_joint_position_rad=None,
                    t_monotonic_ns=t1,
                )
                data.ctrl[arm_actuators] = np.asarray(command1)
                _integrate_duration(
                    self.mujoco, model, data, 0.1207,
                    time_tolerance_s=1e-10, max_steps_per_interval=1000,
                )
                writer.record(
                    command_joint_position_rad=command1,
                    command_gripper_full_opening_m=opening,
                    feedback_joint_position_rad=[float(data.qpos[position]) for position in arm_qpos],
                    feedback_gripper_full_opening_m=opening,
                    t_monotonic_ns=t2,
                    feedback_source_timestamp_ns=t2,
                )
            report = replay_trace(read_trace(path), model=self.model, mujoco_module=self.mujoco)
            self.assertEqual(report["feedback_samples_compared"], 1)
            self.assertEqual(report["fractional_intervals"], 2)
            self.assertEqual(report["replay_steps"], 67)
            self.assertLess(report["joint_metrics"]["joint1"]["max_abs"], 1e-9)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
