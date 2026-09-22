"""Kinematic and metadata checks for the provisional D455 wrist camera."""

from __future__ import annotations

import math
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from scripts import piper_camera as camera


class PiperCameraGeometryTests(unittest.TestCase):
    def test_menagerie_link6_basis_correction_recovers_upstream_pose(self):
        # Menagerie's body frame is URDF link6 rotated -90 degrees about local
        # z; its geometry then carries +90 degrees.  Recompose that mapping
        # independently from the helper under test.
        angle = math.pi / 2.0
        correction = np.array(
            ((math.cos(angle), -math.sin(angle), 0.0),
             (math.sin(angle), math.cos(angle), 0.0),
             (0.0, 0.0, 1.0))
        )
        body_pos, body_rotation = camera.menagerie_mount_pose()
        expected_urdf_pos = np.array((-0.029, 0.065, 0.022))
        expected_urdf_rotation = camera.rpy_to_matrix((0.0, -1.22, -1.57))
        np.testing.assert_allclose(correction.T @ body_pos, expected_urdf_pos, atol=1e-9)
        np.testing.assert_allclose(correction.T @ body_rotation, expected_urdf_rotation, atol=1e-9)

    def test_d455_rgb_optical_transform_and_forward_axis(self):
        position, optical_rotation = camera.d455_rgb_optical_pose_from_mount()
        np.testing.assert_allclose(position, [0.01115, -0.0115, 0.0145], atol=1e-9)

        # Official ROS optical conversion: camera-link -> ROS optical is
        # rpy(-pi/2, 0, -pi/2), then ROS optical -> MuJoCo camera is Rx(pi).
        expected_optical = np.array(
            ((0.0, 0.0, 1.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0))
        )
        ros_to_mujoco = np.diag([1.0, -1.0, -1.0])
        np.testing.assert_allclose(optical_rotation, expected_optical, atol=1e-9)
        _, camera_rotation = camera.d455_rgb_camera_pose_from_mount()
        np.testing.assert_allclose(camera_rotation, expected_optical @ ros_to_mujoco, atol=1e-9)

        # MuJoCo looks along local -Z.  The resulting forward ray is +X in
        # camera_link, which is the ROS optical +Z direction after conversion.
        np.testing.assert_allclose(camera_rotation @ [0.0, 0.0, -1.0], [1.0, 0.0, 0.0], atol=1e-9)

    def test_generated_camera_has_rigid_policy_camera_and_zero_mass_geometry(self):
        root = camera.make_d455_wrist_body()
        self.assertEqual(root.attrib["name"], "d455_mount")
        optical = root.find("./body[@name='d455_camera_link']/body[@name='d455_color_optical_frame']")
        self.assertIsNotNone(optical)
        policy = optical.find("camera[@name='policy_rgb']")
        self.assertIsNotNone(policy)
        self.assertEqual(policy.attrib["fovy"], "65")
        for geom in root.findall(".//geom"):
            self.assertEqual(geom.attrib["contype"], "0")
            self.assertEqual(geom.attrib["conaffinity"], "0")
            self.assertEqual(geom.attrib["density"], "0")

    def test_camera_metadata_carries_capture_and_checkpoint_contract(self):
        metadata = camera.camera_metadata()
        self.assertEqual(metadata["model"], "Intel RealSense D455")
        self.assertEqual(metadata["attachment"], "rigid_body_link6")
        self.assertEqual(metadata["render_resolution"], [640, 480])
        self.assertEqual(metadata["encoder_image_size"], [128, 128])
        self.assertFalse(metadata["render_aspect_preserved"])
        self.assertEqual(metadata["rgb_optical_to_mujoco_quat_wxyz"], [0.0, 1.0, 0.0, 0.0])
        self.assertIn("9a11121700cb4780e273e34141f6402fe184321d", metadata["d455_source"])
        self.assertIn("8e1f88fdb7afca49c40e9a0c1c01cc588e86f0d2", metadata["mount_source"])


try:
    import mujoco  # type: ignore
except ImportError:  # pragma: no cover - host regression environment may omit MuJoCo
    mujoco = None


@unittest.skipUnless(mujoco is not None, "MuJoCo is installed in the simulation/Docker image")
class PiperCameraModelTests(unittest.TestCase):
    def test_model_camera_is_nested_under_link6(self):
        from scripts.piper_pick_place_env import build_model

        model = build_model()
        link6 = model.body("link6").id
        camera_id = model.camera("policy_rgb").id
        d455_mount = model.body("d455_mount").id
        d455_link = model.body("d455_camera_link").id
        optical = model.body("d455_color_optical_frame").id
        self.assertAlmostEqual(float(model.body_mass[d455_mount]), 0.0, places=9)
        self.assertAlmostEqual(float(model.body_mass[d455_link]), 0.0, places=9)
        self.assertEqual(int(model.body_parentid[d455_mount]), link6)
        self.assertEqual(int(model.cam_bodyid[camera_id]), optical)
        self.assertEqual(int(model.body_parentid[d455_link]), d455_mount)
        self.assertEqual(int(model.body_parentid[optical]), d455_link)
        self.assertAlmostEqual(float(model.cam_fovy[camera_id]), 65.0, places=5)

    def test_compiled_camera_pose_follows_joint6_with_fixed_wrist_transform(self):
        from scripts.piper_pick_place_env import build_model

        model = build_model()
        data = mujoco.MjData(model)
        camera_id = model.camera("policy_rgb").id
        link6 = model.body("link6").id
        joint6 = model.joint("joint6").qposadr[0]

        correction_angle = math.pi / 2.0
        correction = np.array(
            ((math.cos(correction_angle), -math.sin(correction_angle), 0.0),
             (math.sin(correction_angle), math.cos(correction_angle), 0.0),
             (0.0, 0.0, 1.0))
        )
        mount_pos = correction @ np.array([-0.029, 0.065, 0.022])
        mount_rotation = correction @ camera.rpy_to_matrix((0.0, -1.22, -1.57))
        camera_link_to_rgb = np.array([0.01115, -0.0115, 0.0145])
        ros_optical = np.array(
            ((0.0, 0.0, 1.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0))
        )
        optical_to_mujoco = np.diag([1.0, -1.0, -1.0])
        expected_relative_position = mount_pos + mount_rotation @ camera_link_to_rgb
        expected_relative_rotation = mount_rotation @ (ros_optical @ optical_to_mujoco)

        poses = []
        for angle in (0.0, 0.8):
            data.qpos[joint6] = angle
            mujoco.mj_forward(model, data)
            link_rotation = data.xmat[link6].reshape(3, 3)
            relative_position = link_rotation.T @ (data.cam_xpos[camera_id] - data.xpos[link6])
            relative_rotation = link_rotation.T @ data.cam_xmat[camera_id].reshape(3, 3)
            np.testing.assert_allclose(relative_position, expected_relative_position, atol=2e-6)
            np.testing.assert_allclose(relative_rotation, expected_relative_rotation, atol=2e-6)
            np.testing.assert_allclose(
                relative_rotation @ np.array([0.0, 0.0, -1.0]),
                mount_rotation @ np.array([1.0, 0.0, 0.0]),
                atol=2e-6,
            )
            poses.append(data.cam_xpos[camera_id].copy())
        self.assertGreater(float(np.linalg.norm(poses[1] - poses[0])), 1e-3)


if __name__ == "__main__":
    unittest.main()
