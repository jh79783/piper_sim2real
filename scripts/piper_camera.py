"""D455 wrist-camera geometry shared by the MuJoCo model and RGB contract.

The camera dimensions and nominal frame offsets come from the pinned official
Intel RealSense D455 xacro mirrored in AgileX's ``piper_isaac_sim`` repository.
AgileX's Piper bracket pose is available for a D435 and is used here as an
explicitly provisional D455 mount assumption; it is not a claim that the D435
bracket fits the longer D455 body.

The Menagerie Piper model preserves the upstream link6 visual orientation by
rotating its link6 body frame -90 degrees about the link6 z axis and applying
a compensating +90 degree local rotation to the link6 geometry.  Camera
poses authored in the upstream URDF link6 frame therefore need the inverse
rotation before they are attached to the Menagerie link6 body.
"""

from __future__ import annotations

from math import cos, pi, sin, sqrt
import xml.etree.ElementTree as ET

import numpy as np


# Pinned source references used for the camera contract.  The mount pose is
# the D435 bracket pose in the same AgileX file; the D455 body and optical
# offsets below come from the D455 xacro itself.
AGILEX_REPOSITORY = "agilexrobotics/piper_isaac_sim"
AGILEX_COMMIT = "8e1f88fdb7afca49c40e9a0c1c01cc588e86f0d2"
AGILEX_MOUNT_SOURCE = (
    "https://github.com/agilexrobotics/piper_isaac_sim/blob/"
    f"{AGILEX_COMMIT}/piper_description/urdf/"
    "piper_description_v100_realsense_camera.urdf#L367-L373"
)
D455_SOURCE = (
    "https://github.com/realsenseai/realsense-ros/blob/"
    "9a11121700cb4780e273e34141f6402fe184321d/"
    "realsense2_description/urdf/_d455.urdf.xacro#L29-L159"
)
D455_MOUNT_SOURCE = AGILEX_MOUNT_SOURCE
D455_FOV_SOURCE = "https://www.realsenseai.com/depth-camera-d455/"
D455_ISP_SOURCE = (
    "https://github.com/realsenseai/librealsense/blob/"
    "e15c5d6bb1563e778d116f682aeefffbae2daedc/src/ds/d400/d400-private.cpp#L304-L376"
)

CAMERA_MODEL = "Intel RealSense D455"
CAMERA_MOUNT_ASSUMPTION = "AgileX D435 wrist bracket pose reused provisionally for D455"

# AgileX's D435 bracket pose, expressed in the upstream Piper URDF link6
# frame.  The source labels this child d435_camera_link; this project uses the
# same rigid reference for the D455 body until a D455-specific bracket survey
# is available.
D455_MOUNT_POS_URDF = np.array((-0.029, 0.065, 0.022), dtype=np.float64)
D455_MOUNT_RPY_URDF = np.array((0.0, -1.22, -1.57), dtype=np.float64)

# Intel's D455 xacro values.  The xacro defines the bottom screw frame as the
# parent of camera_link and uses these nominal camera-link and color offsets.
D455_BODY_SIZE = np.array((0.026, 0.124, 0.029), dtype=np.float64)
D455_MOUNT_TO_CAMERA_LINK = np.array((0.01115, 0.0475, 0.0145), dtype=np.float64)
D455_COLOR_OFFSET = np.array((0.0, -0.059, 0.0), dtype=np.float64)
D455_COLOR_OPTICAL_RPY = np.array((-pi / 2.0, 0.0, -pi / 2.0), dtype=np.float64)
D455_RGB_FOV_DEGREES = {"horizontal": 90.0, "vertical": 65.0}
D455_NATIVE_RESOLUTION = (1280, 800)

# MuJoCo camera coordinates use +X right, +Y up, -Z forward.  ROS optical
# frames use +X right, +Y down, +Z forward, hence this fixed 180-degree X
# rotation when placing a MuJoCo camera at the ROS RGB optical frame.
ROS_OPTICAL_TO_MUJOCO = np.diag((1.0, -1.0, -1.0))

# The RGB capture contract is 640x480 at the existing 30 Hz camera rate.  This
# is an explicit 4:3 simulation pinhole approximation of the D455's native
# 1280x800 source, rather than a claim of calibrated 640x480 hardware intrinsics.
# The encoder still receives its existing 128x128 center crop, so this does not
# change the learned feature tensor shape or the frozen encoder checkpoint.
POLICY_RENDER_SIZE = (640, 480)
POLICY_IMAGE_SIZE = (128, 128)


def rpy_to_matrix(rpy) -> np.ndarray:
    """Convert URDF roll-pitch-yaw into a right-handed rotation matrix."""

    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = cos(roll), sin(roll)
    cp, sp = cos(pitch), sin(pitch)
    cy, sy = cos(yaw), sin(yaw)
    return np.array(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        ),
        dtype=np.float64,
    )


D455_COLOR_OPTICAL_ROTATION = rpy_to_matrix(D455_COLOR_OPTICAL_RPY)


def _rz(angle: float) -> np.ndarray:
    c, s = cos(angle), sin(angle)
    return np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)), dtype=np.float64)


def matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Return a MuJoCo ``quat`` in w,x,y,z order from a rotation matrix."""

    matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * sqrt(trace + 1.0)
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = 2.0 * sqrt(max(1e-16, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]))
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
        elif index == 1:
            scale = 2.0 * sqrt(max(1e-16, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]))
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = 2.0 * sqrt(max(1e-16, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]))
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
    quaternion = np.array((w, x, y, z), dtype=np.float64)
    return quaternion / np.linalg.norm(quaternion)


def upstream_link6_to_menagerie_body(pos, rotation) -> tuple[np.ndarray, np.ndarray]:
    """Map an upstream URDF link6 pose into the Menagerie link6 body frame.

    Menagerie's link6 body frame is upstream-link6 multiplied by Rz(-pi/2),
    while its link6 geoms carry the compensating Rz(+pi/2) local quaternion.
    """

    correction = _rz(pi / 2.0)
    position = correction @ np.asarray(pos, dtype=np.float64).reshape(3)
    matrix = correction @ np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    return position, matrix


def menagerie_mount_pose() -> tuple[np.ndarray, np.ndarray]:
    """Return the provisional D455 mount pose in Menagerie link6 coordinates."""

    return upstream_link6_to_menagerie_body(
        D455_MOUNT_POS_URDF,
        rpy_to_matrix(D455_MOUNT_RPY_URDF),
    )


def d455_rgb_optical_pose_from_mount() -> tuple[np.ndarray, np.ndarray]:
    """Return the nominal mount-to-RGB optical-frame transform.

    The returned rotation is the ROS camera-link to ROS optical-frame
    rotation.  MuJoCo's camera element applies :data:`ROS_OPTICAL_TO_MUJOCO`
    as a separate local rotation.
    """

    position = D455_MOUNT_TO_CAMERA_LINK + D455_COLOR_OFFSET
    return position, D455_COLOR_OPTICAL_ROTATION.copy()


def d455_rgb_camera_pose_from_mount() -> tuple[np.ndarray, np.ndarray]:
    """Return the mount-to-MuJoCo-camera transform used by ``policy_rgb``."""

    position, optical_rotation = d455_rgb_optical_pose_from_mount()
    return position, optical_rotation @ ROS_OPTICAL_TO_MUJOCO


def _fmt(values) -> str:
    return " ".join(f"{float(value):.9g}" for value in np.asarray(values).reshape(-1))


def make_d455_wrist_body() -> ET.Element:
    """Build a visual-only rigid D455 body with the wrist policy camera."""

    mount_position, mount_rotation = menagerie_mount_pose()
    mount_quat = matrix_to_quat_wxyz(mount_rotation)
    mount = ET.Element(
        "body",
        {
            "name": "d455_mount",
            "pos": _fmt(mount_position),
            "quat": _fmt(mount_quat),
        },
    )
    # This is a lightweight visual marker for the provisional bracket datum;
    # it is intentionally non-colliding and is not a D435/D455 fit claim.
    ET.SubElement(
        mount,
        "geom",
        {
            "name": "d455_mount_assumption",
            "type": "box",
            "group": "2",
            "contype": "0",
            "conaffinity": "0",
            "size": "0.025 0.035 0.003",
            "density": "0",
            "rgba": "0.10 0.10 0.12 1",
        },
    )
    camera_link = ET.SubElement(
        mount,
        "body",
        {
            "name": "d455_camera_link",
            "pos": _fmt(D455_MOUNT_TO_CAMERA_LINK),
            "quat": "1 0 0 0",
        },
    )
    # The official D455 collision box is used as a visual-only body.  The
    # exact STL is deliberately not copied into this repository; the geometry
    # dimensions and source URL are retained in the metadata/documentation.
    half_size = D455_BODY_SIZE / 2.0
    ET.SubElement(
        camera_link,
        "geom",
        {
            "name": "d455_camera_body",
            "type": "box",
            "group": "2",
            "contype": "0",
            "conaffinity": "0",
            "size": _fmt(half_size),
            "pos": "-0.00845 -0.0475 0",
            "density": "0",
            "rgba": "0.16 0.17 0.18 1",
        },
    )
    ET.SubElement(
        camera_link,
        "geom",
        {
            "name": "d455_front_glass",
            "type": "box",
            "group": "2",
            "contype": "0",
            "conaffinity": "0",
            "size": "0.001 0.052 0.010",
            "pos": "0.005 -0.0475 0",
            "density": "0",
            "rgba": "0.03 0.10 0.14 1",
        },
    )
    optical = ET.SubElement(
        camera_link,
        "body",
        {
            "name": "d455_color_optical_frame",
            "pos": _fmt(D455_COLOR_OFFSET),
            "quat": _fmt(matrix_to_quat_wxyz(D455_COLOR_OPTICAL_ROTATION)),
        },
    )
    ET.SubElement(
        optical,
        "camera",
        {
            "name": "policy_rgb",
            "mode": "fixed",
            # ROS optical +X/right,+Y/down,+Z/forward is converted to
            # MuJoCo +X/right,+Y/up,-Z/forward at the camera element.
            "quat": _fmt(matrix_to_quat_wxyz(ROS_OPTICAL_TO_MUJOCO)),
            "fovy": f"{D455_RGB_FOV_DEGREES['vertical']:.9g}",
        },
    )
    return mount


def camera_metadata() -> dict:
    """Return JSON-safe camera geometry/intrinsics for RGB checkpoints."""

    mount_position, mount_rotation = menagerie_mount_pose()
    optical_position, optical_rotation = d455_rgb_optical_pose_from_mount()
    camera_position, camera_rotation = d455_rgb_camera_pose_from_mount()
    return {
        "model": CAMERA_MODEL,
        "camera_name": "policy_rgb",
        "attachment": "rigid_body_link6",
        "mount_assumption": CAMERA_MOUNT_ASSUMPTION,
        "mount_source": AGILEX_MOUNT_SOURCE,
        "d455_source": D455_SOURCE,
        "fov_source": D455_FOV_SOURCE,
        "isp_crop_source": D455_ISP_SOURCE,
        "mount_frame": "menagerie_link6_body",
        "mount_pos_m": [float(value) for value in mount_position],
        "mount_quat_wxyz": [float(value) for value in matrix_to_quat_wxyz(mount_rotation)],
        "mount_pos_urdf_m": [float(value) for value in D455_MOUNT_POS_URDF],
        "mount_rpy_urdf_rad": [float(value) for value in D455_MOUNT_RPY_URDF],
        "menagerie_link6_basis_correction_rpy_rad": [0.0, 0.0, pi / 2.0],
        "camera_link_offset_m": [float(value) for value in D455_MOUNT_TO_CAMERA_LINK],
        "color_frame_offset_m": [float(value) for value in D455_COLOR_OFFSET],
        "rgb_optical_frame_rpy_rad": [float(value) for value in D455_COLOR_OPTICAL_RPY],
        "rgb_optical_axis_conversion": "ROS optical (+X right,+Y down,+Z forward) -> MuJoCo (+X right,+Y up,-Z forward)",
        "rgb_optical_to_mujoco_quat_wxyz": [
            float(value) for value in matrix_to_quat_wxyz(ROS_OPTICAL_TO_MUJOCO)
        ],
        "rgb_optical_pos_from_mount_m": [float(value) for value in optical_position],
        "rgb_optical_frame_quat_wxyz": [float(value) for value in matrix_to_quat_wxyz(optical_rotation)],
        "rgb_camera_pos_from_mount_m": [float(value) for value in camera_position],
        "rgb_camera_quat_from_mount_wxyz": [float(value) for value in matrix_to_quat_wxyz(camera_rotation)],
        "rgb_camera_forward_in_camera_link": [1.0, 0.0, 0.0],
        "native_fov_deg": dict(D455_RGB_FOV_DEGREES),
        "native_resolution": list(D455_NATIVE_RESOLUTION),
        "render_resolution": list(POLICY_RENDER_SIZE),
        "encoder_image_size": list(POLICY_IMAGE_SIZE),
        "source_aspect": D455_NATIVE_RESOLUTION[0] / D455_NATIVE_RESOLUTION[1],
        "render_aspect": POLICY_RENDER_SIZE[0] / POLICY_RENDER_SIZE[1],
        "render_aspect_preserved": False,
        "projection_assumption": "640x480 4:3 simulation pinhole approximation of native 1280x800 D455 RGB",
        "preprocessing": "center_crop_shorter_side_then_resize_to_encoder_image_size",
    }


__all__ = [
    "AGILEX_COMMIT",
    "AGILEX_MOUNT_SOURCE",
    "CAMERA_MOUNT_ASSUMPTION",
    "CAMERA_MODEL",
    "D455_BODY_SIZE",
    "D455_COLOR_OFFSET",
    "D455_FOV_SOURCE",
    "D455_ISP_SOURCE",
    "D455_MOUNT_RPY_URDF",
    "D455_MOUNT_SOURCE",
    "D455_MOUNT_TO_CAMERA_LINK",
    "D455_MOUNT_POS_URDF",
    "D455_NATIVE_RESOLUTION",
    "D455_RGB_FOV_DEGREES",
    "D455_SOURCE",
    "POLICY_IMAGE_SIZE",
    "POLICY_RENDER_SIZE",
    "ROS_OPTICAL_TO_MUJOCO",
    "camera_metadata",
    "d455_rgb_camera_pose_from_mount",
    "d455_rgb_optical_pose_from_mount",
    "make_d455_wrist_body",
    "matrix_to_quat_wxyz",
    "menagerie_mount_pose",
    "rpy_to_matrix",
    "upstream_link6_to_menagerie_body",
]
