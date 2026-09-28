"""Batched Piper forward kinematics and reset-time IK in plain torch.

The chain is parsed from the pinned Menagerie MJCF, so it is the same
kinematic tree that the USD was converted from.  Positions are expressed in the
robot base frame (``base_link``), which the environment places at each env
origin with identity orientation.

The IK mirrors the damped-least-squares solver of the MuJoCo implementation
(``PiperPickPlaceEnv.solve_ik``): TCP position plus a weighted orientation
error towards a fixed, slightly tilted downward approach.  It runs on the GPU
for every resetting environment at once and never touches the simulator.
"""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import torch

from isaac_piper.paths import MENAGERIE_PIPER_XML

ARM_BODIES = ("link1", "link2", "link3", "link4", "link5", "link6")
TCP_OFFSET_LINK6 = (0.0, 0.0, 0.12)
HOME_JOINTS = (0.0, 1.57, -1.3485, 0.0, 0.0, 0.0)
# Rotation about +y by 2.8 rad: the downward approach used by the MuJoCo task.
APPROACH_ANGLE = 2.8


def _quat_wxyz_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = (quat / quat.norm()).tolist()
    return torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=torch.float64,
    )


def _floats(text: str | None, default: tuple[float, ...]) -> torch.Tensor:
    if text is None:
        return torch.tensor(default, dtype=torch.float64)
    return torch.tensor([float(v) for v in text.split()], dtype=torch.float64)


def approach_rotation(device=None, dtype=torch.float32) -> torch.Tensor:
    c, s = torch.cos(torch.tensor(APPROACH_ANGLE)), torch.sin(torch.tensor(APPROACH_ANGLE))
    return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], device=device, dtype=dtype)


class PiperKinematics:
    """Six-joint arm chain from ``base_link`` to the grasp TCP on ``link6``."""

    def __init__(self, mjcf: Path = MENAGERIE_PIPER_XML, device="cpu", dtype=torch.float32):
        root = ET.parse(mjcf).getroot()
        rotations, translations, axes, limits = [], [], [], []
        for name in ARM_BODIES:
            body = root.find(f".//body[@name='{name}']")
            if body is None:
                raise RuntimeError(f"Piper MJCF has no body {name}")
            joint = body.find("joint")
            if joint is None or joint.get("pos", "0 0 0").split() != ["0", "0", "0"]:
                raise RuntimeError(f"{name} must have one joint at the body origin")
            rotations.append(_quat_wxyz_to_matrix(_floats(body.get("quat"), (1, 0, 0, 0))))
            translations.append(_floats(body.get("pos"), (0, 0, 0)))
            axes.append(_floats(joint.get("axis"), (0, 0, 1)))
            limits.append(_floats(joint.get("range"), (0, 0)))
        self.device = torch.device(device)
        self.dtype = dtype
        self.rotations = torch.stack(rotations).to(self.device, dtype)
        self.translations = torch.stack(translations).to(self.device, dtype)
        self.axes = torch.stack(axes).to(self.device, dtype)
        limits = torch.stack(limits).to(self.device, dtype)
        self.joint_low, self.joint_high = limits[:, 0], limits[:, 1]
        self.tcp_offset = torch.tensor(TCP_OFFSET_LINK6, device=self.device, dtype=dtype)
        self.home = torch.tensor(HOME_JOINTS, device=self.device, dtype=dtype)
        self.target_rotation = approach_rotation(self.device, dtype)

    @staticmethod
    def _axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        """Rodrigues rotation for a fixed unit axis and a batch of angles."""
        x, y, z = axis
        k = torch.tensor([[0, -z, y], [z, 0, -x], [-y, x, 0]], device=angle.device, dtype=angle.dtype)
        eye = torch.eye(3, device=angle.device, dtype=angle.dtype)
        s, c = torch.sin(angle)[:, None, None], torch.cos(angle)[:, None, None]
        return eye + s * k + (1 - c) * (k @ k)

    def forward(self, joints: torch.Tensor):
        """Return TCP position, TCP rotation, joint origins and world joint axes."""
        count = joints.shape[0]
        rotation = torch.eye(3, device=joints.device, dtype=joints.dtype).expand(count, 3, 3)
        position = torch.zeros((count, 3), device=joints.device, dtype=joints.dtype)
        origins, world_axes = [], []
        for i in range(6):
            position = position + rotation @ self.translations[i]
            rotation = rotation @ self.rotations[i]
            origins.append(position)
            world_axes.append(rotation @ self.axes[i])
            rotation = rotation @ self._axis_angle(self.axes[i], joints[:, i])
        tcp = position + rotation @ self.tcp_offset
        return tcp, rotation, torch.stack(origins, 1), torch.stack(world_axes, 1)

    def tcp_position(self, joints: torch.Tensor) -> torch.Tensor:
        return self.forward(joints)[0]

    def solve_ik(self, target: torch.Tensor, initial: torch.Tensor | None = None, iterations: int = 150):
        """Batched DLS IK to ``target`` with the fixed approach orientation.

        Returns ``(joints, position_residual)``; callers decide how large a
        residual is acceptable, as in the MuJoCo reset path (5 mm).
        """
        count = target.shape[0]
        q = (self.home if initial is None else initial).expand(count, 6).clone()
        low, high = self.joint_low + 0.002, self.joint_high - 0.002
        damping = torch.eye(6, device=q.device, dtype=q.dtype) * 0.003**2
        for _ in range(iterations):
            tcp, rotation, origins, axes = self.forward(q)
            # 0.5 * sum_i r_i x t_i over rotation columns, as in the MuJoCo IK.
            rotation_error = 0.5 * torch.cross(rotation, self.target_rotation.expand_as(rotation), dim=1).sum(-1)
            error = torch.cat((target - tcp, 0.3 * rotation_error), dim=1)
            jac_p = torch.cross(axes, tcp[:, None, :] - origins, dim=2)
            jac = torch.cat((jac_p, 0.3 * axes), dim=2).transpose(1, 2)
            step = jac.transpose(1, 2) @ torch.linalg.solve(jac @ jac.transpose(1, 2) + damping, error[..., None])
            q = torch.clamp(q + step[..., 0].clamp(-0.1, 0.1), low, high)
        residual = torch.linalg.vector_norm(target - self.tcp_position(q), dim=1)
        return q, residual
