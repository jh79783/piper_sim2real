"""Torch Piper kinematics must match the MuJoCo compiler's kinematic tree."""

import mujoco
import numpy as np
import pytest
import torch

from isaac_piper.kinematics import PiperKinematics
from isaac_piper.paths import MENAGERIE_PIPER_XML


@pytest.fixture(scope="module")
def mj():
    model = mujoco.MjModel.from_xml_path(str(MENAGERIE_PIPER_XML))
    return model, mujoco.MjData(model)


def _mujoco_tcp(model, data, q):
    data.qpos[:] = 0
    data.qpos[:6] = q
    mujoco.mj_kinematics(model, data)
    link6 = model.body("link6").id
    return data.xpos[link6] + data.xmat[link6].reshape(3, 3) @ np.array([0, 0, 0.12]), data.xmat[
        link6
    ].reshape(3, 3)


def test_forward_matches_mujoco(mj):
    model, data = mj
    kin = PiperKinematics(dtype=torch.float64)
    rng = np.random.default_rng(0)
    q = rng.uniform(kin.joint_low.numpy(), kin.joint_high.numpy(), size=(64, 6))
    tcp, rotation, _, _ = kin.forward(torch.as_tensor(q))
    for row in range(q.shape[0]):
        expected_tcp, expected_rot = _mujoco_tcp(model, data, q[row])
        np.testing.assert_allclose(tcp[row].numpy(), expected_tcp, atol=1e-9)
        np.testing.assert_allclose(rotation[row].numpy(), expected_rot, atol=1e-9)


def test_joint_limits_match_mjcf(mj):
    model, _ = mj
    kin = PiperKinematics(dtype=torch.float64)
    ranges = model.jnt_range[:6]
    np.testing.assert_allclose(kin.joint_low.numpy(), ranges[:, 0])
    np.testing.assert_allclose(kin.joint_high.numpy(), ranges[:, 1])


def test_ik_reaches_reset_workspace():
    kin = PiperKinematics(dtype=torch.float32)
    # Above-cube starts over the full final layout plus the home TCP target.
    xs = torch.linspace(0.25, 0.43, 7)
    ys = torch.linspace(-0.18, 0.18, 7)
    grid = torch.cartesian_prod(xs, ys)
    targets = torch.cat((grid, torch.full((grid.shape[0], 1), 0.10)), dim=1)
    targets = torch.cat((targets, torch.tensor([[0.33, 0.0, 0.14]])), dim=0)
    joints, residual = kin.solve_ik(targets)
    assert residual.max().item() < 0.005
    assert torch.all(joints >= kin.joint_low) and torch.all(joints <= kin.joint_high)
