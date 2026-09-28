"""The importer MJCF keeps the Menagerie tree and removes MuJoCo-only features."""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from isaac_piper.mjcf import build_isaac_mjcf
from isaac_piper.paths import MENAGERIE_PIPER_XML


def test_rewrite_bakes_capsules_and_preserves_tree(tmp_path):
    path = build_isaac_mjcf(destination=tmp_path / "piper_isaac.xml")
    root = ET.parse(path).getroot()
    assert all(g.get("mesh") is None for g in root.iter("geom") if g.get("class") == "collision")
    assert all(b.get("gravcomp") is None for b in root.iter("body"))
    for tag in ("actuator", "equality", "keyframe"):
        assert root.find(tag) is None

    original = mujoco.MjModel.from_xml_path(str(MENAGERIE_PIPER_XML))
    rewritten = mujoco.MjModel.from_xml_path(str(path))
    np.testing.assert_allclose(rewritten.body_pos, original.body_pos)
    np.testing.assert_allclose(rewritten.body_quat, original.body_quat)
    np.testing.assert_allclose(rewritten.body_mass, original.body_mass)
    np.testing.assert_allclose(rewritten.jnt_range, original.jnt_range)
    collision = original.geom_group == 3
    np.testing.assert_allclose(rewritten.geom_size[collision], original.geom_size[collision], atol=1e-8)
    np.testing.assert_allclose(rewritten.geom_pos[collision], original.geom_pos[collision], atol=1e-8)
