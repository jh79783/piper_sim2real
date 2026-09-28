"""Rewrite the Menagerie Piper MJCF into a form Isaac Sim's MJCF importer maps 1:1.

The Menagerie model relies on three MuJoCo-only features:

* collision capsules fitted to meshes at compile time (``type="capsule"`` plus
  ``mesh=...``),
* ``gravcomp="1"`` gravity compensation on every body, and
* an equality constraint that mirrors ``joint8 = -joint7``.

PhysX has no mesh-fitted capsule and no joint equality.  This module compiles
the untouched Menagerie file with MuJoCo once, bakes each fitted capsule into
explicit ``size``/``pos``/``quat`` attributes, and removes the equality,
actuators, gravcomp, lights and keyframes.  The Isaac Lab articulation config
replaces those with ``disable_gravity=True`` (full gravity compensation), two
independent finger drives commanded to ``+g``/``-g``, and implicit PD drives
carrying the Menagerie ``kp``/``kv``/force limits.

Only MuJoCo's compiler is used here; no MuJoCo physics is stepped.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

from isaac_piper.paths import MENAGERIE_PIPER_DIR, MENAGERIE_PIPER_XML, PIPER_ISAAC_MJCF

_REMOVED_SECTIONS = ("keyframe", "equality", "actuator", "contact", "option")


def _fmt(values) -> str:
    return " ".join(f"{float(v):.9g}" for v in values)


def build_isaac_mjcf(
    source: Path = MENAGERIE_PIPER_XML,
    destination: Path = PIPER_ISAAC_MJCF,
) -> Path:
    """Write the importer-friendly MJCF and return its path."""

    import mujoco

    source = Path(source)
    destination = Path(destination)
    root = ET.parse(source).getroot()
    # The Isaac importer joins meshdir onto its output directory, so it needs a
    # relative meshdir.  Copy the (32 MB) Menagerie meshes next to the output
    # instead of pointing back into the read-only submodule.
    mesh_dir = destination.parent / "meshes"
    if mesh_dir.exists():
        shutil.rmtree(mesh_dir)
    shutil.copytree(source.parent / "assets", mesh_dir)
    compiler = root.find("compiler")
    compiler.set("meshdir", str(mesh_dir.resolve()))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    compiler.set("meshdir", mesh_dir.name)
    capsule = int(mujoco.mjtGeom.mjGEOM_CAPSULE)

    baked = 0
    for body in root.iter("body"):
        body_id = model.body(body.get("name")).id
        first = int(model.body_geomadr[body_id])
        count = int(model.body_geomnum[body_id])
        # Compiled geoms of a body keep XML order, so the n-th XML geom is the
        # n-th compiled geom of that body.
        xml_geoms = body.findall("geom")
        if len(xml_geoms) != count:
            raise RuntimeError(f"geom count mismatch for body {body.get('name')}")
        for offset, geom in enumerate(xml_geoms):
            if geom.get("class") != "collision" or geom.get("mesh") is None:
                continue
            geom_id = first + offset
            if int(model.geom_type[geom_id]) != capsule:
                raise RuntimeError(f"unexpected fitted collision type in {body.get('name')}")
            radius, half_length = model.geom_size[geom_id][:2]
            del geom.attrib["mesh"]
            geom.attrib.pop("quat", None)
            geom.set("type", "capsule")
            geom.set("size", _fmt((radius, half_length)))
            geom.set("pos", _fmt(model.geom_pos[geom_id]))
            geom.set("quat", _fmt(model.geom_quat[geom_id]))
            baked += 1
        body.attrib.pop("gravcomp", None)
    if baked == 0:
        raise RuntimeError("no mesh-fitted collision capsules found; Menagerie layout changed?")

    worldbody = root.find("worldbody")
    for light in worldbody.findall("light"):
        worldbody.remove(light)
    for tag in _REMOVED_SECTIONS:
        for section in root.findall(tag):
            root.remove(section)
    # Actuator defaults are meaningless without actuators and some importer
    # versions warn on orphan <position> defaults.
    for default in root.iter("default"):
        for position in default.findall("position"):
            default.remove(position)

    ET.indent(root)
    destination.write_text(ET.tostring(root, encoding="unicode") + "\n")
    # Validate the rewritten file still compiles with identical kinematics.
    rewritten = mujoco.MjModel.from_xml_path(str(destination))
    if rewritten.njnt != model.njnt or rewritten.nbody != model.nbody:
        raise RuntimeError("rewritten MJCF changed the kinematic tree")
    return destination


if __name__ == "__main__":
    print(build_isaac_mjcf())
    print("mesh source:", MENAGERIE_PIPER_DIR)
