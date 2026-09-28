"""Build the Piper USD used by the Isaac Sim environment.

Run inside the piper-isaac image (``bash scripts/run_isaac.sh convert``).  The
pipeline is: Menagerie MJCF -> importer-friendly MJCF (isaac_piper.mjcf) ->
Isaac Sim MJCF importer -> ``.cache/isaac/assets/piper/piper.usd``.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--force", action="store_true", help="Rebuild even if the USD exists.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
simulation_app = AppLauncher(args).app

# The headless Isaac Lab experience omits the MJCF importer extension.
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("isaacsim.asset.importer.mjcf")

from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg  # noqa: E402
from pxr import PhysxSchema, Usd, UsdPhysics  # noqa: E402

from isaac_piper.mjcf import build_isaac_mjcf  # noqa: E402
from isaac_piper.paths import PIPER_USD  # noqa: E402


def main():
    mjcf = build_isaac_mjcf()
    print(f"[piper] importer MJCF: {mjcf}")
    converter = MjcfConverter(
        MjcfConverterCfg(
            asset_path=str(mjcf),
            usd_dir=str(PIPER_USD.parent),
            usd_file_name=PIPER_USD.name,
            fix_base=True,
            import_sites=False,
            self_collision=False,
            make_instanceable=True,
            force_usd_conversion=args.force or not PIPER_USD.exists(),
        )
    )
    print(f"[piper] USD: {converter.usd_path}")

    stage = Usd.Stage.Open(converter.usd_path)
    # The importer marks both the empty worldBody Xform and the base_link body
    # as articulation roots; Isaac Lab requires exactly one.  Keep base_link,
    # which owns the fixed root joint.
    for prim in stage.Traverse():
        if prim.GetName() == "worldBody" and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
                prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
            print(f"[piper] removed duplicate articulation root at {prim.GetPath()}")
    stage.GetRootLayer().Save()
    roots = [p.GetPath() for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if len(roots) != 1:
        raise RuntimeError(f"expected one articulation root, found {roots}")
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint) or prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(
            UsdPhysics.ArticulationRootAPI
        ):
            kinds = [s for s in prim.GetAppliedSchemas() if "Physics" in s]
            print(f"  {prim.GetPath()} [{prim.GetTypeName()}] {kinds}")


if __name__ == "__main__":
    main()
    simulation_app.close()
