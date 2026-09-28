"""Project paths shared by the Isaac Sim asset builder, env and launcher."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MENAGERIE_PIPER_DIR = PROJECT_ROOT / "third_party/mujoco_menagerie/agilex_piper"
MENAGERIE_PIPER_XML = MENAGERIE_PIPER_DIR / "piper.xml"

# Generated assets live under the git-ignored .cache so a fresh checkout
# rebuilds them from the pinned Menagerie submodule instead of storing USD.
ISAAC_ASSET_DIR = PROJECT_ROOT / ".cache/isaac/assets/piper"
PIPER_ISAAC_MJCF = ISAAC_ASSET_DIR / "piper_isaac.xml"
PIPER_USD = ISAAC_ASSET_DIR / "piper.usd"
