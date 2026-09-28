from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
PIPER_URDF = PROJECT_DIR / "assets" / "piper_description" / "urdf" / "piper_description.urdf"
# The URDF importer writes its USD here and reuses it until the URDF or the
# converter config changes.
PIPER_USD_DIR = PROJECT_DIR / ".cache" / "assets" / "piper"
