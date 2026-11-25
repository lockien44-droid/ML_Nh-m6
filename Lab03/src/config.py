"""Paths and defaults shared by the two Lab03 exercises."""
from pathlib import Path
import yaml

LAB_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = LAB_DIR / "config.yaml"
DEFAULT_SEED = 42
DEFAULT_K_VALUES = list(range(2, 11))
DEFAULT_EPS_VALUES = [0.3, 0.5, 0.7]
DEFAULT_IMAGE_SIZE = 192


def load_config(path=None):
    settings = yaml.safe_load(Path(path or CONFIG_PATH).read_text(encoding="utf-8"))
    clustering = settings["customers"]["kmeans"]
    clustering.setdefault("random_state", DEFAULT_SEED)
    clustering.setdefault("k_values", DEFAULT_K_VALUES.copy())
    clustering.setdefault("algorithms", ["kmeans"])
    clustering.setdefault("eps_values", DEFAULT_EPS_VALUES.copy())
    clustering.setdefault("min_samples_values", [5])
    clustering.setdefault("linkages", ["ward"])
    clustering.setdefault("min_coverage", 0.8)
    image = settings["image"]
    image.setdefault("random_state", DEFAULT_SEED)
    image.setdefault("max_size", DEFAULT_IMAGE_SIZE)
    return settings
