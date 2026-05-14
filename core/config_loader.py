import yaml
from pathlib import Path

_config = None

def load_config(path: str = "config.yaml") -> dict:
    """
    Load and cache the YAML config. Called once at startup;
    subsequent calls return the cached dict.
    """
    global _config
    if _config is None:
        cfg_path = Path(path)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path.resolve()}")
        with open(cfg_path, "r") as f:
            _config = yaml.safe_load(f)
    return _config