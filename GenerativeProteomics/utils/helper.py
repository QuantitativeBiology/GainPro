import yaml
from pathlib import Path

def read_config(
    cfg_path: Path
) -> dict:
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML configuration file '{cfg_path}': {e}") from e
