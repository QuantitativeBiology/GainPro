import yaml
import logging
from pathlib import Path

from models.GainPro.gain import Gain
from models.AutoEncoder.autoencoder import AutoEncoder
from utils.configs.benchmark_config import BenchmarkConfig
from utils.configs.model_config import GainConfig, AutoEncoderConfig
from utils.configs.training_config import GainTrainingConfig, AutoEncoderTrainingConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "protogain": {
        "model": Gain,
        "model_config": GainConfig,
        "training_config": GainTrainingConfig,
    },
    "autoencoder": {
        "model": AutoEncoder,
        "model_config": AutoEncoderConfig,
        "training_config": AutoEncoderTrainingConfig,
    },
}

def load_yaml(cfg_path: Path):
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML configuration file '{cfg_path}': {e}") from e

def load_benchmark(cfg_path: Path):
    raw = load_yaml(cfg_path)
    cfg = BenchmarkConfig.model_validate(raw)

    resolved_models = []
    for m in cfg.models:
        registry_entry = MODEL_REGISTRY[m.name]

        model_raw = load_yaml(m.model_config_path)
        model_cfg = registry_entry["model_config"].model_validate(model_raw)

        if m.name in ("protogain", "autoencoder"):
            training_raw = load_yaml(m.training_config_path)

        resolved_models.append({
            "name": m.name,
            "model_cfg": model_cfg,
            "training_cfg": training_raw,
            "model_class": registry_entry["model"],
        })
    return cfg, resolved_models
