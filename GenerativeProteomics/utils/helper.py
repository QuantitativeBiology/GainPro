import yaml
from pathlib import Path
from datetime import datetime

from utils.configs.benchmark_config import BenchmarkConfig
from utils.configs.model_config import (
    GainConfig, 
    AutoEncoderConfig, 
    MissForestConfig,
    MeanConfig,
)
from utils.configs.training_config import GainTrainingConfig, AutoEncoderTrainingConfig

MODEL_REGISTRY = {
    "gain": {
        "model_config": GainConfig,
        "training_config": GainTrainingConfig,
    },
    "autoencoder": {
        "model_config": AutoEncoderConfig,
        "training_config": AutoEncoderTrainingConfig,
    },
    "missforest": {
        "model_config": MissForestConfig,
    },
    "global_mean": {
        "model_config": MeanConfig,
    },
    "tissue_mean": {
        "model_config": MeanConfig,
    },
}

def load_yaml(cfg_path: Path):
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        try:
            return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML configuration file '{cfg_path}': {e}") from e

def load_benchmark(cfg_path: Path) -> BenchmarkConfig:
    raw = load_yaml(cfg_path)
    cfg = BenchmarkConfig.model_validate(raw)

    # Validate configuration files
    for m in cfg.models:
        registry_entry = MODEL_REGISTRY[m.name.lower()]
        registry_entry["model_config"].model_validate(load_yaml(m.model_cfg_path))
        if m.name == "gain":
            GainTrainingConfig.model_validate(load_yaml(m.training_cfg_path))
        if m.name == "autoencoder":
            AutoEncoderTrainingConfig.model_validate(load_yaml(m.training_cfg_path))
    return cfg

def make_run_dir(
    parent: Path, 
    run: int, 
    seed: int,
) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    run_dir = parent / f"run_{run:03d}_seed_{seed}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir