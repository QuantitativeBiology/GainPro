import os
import yaml
import errno
from pathlib import Path
from datetime import datetime

from utils.paths import get_project_root
from imputation_manager import ImputationManager
from utils.data.dataset_builder import DatasetBuilder
from utils.writers.experiment_writer import ExperimentWriter

def read_config(
    config_path: str
) -> dict:
    config_path = Path(config_path)

    if not config_path.exists() or not config_path.is_file():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), config_path.name)
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError:
        raise yaml.YAMLError(f"{config_path} is an invalid YAML configuration file")
    
    return config

def run_evaluate(
    strategy: str,
    dataset_cfg: dict,
    model_cfg: dict=None,
    train_cfg: dict=None,
) -> None:
    print("Evaluating...\n")

    model_name = model_cfg["name"]
    
    experiment_dir = get_project_root() / "experiments"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir = experiment_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    model_dir = evaluation_dir / f"{model_name}"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Run corresponding model
    print("Model", model_name)

    builder = DatasetBuilder(dataset_cfg)
    dataset_name = builder.dataset_name

    if model_name == "protogain":
        data = builder.build(fill_zeros=True) #todo change seed to have different runs
    else:
        data = builder.build(fill_zeros=False) #todo change seed to have different runs

    dataset_dir = model_dir / f"{dataset_name}"
    miss_dir = dataset_dir / f"miss_{int(round(builder.miss_rate*100))}"
    miss_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    run_dir = miss_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    experiment_writer = ExperimentWriter(run_dir)

    input_dim = data.reference.shape[1]
    imputation_manager = ImputationManager(
        experiment_writer=experiment_writer,
        input_dim=input_dim, 
        model_cfg=model_cfg,
        train_cfg=train_cfg,
    )
    imputation_manager.run_evaluate(model_name=model_name, data=data, strategy=strategy)
