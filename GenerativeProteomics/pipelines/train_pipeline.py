import logging
from pathlib import Path
from datetime import datetime

from utils.paths import get_project_root
from imputation_manager import ImputationManager
from utils.data.dataset_builder import DatasetBuilder
from utils.writers.experiment_writer import ExperimentWriter
from utils.writers.metadata_writer import MetadataWriter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_train(
    model_cfg: dict,
    dataset_cfg: dict,
) -> None:
    model_name = model_cfg["name"]

    experiment_dir = get_project_root() / "experiments"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    train_dir = experiment_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    model_dir = train_dir / f"{model_name}"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Run corresponding model
    print("Model", model_name)
    builder = DatasetBuilder(dataset_path=Path(dataset_cfg["dataset"]["path"]))
    dataset_name = builder.dataset_name

    if model_name == "protogain":
        data = builder.build(fill_zeros=True)
    else:
        data = builder.build(fill_zeros=False)

    dataset_dir = model_dir / f"{dataset_name}"
    miss_dir = dataset_dir / f"miss_{int(round(builder.miss_rate*100))}"
    miss_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    run_dir = miss_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    experiment_writer = ExperimentWriter(run_dir)

    imputation_manager = ImputationManager(experiment_writer)
    imputation_manager.run_train(model_cfg=model_cfg, dataset_name=dataset_name, data=data)