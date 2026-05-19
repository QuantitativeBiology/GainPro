import os
import yaml
import errno
import numpy as np
from pathlib import Path
from datetime import datetime

from utils.helper import load_yaml
from utils.paths import get_project_root
from imputation_manager import ImputationManager
from utils.data.dataset_builder import DatasetBuilder
from utils.writers.experiment_writer import ExperimentWriter

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

    # Retrieve the train/test fold's indices
    if strategy == "k-fold":
        # Iterate over the directory
        dataset_path = Path(dataset_cfg["dataset"]["path"])
        print("dataset path", dataset_path)
        folds_dir = get_project_root() / f"data/splits/{dataset_path.stem}/k-fold"
        print("folds dir", folds_dir)

        idxs_folds = list()
        for fold in folds_dir.iterdir():
            trainval_idx = np.load(fold/f"trainval_idx.npy")
            test_idx = np.load(fold/f"test_idx.npy")
            idxs_folds.append({"trainval_idx": trainval_idx, "test_idx": test_idx})
    else:
        idxs_folds = None

    input_dim = data.reference.shape[1]
    imputation_manager = ImputationManager(
        experiment_writer=experiment_writer,
        model_name=model_name, 
        input_dim=input_dim, 
        model_cfg=model_cfg,
        train_cfg=train_cfg,
    )
    imputation_manager.run_evaluate(
        strategy=strategy,
        idxs_folds=idxs_folds,
        data=data,
    )
