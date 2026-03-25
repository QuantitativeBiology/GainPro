import yaml
import numpy as np
from pathlib import Path
from datetime import datetime

from utils.paths import get_project_root
from utils.data.dataset_builder import DatasetBuilder
from utils.writers.config_writer import ConfigWriter
from utils.writers.dataset_writer import DatasetWriter
from utils.writers.experiment_writer import ExperimentWriter

from utils.train_hypers import TrainHypers
from wrappers.gain import GainImputationModel
from wrappers.missforest import MissForestRImputationModel
from utils.model_hypers import GainHypers, MissForestHypers

from scripts.generate_dev_split import kfold

def read_config(
    cfg_path: Path,
    ) -> dict:
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def run_benchmark(
    benchmark_cfg_path: Path,
    dataset_cfg_path: Path,
):
    benchmark_cfg = read_config(benchmark_cfg_path)
    dataset_cfg = read_config(dataset_cfg_path)
    
    dataset_name = Path(dataset_cfg["dataset"]["path"]).stem
    root_results = get_project_root() / "experiments/benchmark"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    benchmark_dir = root_results / f"{dataset_name}/benchmark_{timestamp}"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    # Writers
    config_writer = ConfigWriter()
    dataset_writer = DatasetWriter()

    for model in benchmark_cfg["models"]:

        model_dir = benchmark_dir / f"{model['name']}"
        model_dir.mkdir(parents=True, exist_ok=True)

        for miss_level in benchmark_cfg["missingness_levels"]:
            print(f"\n Missingness: {miss_level}")

            experiment_dir = model_dir / f"miss_{int(miss_level*100)}"
            experiment_dir = Path(experiment_dir)
            experiment_dir.mkdir(parents=True, exist_ok=True)

            # Run each missingness level 'n_runs' times
            for run in range(1, benchmark_cfg["n_runs"]+1):
                print(f"\n============  Run {run}/{benchmark_cfg['n_runs']} ============")
                
                seed = benchmark_cfg["initial_seed"] + run

                timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                run_dir = experiment_dir / f"run_{run:03d}_seed_{seed}_{timestamp}"
                run_dir = Path(run_dir)
                run_dir.mkdir(parents=True, exist_ok=True)
                experiment_writer = ExperimentWriter(run_dir)

                config_writer.snapshot_config_tree(benchmark_cfg_path, out_dir=experiment_writer.cfg_dir)

                # Add missingness
                if model["name"] == "protogain":
                    fill_zeros = True
                else:
                    fill_zeros = False

                dataset_builder = DatasetBuilder(
                    dataset_cfg,
                    miss_rate=miss_level,
                    hint_rate=0.9,
                )

                data = dataset_builder.build(
                    fill_zeros=fill_zeros,
                    seed=seed
                )

                data_dir = Path(f"{dataset_builder.get_dataset_dir()}/miss_{int(miss_level*100)}")
                data_dir.mkdir(exist_ok=True)
                dataset_writer.save_data(data, data_dir)
                dataset_writer.save_data(data, experiment_writer.data_dir)
                dataset_writer.save_data_metadata(
                    original_missingness=dataset_builder.original_missingness, 
                    miss_rate=miss_level, 
                    current_missingness=dataset_builder.current_missingness,
                    induction_strategy="None",
                    seed=seed,
                    out_dir=experiment_writer.data_dir
                )

                if model["name"] == "protogain":
                    input_dim = len(data.feature_names)
                    model_config = read_config(model["model_config"])
                    train_config = read_config(model["train_config"])
                    gain = GainImputationModel(
                        input_dim=input_dim,
                        gain_hypers=GainHypers(model_config),
                        train_hypers=TrainHypers(train_config),
                    )
                if model["name"] == "missForest":
                    model_config = read_config(model["model_config"])
                    missForest = MissForestRImputationModel(
                        missforest_hypers=MissForestHypers(model_config)
                    )

                # Validation strategy
                val_strategy = benchmark_cfg["validation_strategy"][0]["name"]

                if val_strategy == "k-fold":
                    # Split and save folds
                    num_folds=benchmark_cfg["validation_strategy"][0]["num_folds"]
                    kfold(
                        dataset_path=Path(dataset_cfg["dataset"]["path"]),
                        num_folds=num_folds,
                        seed=seed
                    )

                    # Retrieve the train/test fold's indices
                    folds_dir = get_project_root() / f"data/splits/{Path(dataset_cfg["dataset"]["path"]).stem}/k-fold"

                    idxs_folds = list()
                    for fold in folds_dir.iterdir():
                        trainval_idx = np.load(fold/f"trainval_idx.npy")
                        test_idx = np.load(fold/f"test_idx.npy")
                        idxs_folds.append({"trainval_idx": trainval_idx, "test_idx": test_idx})
                    
                    if model["name"] == "protogain":
                        gain.evaluate(
                            strategy=val_strategy,
                            data=data,
                            experiment_writer=experiment_writer,
                            idxs_folds=idxs_folds,
                            num_folds=num_folds,
                        )
                    if model["name"] == "missForest":
                        missForest.evaluate(
                            data=data,
                            strategy=val_strategy,
                            experiment_writer=experiment_writer,
                            idxs_folds=idxs_folds,
                            num_folds=num_folds
                        )
