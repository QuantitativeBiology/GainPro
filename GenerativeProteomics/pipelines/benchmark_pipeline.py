import yaml
from pathlib import Path
from datetime import datetime

from utils.paths import get_project_root
from evaluation.evaluator import Evaluator
from utils.data.dataset_builder import DatasetBuilder
from utils.writers.config_writer import ConfigWriter
from utils.writers.dataset_writer import DatasetWriter
from utils.writers.experiment_writer import ExperimentWriter

from wrappers.gain import GainImputer
from wrappers.global_mean import GlobalMeanImputer
from wrappers.tissue_mean import TissueMeanImputer
from wrappers.missforest import MissForestRImputer

from utils.train_hypers import TrainHypers
from utils.model_hypers import GainHypers, MissForestHypers

def read_config(
    cfg_path: Path,
    ) -> dict:
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def build_dataset(
    dataset_cfg: dict, 
    miss_rate: float,
    seed: int, 
    fill_zeros: bool,
    hint_rate: float=0.9,
):
    builder = DatasetBuilder(dataset_cfg, miss_rate=miss_rate, hint_rate=hint_rate)
    data = builder.build(fill_zeros=fill_zeros, seed=seed)
    return builder, data

def make_run_dir(parent: Path, run: int, seed: int) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    run_dir = parent / f"run_{run:03d}_seed_{seed}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def save_run_data(
    dataset_writer: DatasetWriter,
    builder: DatasetBuilder,
    data,
    experiment_writer: ExperimentWriter,
    miss_rate: float,
    seed: int,
) -> None:
    dataset_writer.save_data(data, experiment_writer.data_dir)
    dataset_writer.save_data_metadata(
        original_missingness=builder.original_missingness,
        miss_rate=miss_rate,
        current_missingness=builder.current_missingness,
        induction_strategy="None",
        seed=seed,
        out_dir=experiment_writer.data_dir,
    )

def needs_zero_fill(model_name: str) -> bool:
    return model_name == "protogain"

def create_imputer_factory(
    model: dict,
    input_dim: int=None,
):
    """Return a zero-argument callable that produces a fresh imputer instance."""
    name = model["name"]

    if name == "protogain":
        model_config = read_config(model["model_config"])
        train_config = read_config(model["train_config"])

        gain_hypers = GainHypers(model_config)
        train_hypers = TrainHypers(train_config)

        return lambda input_dim: GainImputer(input_dim=input_dim, gain_hypers=gain_hypers, train_hypers=train_hypers)

    if name == "missForest":
        model_config = read_config(model["model_config"])
        missforest_hypers = MissForestHypers(model_config)
        return lambda input_dim: MissForestRImputer(missforest_hypers=missforest_hypers)

    if name == "global_mean":
        return lambda input_dim: GlobalMeanImputer()

    if name == "tissue_mean":
        return lambda input_dim: TissueMeanImputer()

    if name == "mice":
        raise NotImplementedError("MICE imputer is not yet implemented.")

    raise ValueError(
        f"Unknown model '{name}'. "
        "Expected one of: 'protogain', 'missForest', 'global_mean', 'tissue_mean', 'mice'."
    )

def run_holdout(
    benchmark_cfg: dict,
    dataset_cfg: dict,
    model: dict,
    model_dir: Path,
    dataset_writer: DatasetWriter,
):
    strategy_cfg = benchmark_cfg["validation_strategy"][0]
    n_runs = benchmark_cfg["n_runs"]
    initial_seed = benchmark_cfg["initial_seed"]
    fill_zeros = needs_zero_fill(model["name"])

    for miss_level in strategy_cfg["missingness_levels"]:
        print(f"\nMissingness: {miss_level}")
        experiment_dir = model_dir / f"miss_{int(miss_level * 100)}"
        experiment_dir.mkdir(parents=True, exist_ok=True)

        for run in range(1, n_runs+1):
            print(f"\n============  Run {run}/{n_runs} ============")
            seed = initial_seed + (run - 1)

            run_dir = make_run_dir(experiment_dir, run, seed)
            experiment_writer = ExperimentWriter(run_dir)

            builder, data = build_dataset(dataset_cfg, miss_level, seed, fill_zeros)

            data_dir = Path(f"{builder.get_dataset_dir()}/miss_{int(miss_level * 100)}")
            data_dir.mkdir(exist_ok=True)
            dataset_writer.save_data(data, data_dir)
            save_run_data(dataset_writer, builder, data, experiment_writer, miss_level, seed)

            evaluator = Evaluator(strategy="holdout", experiment_writer=experiment_writer)
            if model["name"] == "protogain":
                imputer_factory = create_imputer_factory(
                    model,
                    input_dim=len(data.feature_names)
                )
            else:
                imputer_factory = create_imputer_factory(model)
            evaluator.evaluate(
                imputer_factory=imputer_factory, 
                data=data,
                x_tissue=data.tissue.detach().cpu(),
            )


def run_groupkfold(
    benchmark_cfg: dict,
    dataset_cfg: dict,
    model: dict,
    model_dir: Path,
    dataset_writer: DatasetWriter,
):
    strategy_cfg = benchmark_cfg["validation_strategy"][0]
    n_runs = benchmark_cfg["n_runs"]
    initial_seed = benchmark_cfg["initial_seed"]

    miss_rate = strategy_cfg["miss_rate"]

    num_folds = strategy_cfg["num_folds"]
    holdout_tissues = strategy_cfg["holdout_tissues"]

    fill_zeros = needs_zero_fill(model["name"])

    for run in range(1, n_runs+1):
        print(f"\n============  Run {run}/{n_runs} ============")
        seed = initial_seed + (run - 1)

        run_dir = make_run_dir(model_dir, run, seed)
        experiment_writer = ExperimentWriter(run_dir)

        builder, data = build_dataset(dataset_cfg, miss_rate, seed, fill_zeros)
        save_run_data(dataset_writer, builder, data, experiment_writer, miss_rate, seed)

        evaluator = Evaluator(strategy="groupkfold", experiment_writer=experiment_writer)
        if model["name"] == "protogain":
            imputer_factory = create_imputer_factory(
                model,
                input_dim=len(data.feature_names)
            )
        else:
            imputer_factory = create_imputer_factory(model)
        
        evaluator.evaluate(
            imputer_factory=imputer_factory,
            data=data,
            num_folds=num_folds,
            holdout_tissues=holdout_tissues,
        )

STRATEGY_RUNNERS = {
    "holdout": run_holdout,
    "groupkfold": run_groupkfold,
}

def run_benchmark(benchmark_cfg_path: Path, dataset_cfg_path: Path):
    benchmark_cfg = read_config(benchmark_cfg_path)
    dataset_cfg = read_config(dataset_cfg_path)

    validation_strategy = benchmark_cfg["validation_strategy"][0]["name"]
    strategy_runner = STRATEGY_RUNNERS.get(validation_strategy)
    if strategy_runner is None:
        raise ValueError(
            f"Invalid validation strategy '{validation_strategy}'. "
            f"Available strategies: {list(STRATEGY_RUNNERS)}."
        )

    dataset_name = Path(dataset_cfg["dataset"]["path"]).stem
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    benchmark_dir = (
        get_project_root() / "experiments/benchmark" / dataset_name / f"benchmark_{timestamp}"
    )
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    dataset_writer = DatasetWriter()

    for model in benchmark_cfg["models"]:
        print(f"\n{'='*50}\nModel: {model['name']}\n{'='*50}")
        model_dir = benchmark_dir / model["name"]
        model_dir.mkdir(parents=True, exist_ok=True)

        strategy_runner(benchmark_cfg, dataset_cfg, model, model_dir, dataset_writer)