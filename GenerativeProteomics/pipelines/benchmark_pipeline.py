import logging
from pathlib import Path
from datetime import datetime

from utils.configs.dataset_config import DatasetConfig
from utils.configs.benchmark_config import BenchmarkConfig
from utils.configs.model_config import GainConfig, MissForestConfig, AutoEncoderConfig

from imputation_manager import ImputationManager

from utils.paths import get_project_root
from utils.helper import load_benchmark, load_yaml, make_run_dir

from utils.data.dataset_builder import DatasetBuilder

from evaluation.evaluator import Evaluator

from utils.writers.dataset_writer import DatasetWriter
from utils.writers.experiment_writer import ExperimentWriter

logger = logging.getLogger(__name__)

def build_dataset(
    dataset_path: Path, 
    miss_rate: float,
    seed: int, 
    fill_zeros: bool,
):
    builder = DatasetBuilder(dataset_path, miss_rate=miss_rate)
    data = builder.build(fill_zeros=fill_zeros, seed=seed)
    return builder, data

def needs_zero_fill(model_name: str) -> bool:
    return model_name == "protogain"

def execute_run(
    seed: int,
    run_dir: Path,
    model_cfg: GainConfig | MissForestConfig | AutoEncoderConfig,
    dataset_path: Path,
    miss_rate: float,
    fill_zeros: bool,
    evaluator_kwargs: dict,
):
    """Build dataset, create imputer, and evaluate — shared by all strategies."""
    experiment_writer = ExperimentWriter(run_dir)

    builder, data = build_dataset(
        dataset_path=dataset_path,
        miss_rate=miss_rate,
        seed=seed,
        fill_zeros=fill_zeros,
    )
    dataset_writer = DatasetWriter()
    dataset_writer.save_data(data, experiment_writer.data_dir, transpose=data.transpose)
    dataset_writer.save_data_metadata(
        original_missingness=builder.original_missingness,
        miss_rate=miss_rate,
        current_missingness=builder.current_missingness,
        induction_strategy="None",
        seed=seed,
        out_dir=experiment_writer.data_dir,
    )

    manager = ImputationManager(model_cfg=model_cfg, data=data)
    evaluator = Evaluator(experiment_writer=experiment_writer, **evaluator_kwargs)
    evaluator.evaluate(imputer_factory=manager.build, data=data)

def run_holdout(
    benchmark_cfg: BenchmarkConfig,
    model_cfg: GainConfig | MissForestConfig | AutoEncoderConfig,
    dataset_cfg: DatasetConfig,
    benchmark_dir: Path,
) -> None:
    strategy_cfg = benchmark_cfg.validation
    dataset_path = dataset_cfg.dataset_path
    fill_zeros = needs_zero_fill(model_cfg.name)

    for miss_level in strategy_cfg.missing_levels:
        logger.info(f"\n Missingness: {miss_level}")
        experiment_dir = benchmark_dir / model_cfg.name / f"miss_{int(miss_level * 100)}"
        experiment_dir.mkdir(parents=True, exist_ok=True)

        for run in range(1, benchmark_cfg.n_runs + 1):
            logger.info(f"\n ============  Run {run}/{benchmark_cfg.n_runs} ============")
            seed = benchmark_cfg.initial_seed + (run - 1)

            execute_run(
                seed=seed,
                run_dir=make_run_dir(experiment_dir, run, seed),
                model_cfg=model_cfg,
                dataset_path=dataset_path,
                miss_rate=miss_level,
                fill_zeros=fill_zeros,
                evaluator_kwargs={
                    "strategy": "holdout", 
                },
            )

def run_groupkfold(
    benchmark_cfg: BenchmarkConfig,
    model_cfg: GainConfig | MissForestConfig | AutoEncoderConfig,
    dataset_cfg: DatasetConfig,
    benchmark_dir: Path,
) -> None:
    strategy_cfg = benchmark_cfg.validation
    dataset_path = dataset_cfg.dataset_path
    fill_zeros = needs_zero_fill(model_cfg.name)

    for run in range(1, benchmark_cfg.n_runs + 1):
        seed = benchmark_cfg.initial_seed + (run - 1)
        for fold_id in range(1, strategy_cfg.num_folds + 1):
            logger.info(f"\n ============  Run {run}/{benchmark_cfg.n_runs} ============")

            experiment_dir = benchmark_dir / model_cfg.name / f"fold_{fold_id}"
            experiment_dir.mkdir(parents=True, exist_ok=True)
        
            execute_run(
                seed=seed,
                run_dir=make_run_dir(experiment_dir, run, seed),
                model_cfg=model_cfg,
                dataset_path=dataset_path,
                miss_rate=strategy_cfg.miss_rate,
                fill_zeros=fill_zeros,
                evaluator_kwargs={
                    "strategy": "groupkfold",
                    "num_folds": strategy_cfg.num_folds,
                    "holdout_tissues": strategy_cfg.holdout_tissues,
                },
            )

STRATEGY_RUNNERS = {
    "holdout": run_holdout,
    "groupkfold": run_groupkfold,
}

def run_benchmark(
    benchmark_cfg_path: Path, 
    dataset_cfg_path: Path
) -> None:
    benchmark_cfg, _ = load_benchmark(benchmark_cfg_path)
    dataset_cfg = DatasetConfig.model_validate(load_yaml(dataset_cfg_path))

    validation_strategy = benchmark_cfg.validation.name
    strategy_runner = STRATEGY_RUNNERS.get(validation_strategy)
    if strategy_runner is None:
        raise ValueError(
            f"Invalid validation strategy '{validation_strategy}'. "
            f"Available strategies: {list(STRATEGY_RUNNERS)}."
        )

    dataset_name = dataset_cfg.dataset_path.stem
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    benchmark_dir = (
        get_project_root() / "experiments/benchmark" / dataset_name / f"benchmark_{timestamp}"
    )
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    for model in benchmark_cfg.models:
        logger.info(f"\n{'='*50}\nModel: {model.name}\n{'='*50}")

        strategy_runner(
            benchmark_cfg=benchmark_cfg,
            benchmark_dir=benchmark_dir,
            model_cfg=model,
            dataset_cfg=dataset_cfg,
        )