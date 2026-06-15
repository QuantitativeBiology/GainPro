import logging
from pathlib import Path
from datetime import datetime

from imputation_manager import ImputationManager
from evaluation.evaluator import Evaluator

from utils.paths import get_project_root
from utils.data.dataset_builder import DatasetBuilder
from utils.configs.dataset_config import DatasetConfig
from utils.configs.benchmark_config import BenchmarkConfig
from utils.configs.model_entry_config import ModelEntryConfig
from utils.writers.dataset_writer import DatasetWriter
from utils.writers.experiment_writer import ExperimentWriter
from utils.helper import load_benchmark, load_yaml, make_run_dir

logger = logging.getLogger(__name__)

def execute_run(
    seed: int,
    run_dir: Path,
    model_cfg: ModelEntryConfig,
    dataset_cfg: DatasetConfig,
    miss_rate: float,
    evaluator_kwargs: dict,
) -> None:
    experiment_writer = ExperimentWriter(run_dir)

    manager = ImputationManager(model_cfg=model_cfg)

    builder = DatasetBuilder(cfg=dataset_cfg, model_name=model_cfg.name, miss_rate=miss_rate)
    data = builder.build(fill_strategy=manager.fill_strategy, seed=seed)
    manager.set_data(data)

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
    
    evaluator = Evaluator(experiment_writer=experiment_writer, **evaluator_kwargs)
    evaluator.evaluate(imputer_factory=manager.build, data=data)

def run_holdout(
    benchmark_cfg: BenchmarkConfig,
    model_cfg: ModelEntryConfig,
    dataset_cfg: DatasetConfig,
    benchmark_dir: Path,
) -> None:
    strategy_cfg = benchmark_cfg.validation
    logger.debug(f"\n Model name: {model_cfg.name.lower()}")

    for miss_level in strategy_cfg.target_missing:
        logger.info(f"\n Missingness: {miss_level}")
        experiment_dir = benchmark_dir / model_cfg.name.lower() / f"miss_{int(miss_level * 100)}"
        experiment_dir.mkdir(parents=True, exist_ok=True)

        for run in range(1, benchmark_cfg.n_runs + 1):
            logger.info(f"\n ============  Run {run}/{benchmark_cfg.n_runs} ============")
            seed = benchmark_cfg.initial_seed + (run - 1)

            execute_run(
                seed=seed,
                run_dir=make_run_dir(experiment_dir, run, seed),
                model_cfg=model_cfg,
                dataset_cfg=dataset_cfg,
                miss_rate=miss_level,
                evaluator_kwargs={
                    "strategy": "holdout", 
                },
            )

def run_groupkfold(
    benchmark_cfg: BenchmarkConfig,
    model_cfg: ModelEntryConfig,
    dataset_cfg: DatasetConfig,
    benchmark_dir: Path,
) -> None:
    strategy_cfg = benchmark_cfg.validation

    for run in range(1, benchmark_cfg.n_runs + 1):
        seed = benchmark_cfg.initial_seed + (run - 1)
        for fold_id in range(1, strategy_cfg.num_folds + 1):
            logger.info(f"\n ============  Run {run}/{benchmark_cfg.n_runs} ============")

            experiment_dir = benchmark_dir / model_cfg.name.lower() / f"fold_{fold_id}"
            experiment_dir.mkdir(parents=True, exist_ok=True)
        
            execute_run(
                seed=seed,
                run_dir=make_run_dir(experiment_dir, run, seed),
                model_cfg=model_cfg,
                dataset_cfg=dataset_cfg,
                miss_rate=strategy_cfg.miss_rate,
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
    benchmark_cfg = load_benchmark(benchmark_cfg_path)
    dataset_cfg = DatasetConfig.model_validate(load_yaml(dataset_cfg_path))

    validation_strategy = benchmark_cfg.validation.name.lower()
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

    for model_cfg in benchmark_cfg.models:
        logger.info(f"\n{'='*50}\nModel: {model_cfg.name.lower()}\n{'='*50}")

        strategy_runner(
            benchmark_cfg=benchmark_cfg,
            benchmark_dir=benchmark_dir,
            model_cfg=model_cfg,
            dataset_cfg=dataset_cfg,
        )