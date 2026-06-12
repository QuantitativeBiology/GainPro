from pathlib import Path

from pipelines.benchmark_pipeline import run_benchmark

def add_parser(subparsers):
    parser = subparsers.add_parser(
        "benchmark",
        help="Evaluate one or more models on a dataset.",
        description="Run a full benchmark across models and datasets defined in the config.",
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        metavar="FILE",
        help="Path to the benchmark configuration file.",
    )
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        required=True,
        metavar="FILE",
        help="Path to the dataset configuration file.",
    )
    parser.set_defaults(func=run)
    return parser

def run(args) -> None:
    run_benchmark(
        benchmark_cfg_path=Path(args.config), 
        dataset_cfg_path=Path(args.dataset)
    )