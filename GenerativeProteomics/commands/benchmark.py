from pathlib import Path

from pipelines.benchmark_pipeline import run_benchmark

def add_parser(subparsers):
    parser = subparsers.add_parser("benchmark")
    parser.add_argument("-cfg",
        type=str, 
        help="Path to the benchmark configuration file"
    )
    parser.add_argument("-dataset",
        type=str, 
        help="Path to the dataset configuration file"
    )
    parser.set_defaults(func=run)
    return parser

def run(args) -> None:
    run_benchmark(
        benchmark_cfg_path=Path(args.cfg), 
        dataset_cfg_path=Path(args.dataset)
    )