import yaml
from pathlib import Path

from pipelines.evaluate_pipeline import run_evaluate

def add_parser(subparsers):
    parser = subparsers.add_parser("evaluate")
    parser.add_argument("-dataset",
        type=str, 
        help="Path to the dataset configuration file"
    )
    parser.add_argument("-model",
        type=str, 
        help="Path to the model configuration file"
        # choices=["missForest", "gain", "mice", "mean"]
    )
    parser.add_argument("-train",
        type=str, 
        help="Path to the train configuration file"
    )
    parser.add_argument("-strategy",
        type=str, 
        choices=["hold-out", "k-fold", "group-k-fold"]
    )
    parser.set_defaults(func=run)
    return parser

def read_config(
    cfg_path: Path,
    ) -> dict:
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def run(args) -> None:
    dataset_cfg = read_config(args.dataset)
    model_cfg = read_config(args.model)
    train_cfg = read_config(args.train)

    run_evaluate(
        strategy=args.strategy,
        dataset_cfg=dataset_cfg,
        model_cfg=model_cfg,
        train_cfg=train_cfg
    )