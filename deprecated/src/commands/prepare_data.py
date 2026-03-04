from pipelines.dataset_pipeline import run_dataset_pipeline

def add_parser(subparsers):
    parser = subparsers.add_parser("prepare-data")
    parser.add_argument("--config",
                        required=True, 
                        help="Path to the configuration file")
    return parser

def run(args):
    run_dataset_pipeline(args.config)