from pipelines.train_pipeline import run_train

def add_parser(subparsers):
    parser = subparsers.add_parser("train")
    parser.add_argument("--config",
        required=True, 
        help="Path to the configuration file"
    )
    parser.add_argument("--save",
        action="store_true", 
        help="Save model"
    )
    parser.add_argument("--run_dir",
        type=str,
        default=None
    )
    return parser

def run(args):
    run_train(args.config, args.save, args.run_dir)