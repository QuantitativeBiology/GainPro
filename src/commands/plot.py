from pipelines.plot_pipeline import run_plot_pipeline

def add_parser(subparsers):
    parser = subparsers.add_parser("plot")

    plot_subparsers = parser.add_subparsers(dest="plot_type", required=True)

    for name in ["training", "evaluation", "latent"]:
        p = plot_subparsers.add_parser(name)
        p.add_argument(
            "--run_dir",
            required=True,
            help="Path to the experiment directory"
        )

    return plot_subparsers

def run(args):
    run_plot_pipeline(args.plot_type, args.run_dir)