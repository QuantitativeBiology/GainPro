import argparse

from commands import train, evaluate, benchmark

def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    train.add_parser(subparsers)
    evaluate.add_parser(subparsers)
    benchmark.add_parser(subparsers)

    args = parser.parse_args()
    args.func(args)

    # todo prefiro assim: (adaptar depois)
    # parser.add_argument(
    #     "model",
    #     choices=["gainpro", "missForest", "mice", "mean"],
    #     default="gainpro",
    #     type=str,
    #     help="Which model to run. Available models: 'gainpro', 'missForest', 'mice', 'mean'."
    # )
    # parser.add_argument(
    #     "command",
    #     choices=["train", "evaluate", "predict", "impute"],
    #     default="train",
    #     type=str,
    #     help="Which mode to run. Available modes: 'train', 'evaluate', 'predict', 'impute'."
    # )
    # parser.add_argument(
    #     "--dataset-path",
    #     type=str,
    #     required=True,
    #     help="Path to the dataset."
    # )

if __name__ == "__main__":
    main()