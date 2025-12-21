import argparse
from commands import prepare_data, train, evaluate, predict, impute, transfer_impute, benchmark, experiment, plot

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    prepare_data.add_parser(subparsers)
    train.add_parser(subparsers)
    evaluate.add_parser(subparsers)
    predict.add_parser(subparsers)
    impute.add_parser(subparsers)
    benchmark.add_parser(subparsers)
    experiment.add_parser(subparsers)
    plot.add_parser(subparsers)

    args = parser.parse_args()

    if args.command == "prepare-data":
        prepare_data.run(args)
    if args.command == "train":
        train.run(args)
    if args.command == "evaluate":
        evaluate.run(args)
    if args.command == "predict":
        predict.run(args)
    if args.command == "impute":
        impute.run_impute(args)
    if args.command == "transfer-impute":
        transfer_impute.run(args)
    if args.command == "benchmark":
        benchmark.run(args)
    if args.command == "create-experiment":
        experiment.run(args)
    if args.command == "plot":
        plot.run(args)

if __name__ == "__main__":
    #todo cli
    main()