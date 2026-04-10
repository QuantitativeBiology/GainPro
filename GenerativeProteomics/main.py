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

if __name__ == "__main__":
    main()