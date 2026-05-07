import argparse

from commands import benchmark

def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark.add_parser(subparsers)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()