import logging
import argparse

from commands import benchmark

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", default="WARNING")
    subparsers = parser.add_subparsers(dest="command", required=True)
    benchmark.add_parser(subparsers)
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        force=True,
    )
    logger = logging.getLogger()

    args.func(args)

if __name__ == "__main__":
    main()