import logging
import argparse

from commands import benchmark

def main() -> None:
    parser = argparse.ArgumentParser(
        epilog="Example: python main.py --log-level INFO benchmark --config \"../configs/benchmark/protogain/holdout/benchmark_miss10.yaml\" --dataset \"../configs/datasets/PXD030304/PXD030304_no_control_multi_peptide_50pct_tissue/PXD030304_no_control_multi_peptide_50pct_tissue_transpose.yaml\"",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["INFO", "DEBUG"],
        metavar="LEVEL",
        help="Logging level. Choices: INFO, DEBUG. (default: INFO)",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        metavar="command",
        title="available commands",
        required=True
    )
    benchmark.add_parser(subparsers)
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        force=True,
    )

    args.func(args)

if __name__ == "__main__":
    main()