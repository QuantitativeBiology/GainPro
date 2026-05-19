import argparse
import pandas as pd
from pathlib import Path
from typing import Dict

from utils.data.helper import load_tsv
from utils.helper import read_config
from utils.config_parser import MergeConfig

def load_datasets(
    dataset_paths: Dict[str, str],
) -> Dict[str, pd.DataFrame]:
    return {
        name: load_tsv(Path(path))
        for name, path in dataset_paths.items()
    }

def find_common_proteins(
    datasets: Dict[str, pd.DataFrame],
) -> list[str]:
    protein_sets = [set(df.columns) for df in datasets.values()]
    common_proteins = set.intersection(*protein_sets)

    if not common_proteins:
        raise ValueError("No overlapping proteins found.")
    return sorted(common_proteins)

def merge_proteomics_datasets(
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    common_proteins = find_common_proteins(datasets)
    processed = [
        df.loc[:, common_proteins].copy()
        for df in datasets.values()
    ]
    return pd.concat(processed, axis=0)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML configuration file",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    config = read_config(Path(args.config))
    MergeConfig.model_validate(config)

    datasets = load_datasets(config["datasets"])
    merged_df = merge_proteomics_datasets(datasets)

    output_path = Path(config["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path)


if __name__ == "__main__":
    main()