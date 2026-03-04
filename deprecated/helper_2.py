import pandas as pd
from pathlib import Path

def load_tsv(
    dataset_path: Path
) -> pd.DataFrame:
    "Tailored to PRIDE datasets."
    data = pd.read_csv(
        dataset_path,
        sep="\t",
        lineterminator="\n",
        skiprows=(10),
        header=(0),
        usecols=(0, 1, 4),
    )
    df = data.pivot(index="sample_accession", columns="protein", values="ribaq")
    return df

def load_csv(
    dataset_path: Path,
) -> pd.DataFrame:
    df = pd.read_csv(
        dataset_path,
        index_col=0
    )
    return df

def load_dataset(
    dataset_path: Path,
) -> pd.DataFrame:
    if dataset_path.suffix == ".tsv":
        return load_tsv(dataset_path)
    elif dataset_path.suffix == ".csv":
        return load_csv(dataset_path)
    else:
        raise ValueError(f"Invalid {dataset_path.suffix} file type.")