import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def load_csv(
    dataset_path: Path,
) -> pd.DataFrame:
    """
    Load the dataset from the CSV file.
    Args:
        dataset_path (str): Path to the dataset.
    Returns:
        pd.DataFrame: Dataset.
    """
    df = pd.read_csv(
        dataset_path, 
        index_col=0
    )  # samples as rows (obs.) and proteins as columns
    return df

def load_anndata(
    dataset_path: Path
) -> pd.DataFrame:
    "Tailored to HeLa datasets provided by PRIDE."
    adata = ad.read_h5ad(dataset_path)
    df = pd.DataFrame(
        adata.layers["IbaqLog"],
        index=adata.obs["SampleID"],
        columns=adata.var["ProteinName"]
    )
    return df

def load_tsv(
    dataset_path: Path
) -> pd.DataFrame:
    "Tailored to PRIDE datasets."
    data = pd.read_csv(
        dataset_path,
        sep="\t",
        lineterminator="\n",
        skiprows=10,
        header=0,
        usecols=("protein", "sample_accession", "ribaq"),
    )
    df = data.pivot(index="sample_accession", columns="protein", values="ribaq")
    return df

def load_tsv_with_condition(
    dataset_path: Path
) -> pd.DataFrame:
    "Tailored to PRIDE datasets."
    data = pd.read_csv(
        dataset_path,
        sep="\t",
        lineterminator="\n",
        skiprows=10,
        header=0,
        usecols=["protein", "sample_accession", "condition", "ribaq"],
    )

    sample_condition = (
        data[["sample_accession", "condition"]]
        .drop_duplicates()
        .set_index("sample_accession")["condition"]
    )

    X = data.pivot(index="sample_accession", columns="protein", values="ribaq")
    sample_condition = sample_condition.loc[X.index]
    X["condition"] = sample_condition
    return X

def load_reference(dir: Path) -> pd.DataFrame:
    return load_csv(f"{dir}/reference.csv")

def load_missing(dir: Path) -> pd.DataFrame:
    return load_csv(f"{dir}/missing.csv")

def load_mask(dir: Path) -> pd.DataFrame:
    return load_csv(f"{dir}/mask.csv")

def load_domain(dir: Path) -> pd.DataFrame:
    return load_csv(f"{dir}/domain.csv")

def load_domain_mapped(dir: Path) -> pd.DataFrame:
    return load_csv(f"{dir}/domain_mapped.csv")

def load_dfs(dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reference = load_reference(dir)
    missing = load_missing(dir)
    mask = load_mask(dir)
    domain = load_domain(dir)
    domain_mapped = load_domain_mapped(dir)
    return reference, missing, mask, domain, domain_mapped

def replace_zeros_with_nans(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Replace zeros values with NaN.
    """
    df = df.replace(0.0, np.nan)
    assert df[df == 0.0].count().sum() == 0, f"Expected no zeros after replacement, found {df[df == 0].count().sum()}."
    return df

def drop_all_missing_features_and_samples(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Drop rows (samples) and columns (features) that are entirely missing.
    """
    return df.dropna(axis=0, how="all").dropna(axis=1, how="all")

def drop_top_n_missing_proteins(
    df: pd.DataFrame, 
    n_proteins: int,
) -> pd.DataFrame:
    """
    Drop the top 'n_proteins' proteins with the highest missingness.
    """
    missing_pct_protein = df.isna().mean(axis=0)
    proteins_to_drop = missing_pct_protein.nlargest(n_proteins).index # index of the proteins to drop
    df_out = df.drop(columns=proteins_to_drop)
    assert df.shape[1] - df_out.shape[1] == n_proteins, (
        f"Expected to drop {n_proteins} proteins, dropped {df.shape[1] - df_out.shape[1]}"
    )
    return df_out

def drop_proteins_with_missingness_threshold(
    df: pd.DataFrame, 
    max_missingness: float,
) -> pd.DataFrame:
    """
    Drop proteins whose missingness proportion is greater than `miss_rate`.
    """
    missing_fraction = df.isna().mean(axis=0) # per protein (column-wise)
    proteins_to_drop = missing_fraction[missing_fraction > max_missingness].index
    filtered_df = df.drop(columns=proteins_to_drop)
    print(
        f"Dropped {len(proteins_to_drop)} proteins "
        f"({len(proteins_to_drop) / df.shape[1]:.2%} of total)."
    )
    return filtered_df

def compute_missing_rate(
    df: pd.DataFrame
) -> float:
    total_missing = df.isna().sum().sum()
    missing_rate = total_missing / df.size
    return missing_rate

def induce_missing(
    df: pd.DataFrame, 
    seed: int,
    # ceiling_miss_rate: float=1.0,
    miss_rate: float=0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    rng = np.random.default_rng(seed)

    df_missing = df.copy()

    current_missing = df_missing.isna().sum().sum()
    total_entries = df_missing.size

    additional_to_mask = int(miss_rate * total_entries)

    # positions that currently have values
    eligible_positions = np.argwhere(df_missing.notna().values)

    if additional_to_mask > len(eligible_positions):
        raise ValueError("Not enough observed entries to mask.")

    chosen_idx = rng.choice(len(eligible_positions), size=additional_to_mask, replace=False)
    
    for i, j in eligible_positions[chosen_idx]:
        df_missing.iat[i, j] = np.nan

    return df_missing

def compute_observed_mask(
    reference_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create binary mask indicating originally observed entries (observed entries = 1, missing = 0).
    """
    return reference_df.notna()

def compute_evaluation_mask(
    reference_df: pd.DataFrame,
    missing_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create binary mask indicating artificially introduced removed entries (artificially removed = 0).

    Used to assess the imputation performance.
    """
    artificially_removed = ~(~reference_df.isna() & missing_df.isna())
    return artificially_removed

def build_domain(
    df: pd.DataFrame
) -> pd.DataFrame:
    # todo alterar isto de forma a puder receber do user
    domain_df = pd.DataFrame({
            "Domain": df.index.str.split('-Sample').str[0].to_list()
            },
            index=df.index
    )
    return domain_df
