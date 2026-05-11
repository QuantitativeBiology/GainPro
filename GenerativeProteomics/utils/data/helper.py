import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path

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
        index_col=0,
        low_memory=False,
    )
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
    miss_rate: float=0.0,
    restrict_to_observed: bool=False,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df_missing = df.copy()

    n_rows, n_cols = df_missing.shape
    total_entries = n_rows * n_cols
    n_to_mask = int(miss_rate * total_entries)

    if restrict_to_observed:
        mask = df_missing.notna().values
    else:
        mask = np.ones((n_rows, n_cols), dtype=bool)

    eligible_positions = np.argwhere(mask)
    if n_to_mask > len(eligible_positions):
        raise ValueError("Not enough eligible entries to mask.")
    
    chosen = eligible_positions[
        rng.choice(len(eligible_positions), size=n_to_mask, replace=False)
    ]

    rows, cols = chosen[:, 0], chosen[:, 1]
    df_missing.values[rows, cols] = np.nan

    return df_missing

def compute_observed_mask(
    reference_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create binary mask indicating originally observed entries (observed entries=1, missing=0).
    """
    return reference_df.notna()

def compute_evaluation_mask(
    reference_df: pd.DataFrame,
    missing_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create binary mask indicating artificially missing entries (artificially missing=1).
    Used to assess the imputation performance.
    """
    observed_mask = reference_df.notna()
    missing_mask = missing_df.isna()
    artificial_mask = (observed_mask.values == True) & (missing_mask.values == True)
    artificial_mask = pd.DataFrame(
        artificial_mask,
        index=reference_df.index,
        columns=reference_df.columns
    )
    return artificial_mask

def generate_hint(
    observed_mask, 
    hint_rate: float,
):
    # For each entry, with probability hint_rate, reveal the true mask value
    # Otherwise, give 0.5 (uninformative)
    reveal = np.random.binomial(1, hint_rate, observed_mask.shape)  # 1 = reveal this entry
    hint = reveal * observed_mask + (1 - reveal) * 0.5
    return hint
