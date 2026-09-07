import torch
import logging
from typing import List
from enum import Enum
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path

logger = logging.getLogger(__name__)
 
class MissingMechanism(Enum):
    """Missing data mechanisms"""
    MCAR = "MCAR" # Missing Completely At Random
    MNAR = "MNAR" # Missing Not At Random

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
) -> np.ndarray:
    # For each entry, with probability hint_rate, reveal the true mask value
    # Otherwise, give 0.5 (uninformative)
    reveal = np.random.binomial(1, hint_rate, observed_mask.shape)  # 1 = reveal this entry
    hint = reveal * observed_mask + (1 - reveal) * 0.5
    return hint

def convert_tensors_dtype(
    tensor: torch.Tensor | List[torch.Tensor],
    dtype: torch.dtype,
) -> List[torch.Tensor]:
    if isinstance(tensor, list):
        return [t.to(dtype) for t in tensor]
    return tensor.to(dtype)

def induce_missing(
    df: pd.DataFrame,
    seed: int,
    miss_rate: float,
    missing_mechanism: MissingMechanism=MissingMechanism.MCAR,
    steepness: float=1.0,
) -> pd.DataFrame:
    r"""Induce missing values in a DataFrame under the specified mechanism.

    Args:
        - df (pd.DataFrame) : Must contain at least one observed (non-NaN) value.
        - seed (int) : Seed for reproducibility.
        - miss_rate (float) : Target missing rate over the full matrix. :math:`\text{miss\_rate} \in [0, 1]`.
        - missing_mechanism (`MissingMechanism`): `MissingMechanism.MCAR` (default) masks entries uniformly at random; 
            `MissingMechanism.MNAR` masks low-valued entries with higher probability via a sigmoid function.
        - steepness (float): Sigmoid steepness used only when `mechanism=MissingMechanism.MNAR`. 
            Higher values create a sharper boundary between masked and unmasked entries. 
            Ignored for `mechanism=MissingMechanism.MCAR`. Default = 5.0.

    Returns:
        A copy of `df` with additional NaN values introduced.
    """
    if not 0.0 <= miss_rate <= 1.0:
        raise ValueError(f"`miss_rate` must be in [0, 1], got {miss_rate}.")
    if not isinstance(missing_mechanism, MissingMechanism):
        raise ValueError(
            f"`mechanism` must be a MissingMechanism instance, got {mechanism!r}. "
            f"Valid options: {[m.value for m in MissingMechanism]}."
        )

    if missing_mechanism is MissingMechanism.MCAR:
        return _induce_mcar(df, seed=seed, miss_rate=miss_rate)
    return _induce_mnar(df, seed=seed, miss_rate=miss_rate, steepness=steepness)

def _induce_mcar(
    df: pd.DataFrame,
    seed: int,
    miss_rate: float,
) -> pd.DataFrame:
    """Uniformly mask observed entries to reach `miss_rate` across the matrix."""
    rng = np.random.default_rng(seed)
    df = df.copy()

    observed_mask = df.notna().values
    n_entries = df.size
    n_observed = int(observed_mask.sum())
    n_target_missing = round(miss_rate * n_entries)
    n_to_mask = n_target_missing - (n_entries - n_observed)

    if n_to_mask <= 0:
        return df  # Already at or above the target miss_rate

    eligible_positions = np.argwhere(observed_mask)
    if n_to_mask > len(eligible_positions):
        raise ValueError(
            f"Cannot mask {n_to_mask} additional entries: only "
            f"{len(eligible_positions)} observed values are available."
        )

    chosen = rng.choice(len(eligible_positions), size=n_to_mask, replace=False)
    rows, cols = eligible_positions[chosen, 0], eligible_positions[chosen, 1]
    df.iloc[rows, cols] = np.nan
    return df

def _induce_mnar(
    df: pd.DataFrame,
    seed: int,
    miss_rate: float,
    steepness: float,
) -> pd.DataFrame:
    """Mask entries via a sigmoid so that lower values are more likely missing.

    The sigmoid probabilities are used as weights to decide which entries will be
    masked. Lower-valued entries carry higher weight.

    The sigmoid inflection point is found with binary search so that the
    expected global miss rate matches `miss_rate`.

    Masking weight over observed entries::
        w(x) = sigmoid(-steepness * (x - inflection))
             = 1 / (1 + exp(steepness * (x - inflection)))
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    values = df.values.astype(float)

    n_entries = values.size
    observed_mask = ~np.isnan(values)
    observed_values = values[observed_mask] # 1-D array of observed entries
    n_already_missing = n_entries - len(observed_values)

    n_target_missing = round(miss_rate * n_entries)
    n_to_mask = n_target_missing - n_already_missing

    if n_to_mask <= 0:
        return df  # Already at or above target miss_rate

    if n_to_mask > len(observed_values):
        raise ValueError(
            f"Cannot mask {n_to_mask} additional entries: only "
            f"{len(observed_values)} observed values are available."
        )
    
    # Sigmoid weights: lower values get higher weight (more likely masked).
    # Anchor the inflection at the median so `steepness` is interpretable
    # independent of the data's absolute scale.
    inflection = float(np.median(observed_values))
    weights = 1.0 / (1.0 + np.exp(steepness * (observed_values - inflection)))
    weights /= weights.sum() # normalize to a probability distribution

    chosen = rng.choice(len(observed_values), size=n_to_mask, replace=False, p=weights)

    observed_positions = np.argwhere(observed_mask)
    rows, cols = observed_positions[chosen, 0], observed_positions[chosen, 1]
    df.values[rows, cols] = np.nan
    return df