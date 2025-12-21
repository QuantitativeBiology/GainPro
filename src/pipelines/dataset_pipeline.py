import os
import errno
import yaml
import pandas as pd
import numpy as np
import anndata as ad
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_config(config: dict) -> None:
    required_keys = {
        "datasets": str,
        "replace zeros": bool,
        "drop top n missing": int,
        "missingness levels": list,
    }
    for key, type in required_keys.items():
        if key not in config.keys():
            raise Exception(f"{key} is missing")
        if not isinstance(config[key], type):
            raise TypeError(f"Expected {key} to be {type.__name__}, instead is {type(config[key])}")

def read_config(config_path: str) -> dict:
    config_path = Path(config_path)

    if not config_path.exists() or not config_path.is_file():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), config_path.name)
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError:
        raise yaml.YAMLError(f"{config_path} is an invalid YAML configuration file")

    validate_config(config)
    return config

def load_csv(dataset_path: Path) -> pd.DataFrame:
    """
    Load the dataset from the CSV file.
    Args:
        dataset_path (str): Path to the dataset.
    Returns:
        pd.DataFrame: Dataset.
    """
    df = pd.read_csv(dataset_path, index_col=0)  # samples as rows (obs.) and proteins as columns
    return df

def load_anndata(dataset_path: Path) -> pd.DataFrame:
    "Tailored to HeLa datasets provided by PRIDE."
    adata = ad.read_h5ad(dataset_path)
    df = pd.DataFrame(
        adata.layers["IbaqLog"],
        index=adata.obs["SampleID"],
        columns=adata.var["ProteinName"]
    )
    return df

def replace_zeros_with_nans(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace zeros values with NaN.
    """
    df = df.replace(0.0, np.nan)
    assert df[df == 0.0].count().sum() == 0, f"Expected no zeros after replacement, found {df[df == 0].count().sum()}."
    return df

def drop_top_missing(df: pd.DataFrame, n_proteins: int) -> pd.DataFrame:
    """
    Drop the `n_proteins` with the highest missingness.
    """
    missing_pct_protein = df.isna().mean(axis=0)
    proteins_to_drop = missing_pct_protein.nlargest(n_proteins).index # index of the proteins to drop
    df_out = df.drop(columns=proteins_to_drop)
    assert df.shape[1] - df_out.shape[1] == n_proteins, (
        f"Expected to drop {n_proteins} proteins, dropped {df.shape[1] - df_out.shape[1]}"
    )
    return df_out

def save_df(df: pd.DataFrame, out: Path) -> None:
    df.to_csv(out, index=True)
    df_out = pd.read_csv(out, index_col=0)
    assert df.shape == df_out.shape, f"Expected shape {df_out.shape}, but got {df.shape}."

def compute_missing_rate(df: pd.DataFrame) -> float:
    total_missing = df.isna().sum().sum()
    missing_rate = total_missing / df.size
    return missing_rate

def induce_missing(df: pd.DataFrame, miss_rate: float=0.0) -> pd.DataFrame:
    np.random.seed(42)

    df_missing = df.copy()

    current_missingness = compute_missing_rate(df_missing)
    target_missingness = min(current_missingness + miss_rate, 1.0)
    if target_missingness == 1.0:
        raise Exception("Not feasible having a dataset with all missing entries.")
    size = df_missing.size
    n_target_missing_entries = int(target_missingness * size) # number of missing entries with target missingness rate

    # how many more values to mask
    n_current_missing_entries = df_missing.isna().sum().sum()
    additional_to_mask = n_target_missing_entries - n_current_missing_entries

    # identify all non-missing (eligible) positions
    eligible_positions = np.argwhere(~df_missing.isna().values)
    if additional_to_mask > len(eligible_positions):
        raise ValueError(f"Cannot induce missingness requested.")

    # randomly sample positions to mask
    to_mask_idx = np.random.choice(len(eligible_positions), size=additional_to_mask, replace=False)
    to_mask = eligible_positions[to_mask_idx]

    for i, j in to_mask:
        df_missing.iat[i, j] = np.nan

    return df_missing

def compute_mask(df: pd.DataFrame) -> pd.DataFrame:
    return ~df.isna()

def build_domain(df: pd.DataFrame) -> pd.DataFrame:
    # todo alterar isto de forma a puder receber do user
    domain_df = pd.DataFrame({
            "Domain": df.index.str.split('-Sample').str[0].to_list()
            },
            index=df.index
    )
    return domain_df

def run_dataset_pipeline(config_path: str) -> None:
    config_path = Path(config_path)
    config = read_config(config_path)
    logger.info(f"config {config}")

    dataset_path = config["datasets"]
    dataset_path = Path(dataset_path)

    if not dataset_path.is_file():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), dataset_path.name)
    
    if dataset_path.suffix == ".csv":
        df = load_csv(dataset_path)
    elif dataset_path.suffix == ".h5ad":
        df = load_anndata(dataset_path)
    else:
        raise Exception("Invalid file format.")

    out_dir = "data/processed" #fix para root do repo, para não ser relative path
    Path(f"{out_dir}/{dataset_path.stem}").mkdir(exist_ok=True)
    out_dir = f"{out_dir}/{dataset_path.stem}"
    logger.info(f"Out dir: {out_dir}")

    if config["replace zeros"] == "true":
        df = replace_zeros_with_nans(df)
    if config["drop top n missing"] != 0:
        df = drop_top_missing(df, config["drop top n missing"])

    missing_rates = config["missingness levels"]
    for miss_rate in missing_rates:
        print(f"Missingness of {miss_rate}")
        ref = df.copy() # reference dataset
        missing = induce_missing(df, miss_rate) # missing dataset
        mask = compute_mask(ref) # mask
        domain = build_domain(ref) # domain column #todo alterar isto de forma a puder receber do user
        codes, domains = domain["Domain"].factorize()
        # mapping = dict(zip(domains, range(len(domains))))
        domain_mapping = pd.DataFrame(codes, index=domain.index, columns=["Domain"])

        Path(f"{out_dir}/miss_{int(miss_rate*100)}").mkdir(exist_ok=True)
        out_dir_miss = f"{out_dir}/miss_{int(miss_rate*100)}"
        logger.info(f"Out dir miss: {out_dir_miss}")

        # Save datasets
        save_df(ref, f"{out_dir_miss}/reference.csv")
        save_df(missing, f"{out_dir_miss}/missing.csv")
        save_df(mask, f"{out_dir_miss}/mask.csv")
        save_df(domain, f"{out_dir_miss}/domain.csv")
        save_df(domain_mapping, f"{out_dir_miss}/domain_mapped.csv")

    #todo engineering wise is it here that the functions should be kept? like the load, drop?