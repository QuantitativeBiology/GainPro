import os
import errno
import numpy as np
import pandas as pd
from pathlib import Path

from utils.data.dataset import Data
from utils.data.helper import (
    load_csv,
    load_tsv,
    load_tsv_with_condition,
    load_anndata,
    compute_observed_mask,
    compute_evaluation_mask,
    induce_missing, 
    compute_missing_rate,
)

class DatasetBuilder:
    def __init__(
        self, 
        dataset_path: Path,
        miss_rate: float=0.0,
    ) -> "DatasetBuilder":
        if not dataset_path.exists():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.dataset_path.name)
        self.dataset_path = dataset_path
        self.dataset_name = self.dataset_path.stem

        self.load_dataset()
        self.clean()
        self.log_transform()

        self.reference = None
        self.missing = None
        self.observed_mask = None
        self.artificial_missing_mask = None

        self.miss_rate = miss_rate
        self.original_missingness = None
        self.current_missingness = None

    def load_dataset(self) -> None:
        if self.dataset_path.suffix == ".csv":
            self.df = load_csv(self.dataset_path)
        elif self.dataset_path.suffix == ".h5ad":
            self.df = load_anndata(self.dataset_path)
        elif self.dataset_path.suffix == ".tsv":
            self.df = load_tsv_with_condition(self.dataset_path)
        else:
            raise ValueError("Invalid file format. Valid formats: csv, tsv and h5ad.") 

    def get_dataset_dir(self) -> Path:
        return self.dataset_path.parent
    
    def clean(self) -> None:
        if "Cell_line" in self.df.columns:
            cat = self.df["tissue"].astype("category")
            self.cell_line = cat.cat.codes
            self.cell_line_mapping = dict(enumerate(cat.cat.categories))
            self.df = self.df.drop(columns=["Cell_line"])

        if "tissue" in self.df.columns:
            cat = self.df["tissue"].astype("category")
            self.tissue = cat.cat.codes
            self.tissue_mapping = dict(enumerate(cat.cat.categories))
            self.df = self.df.drop(columns=["tissue"])

    def log_transform(self) -> None:
        values = self.df.values.astype(float)
        values_transformed = np.log2(values + 1)
        self.df = pd.DataFrame(
            data=values_transformed,
            index=self.df.index,
            columns=self.df.columns
        )

    def build(
        self, 
        fill_zeros: bool,
        seed: int=42,
    ) -> Data:
        self.observed_mask = compute_observed_mask(self.df)

        # - Normalize proteins between [0,1] -
        min_norm = self.df[self.observed_mask].min(axis=0) # min protein value
        max_norm = self.df[self.observed_mask].max(axis=0) # max protein value
        X_norm = (self.df[self.observed_mask] - min_norm) / (max_norm - min_norm)

        self.reference = X_norm
        self.missing = induce_missing(df=self.reference, seed=seed, miss_rate=self.miss_rate, restrict_to_observed=False)

        self.original_missingness = compute_missing_rate(self.df)
        self.current_missingness = compute_missing_rate(self.missing)

        print("Dataset shape:", self.reference.shape)
        print(f"Original missing rate: {self.original_missingness:.2%}")
        print(f"Current missing rate: {self.current_missingness:.2%}")
        print("\n")

        self.artificial_missing_mask = compute_evaluation_mask(
            reference_df=self.df,
            missing_df=self.missing
        )

        if fill_zeros:
            self.reference = self.reference.fillna(0)
            self.missing = self.missing.fillna(0)

        data = Data(
            reference=self.reference,
            missing=self.missing,
            observed_mask=self.observed_mask,
            artificial_missing_mask=self.artificial_missing_mask,
            tissue=self.tissue,
            tissue_mapping=self.tissue_mapping,
            cell_line=self.cell_line,
            cell_line_mapping=self.cell_line_mapping,
            min_norm=min_norm,
            max_norm=max_norm,
        )
        return data