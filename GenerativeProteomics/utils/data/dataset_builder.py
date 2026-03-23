import os
import errno
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

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
    replace_zeros_with_nans,
    drop_top_n_missing_proteins,
    drop_proteins_with_missingness_threshold,
    drop_all_missing_features_and_samples,
)

class DatasetBuilder:
    def __init__(
        self, 
        cfg: dict,
        miss_rate: float=None,
        hint_rate: float=None,
    ) -> "DatasetBuilder":
        self.cfg = cfg
        self.dataset_path = Path(self.cfg["dataset"]["path"])
        self.dataset_name = self.dataset_path.stem
        
        if miss_rate is None:
            self.miss_rate = self.cfg["dataset"]["miss_rate"]
        else:
            self.miss_rate = miss_rate

        print("miss rate in dataset builder", self.miss_rate)
        
        if hint_rate is None:
            self.hint_rate = self.cfg["dataset"]["hint_rate"]
        else:
            self.hint_rate = hint_rate

        if not self.dataset_path.exists():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.dataset_path.name)

        self.df = self._load()

        if "condition" in self.df.columns:
            cat = self.df["condition"].astype("category")
            self.cell_line = cat.cat.codes
            self.condition_mapping = dict(enumerate(cat.cat.categories))
            self.df = self.df.drop(columns=["condition"])

        if "Cell_line" in self.df.columns:
            cat = self.df["Cell_line"].astype("category")
            self.cell_line = cat.cat.codes
            self.condition_mapping = dict(enumerate(cat.cat.categories))
            self.df = self.df.drop(columns=["Cell_line"])

        if "tissue" in self.df.columns: #todo especificar no readme qual o formato dos datasets esperado
            cat = self.df["tissue"].astype("category")
            self.tissue = cat.cat.codes
            self.tissue_mapping = dict(enumerate(cat.cat.categories))
            self.df = self.df.drop(columns=["tissue"])

        self._clean()
        self._log_transform()

        self.reference = None
        self.missing = None
        self.mask = None

        # missingness metadata
        self.original_missingness = None
        self.current_missingness = None

    def _load(self) -> pd.DataFrame:
        if self.dataset_path.suffix == ".csv":
            df = load_csv(self.dataset_path)
        elif self.dataset_path.suffix == ".h5ad":
            df = load_anndata(self.dataset_path)
        elif self.dataset_path.suffix == ".tsv":
            df = load_tsv_with_condition(self.dataset_path)
        else:
            raise ValueError("Invalid file format.") 
        return df

    def _clean(self) -> pd.DataFrame:
        if self.cfg["dataset"]["replace_zeros"]:
            self.df = replace_zeros_with_nans(self.df)
        self.df = drop_all_missing_features_and_samples(self.df)
        if self.cfg["dataset"]["drop_top_n_missing"] != 0:
            self.df = drop_top_n_missing_proteins(self.df, self.cfg["dataset"]["drop_top_n_missing"])

    def _log_transform(self) -> None:
        values = self.df.values.astype(float)
        values_transformed = np.log2(values + 1)

        self.df = pd.DataFrame(
            data=values_transformed,
            index=self.df.index,
            columns=self.df.columns
        )

    def get_dataset_dir(self) -> Path:
        return self.dataset_path.parent

    def build(
        self, 
        fill_zeros: bool,
        seed: int=42,
    ) -> Data:

        self.observed_mask = compute_observed_mask(self.df)

        # - Normalize proteins (columns) between [0,1] -
        # observed mask due to the nans
        min_norm = self.df[self.observed_mask].min(axis=0) # min protein value
        max_norm = self.df[self.observed_mask].max(axis=0) # max protein value
        X_norm = (self.df[self.observed_mask] - min_norm) / (max_norm - min_norm)

        # todo test with standard scaler
        # scalers = {}
        # x_scaled = pd.DataFrame(index=self.df.index, columns=self.df.columns, dtype=float)
        # for col in self.df.columns:
        #     observed = self.df[col][self.observed_mask[col] == 1].values.reshape(-1, 1)
        #     scaler = StandardScaler()
        #     scaler.fit(observed)
        #     scalers[col] = scaler
        #     # transform all values
        #     x_scaled[col] = scaler.transform(self.df[col])

        # self.reference = x_scaled

        self.reference = X_norm
        self.missing = induce_missing(df=self.df, seed=seed, miss_rate=self.miss_rate)

        print("Dataset shape:", self.reference.shape)

        if self.missing is None: # all missing entries
            return None

        # save missingness metadata
        self.original_missingness = compute_missing_rate(self.df)
        self.current_missingness = compute_missing_rate(self.missing)

        print(f"Original missing rate: {self.original_missingness:.2%}")
        print(f"Current missing rate: {self.current_missingness:.2%}")
        print("\n")

        self.artificial_missing_mask = compute_evaluation_mask(
            reference_df=self.df,
            missing_df=self.missing
        )

        if fill_zeros is True:
            self.reference = self.reference.fillna(0)
            self.missing = self.missing.fillna(0)

        data = Data(
            miss_rate=self.miss_rate,
            hint_rate=self.hint_rate,
            reference=self.reference,
            missing=self.missing,
            observed_mask=self.observed_mask,
            artificial_missing_mask=self.artificial_missing_mask,
            cell_line=self.cell_line,
            min_norm=min_norm,
            max_norm=max_norm,
            # col_scalers=scalers,
        )
        return data