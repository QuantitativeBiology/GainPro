import os
import errno
import logging
import numpy as np
import pandas as pd
from pathlib import Path


from utils.data.dataset import Data
from utils.configs.dataset_config import DatasetConfig
from utils.data.normalizer_registry import build_normalizer
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

logger = logging.getLogger(__name__)

class DatasetBuilder:
    def __init__(
        self,
        cfg: DatasetConfig,
        miss_rate: float=0.0,
    ) -> "DatasetBuilder":
        if not cfg.dataset_path.exists():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.dataset_path.name)
        self.dataset_path = cfg.dataset_path
        self.dataset_name = self.dataset_path.stem

        self.load_dataset()

        # Matrix: samples x (proteins + tissue)
        # Transposed matrix: (proteins + tissue) x samples
        self.transposed = "tissue" in self.df.index
        self.extract_tissue_labels()

        if cfg.log_transform:
            self.log2p1_transform()
        self.normalizer = build_normalizer(cfg.normalizer)

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
            raise ValueError(
                "Invalid file format. " \
                "Valid formats: csv, tsv and h5ad."
            ) 

    def get_dataset_dir(self) -> Path:
        return self.dataset_path.parent
    
    def extract_tissue_labels(self) -> None:
        if self.transposed:
            self._extract_tissue_from_index()
        else:
            self._extract_tissue_from_columns()

    def _extract_tissue_from_columns(self) -> None:
        cat = self.df["tissue"].astype("category")
        self.tissue = cat.cat.codes
        self.tissue_mapping = dict(enumerate(cat.cat.categories))
        self.df = self.df.drop(columns=["tissue"])

    def _extract_tissue_from_index(self) -> None:
        cat = self.df.loc["tissue"].astype("category")
        self.tissue = cat.cat.codes
        self.tissue_mapping = dict(enumerate(cat.cat.categories))
        self.df = self.df.drop(index=["tissue"])

    def log2p1_transform(self) -> None:
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

        self.normalizer.fit(self.df, self.observed_mask)
        X_norm = self.normalizer.transform(self.df)

        self.reference = X_norm
        self.missing = induce_missing(df=self.reference, seed=seed, miss_rate=self.miss_rate, restrict_to_observed=False)

        self.original_missingness = compute_missing_rate(self.df)
        self.current_missingness = compute_missing_rate(self.missing)

        logger.info(
            f"\n Dataset shape: {self.reference.shape}"
            f"\n Original missing rate: {self.original_missingness:.2%}"
            f"\n Current missing rate: {self.current_missingness:.2%}"
        )

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
            transpose=self.transposed,
            normalizer=self.normalizer,
        )
        return data