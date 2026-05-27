import torch
import numpy as np
import pandas as pd

from utils.data.normalizer import Normalizer

class Data:
    def __init__(
        self,
        reference: pd.DataFrame,
        missing: pd.DataFrame,
        observed_mask: pd.DataFrame,
        artificial_missing_mask: pd.DataFrame,
        normalizer: Normalizer,
        log_transform: bool,
        transpose: bool=False,
        tissue: pd.DataFrame=None,
        tissue_mapping: dict=None,
    ) -> "Data":
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transpose = transpose

        if self.transpose:
            self.sample_names = list(reference.columns)
            self.feature_names = list(reference.index)
        else:
            self.sample_names = list(reference.index)
            self.feature_names = list(reference.columns)
        
        if tissue_mapping is not None:
            self.tissue_mapping = tissue_mapping

        self.num_samples = len(self.sample_names)
        self.num_features = len(self.feature_names)
        self.input_dim = reference.shape[1]

        self.missing = torch.from_numpy(missing.values).to(dtype=torch.float32, device=self.device)
        self.reference = torch.from_numpy(reference.values).to(dtype=torch.float32, device=self.device)
        self.observed_mask = torch.from_numpy(observed_mask.values).to(device=self.device)
        self.artificial_missing_mask = torch.from_numpy(artificial_missing_mask.values).to(self.device)
        self.tissue = torch.from_numpy(tissue.values.copy()).to(self.device)
        self.tissue_mapping = tissue_mapping

        self.normalizer = normalizer
        self.log_transform = log_transform

    def denormalize(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        return self.normalizer.inverse_transform(values)
    
    def inverse_log2p1(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """Reverse log2(x + 1) transformation."""
        x_log = np.power(2, x) - 1
        return x_log