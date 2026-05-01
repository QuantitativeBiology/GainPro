import torch
import numpy as np
import pandas as pd

class Data:
    def __init__(
        self, 
        miss_rate: float,
        hint_rate: float,
        reference: pd.DataFrame,
        missing: pd.DataFrame,
        observed_mask: pd.DataFrame,
        artificial_missing_mask: pd.DataFrame,
        min_norm: pd.DataFrame,
        max_norm: pd.DataFrame,
        # col_scalers: dict,
        tissue: pd.DataFrame = None,
        tissue_mapping: dict = None,
        cell_line: pd.DataFrame = None,
        cell_line_mapping: dict = None
    ) -> "Data":
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.sample_names = list(reference.index)
        self.feature_names = list(reference.columns)
        
        if tissue_mapping is not None:
            self.tissue_mapping = tissue_mapping

        self.missing = torch.from_numpy(missing.values).to(self.device)
        self.reference = torch.from_numpy(reference.values).to(self.device)
        self.observed_mask = torch.from_numpy(observed_mask.values).to(self.device) # on all observed values
        self.artificial_missing_mask = torch.from_numpy(artificial_missing_mask.values).to(self.device) # on artificially masked values
        self.cell_line = torch.from_numpy(cell_line.values.copy()).to(self.device)
        self.cell_line_mapping = cell_line_mapping
        self.tissue = torch.from_numpy(tissue.values.copy()).to(self.device)
        self.tissue_mapping = tissue_mapping

        self.min_norm = min_norm
        self.max_norm = max_norm
        # self.col_scalers = col_scalers

        self.miss_rate = miss_rate
        self.hint_rate = hint_rate

        hint = generate_hint(observed_mask, self.hint_rate)
        self.hint = torch.from_numpy(hint.values).to(self.device)

def generate_hint(observed_mask, hint_rate):
    hint_mask = generate_mask(observed_mask, 1 - hint_rate)
    hint = observed_mask * hint_mask
    return hint

def generate_mask(data, miss_rate):
    dim = data.shape[1]
    size = data.shape[0]
    A = np.random.uniform(0.0, 1.0, size=(size, dim))
    B = A > miss_rate
    observed_mask = 1.0 * B
    return observed_mask