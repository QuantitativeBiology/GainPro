import torch
import pandas as pd

class Data:
    def __init__(
        self,
        reference: pd.DataFrame,
        missing: pd.DataFrame,
        observed_mask: pd.DataFrame,
        artificial_missing_mask: pd.DataFrame,
        min_norm: pd.DataFrame,
        max_norm: pd.DataFrame,
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

        self.missing = torch.from_numpy(missing.values).to(self.device)
        self.reference = torch.from_numpy(reference.values).to(self.device)
        self.observed_mask = torch.from_numpy(observed_mask.values).to(self.device)
        self.artificial_missing_mask = torch.from_numpy(artificial_missing_mask.values).to(self.device)
        self.tissue = torch.from_numpy(tissue.values.copy()).to(self.device)
        self.tissue_mapping = tissue_mapping

        # Original minimum and maximum values of the dataset before normalization.
        # These are stored so normalized values can later be restored to the
        # original data range (inverse min-max transformation).
        self.min_norm = min_norm
        self.max_norm = max_norm