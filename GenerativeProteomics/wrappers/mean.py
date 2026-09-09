import logging
import torch
import numpy as np
from typing import Optional, Union

from wrappers.imputer import Imputer
from utils.helper import load_yaml
from utils.data.dataset import Data
from utils.configs.model_config import MeanConfig
from utils.writers.experiment_writer import ExperimentWriter

logger = logging.getLogger(__name__)

class MeanImputer(Imputer):
    """
    Imputation model using protein-wise global or tissue-specific means.
    """
    def __init__(self, cfg: MeanConfig, transpose: bool = False) -> None:
        self.cfg = cfg
        self.transpose = transpose
        self.by_tissue = getattr(cfg, "by_tissue", False)
        
        self.global_means: Optional[np.ndarray] = None
        self.tissue_means: dict[Union[str, int], np.ndarray] = {}

    @classmethod
    def from_config(
        cls,
        cfg: MeanConfig,
        data: Data,
    ):
        return cls(
            cfg=MeanConfig.model_validate(load_yaml(cfg.model_cfg_path)),
            transpose=data.transpose,
        )

    def _compute_means(
        self,
        values: np.ndarray,
        tissue_labels: Optional[np.ndarray] = None,
    ) -> None:
        logger.info(f"\n Transpose: {self.transpose}")
        logger.info(f"\n Values shape: {values.shape}")

        # Axis where feature (protein) means are calculated:
        # Standard matrix: samples in rows (0), proteins in cols (1) -> axis=0
        # Transposed matrix: proteins in rows (0), samples in cols (1) -> axis=1
        calc_axis = 1 if self.transpose else 0

        # Always calculate global means (used directly, or as fallback for tissue means)
        self.global_means = np.nanmean(values, axis=calc_axis)

        # Compute per-tissue means if tissue-wise mode is enabled
        if self.by_tissue and tissue_labels is not None:
            for tissue in np.unique(tissue_labels):
                tissue_mask = tissue_labels == tissue
                tissue_data = values[:, tissue_mask] if self.transpose else values[tissue_mask, :]
                self.tissue_means[tissue] = np.nanmean(tissue_data, axis=calc_axis)

    def _impute_means(
        self,
        values: np.ndarray,
        tissue_labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if self.global_means is None or (self.by_tissue and len(self.tissue_means) == 0):
            raise RuntimeError("MeanImputer must be trained via train() prior to imputation.")

        imputed = values.copy()
        
        protein_axis = 0 if self.transpose else 1

        if self.by_tissue and tissue_labels is not None:
            # Align global means orientation with tissue means if matrix is transposed
            global_means = self.global_means if not self.transpose else self.global_means.T
            for tissue, tissue_mean in self.tissue_means.items():
                tissue_mask = tissue_labels == tissue
                tissue_slice = (slice(None), tissue_mask) if self.transpose else (tissue_mask, slice(None))
                
                tissue_data = imputed[tissue_slice]
                nan_positions = np.isnan(tissue_data)

                # Fallback to global means if a tissue-specific protein mean is NaN
                fill_values = np.where(np.isnan(tissue_mean), global_means, tissue_mean)

                protein_indices = np.where(nan_positions)[protein_axis]
                tissue_data[nan_positions] = np.take(fill_values, protein_indices)
                imputed[tissue_slice] = tissue_data
        else:
            nan_positions = np.isnan(imputed)
            protein_indices = np.where(nan_positions)[protein_axis]
            imputed[nan_positions] = np.take(self.global_means, protein_indices)

        return imputed

    def train(
        self,
        x_train: torch.Tensor,
        x_true: torch.Tensor,
        mask_train: torch.Tensor,
        artificial_mask_train: torch.Tensor,
        experiment_writer: ExperimentWriter,
        x_val: Optional[torch.Tensor],
        x_true_val: Optional[torch.Tensor],
        mask_val: Optional[torch.Tensor],
        artificial_mask_val: Optional[torch.Tensor],
        tissue: Optional[torch.Tensor],
    ) -> None:
        _, _, _ = x_true, mask_train, artificial_mask_train
        _, _, _, _ = x_val, x_true_val, mask_val, artificial_mask_val
        _ = experiment_writer

        tissue_labels = tissue.detach().cpu().numpy() if tissue is not None else None
        self._compute_means(
            values=x_train.detach().cpu().numpy(),
            tissue_labels=tissue_labels,
        )

    def predict(
        self,
        x_missing: torch.Tensor,
        observed_mask: Optional[torch.Tensor],
        tissue: Optional[torch.Tensor],
    ) -> torch.Tensor:
        _ = observed_mask
        input_data = np.full_like(x_missing.detach().cpu().numpy(), fill_value=np.nan)
        tissue_labels = tissue.detach().cpu().numpy() if tissue is not None else None

        x_hat = self._impute_means(values=input_data, tissue_labels=tissue_labels)
        return torch.from_numpy(x_hat).to(device=x_missing.device)

    def impute(
        self,
        x_missing: torch.Tensor,
        mask: Optional[torch.Tensor],
        tissue: Optional[torch.Tensor],
    ) -> torch.Tensor:
        _ = mask
        tissue_labels = tissue.detach().cpu().numpy() if tissue is not None else None

        x_hat = self._impute_means(
            values=x_missing.detach().cpu().numpy(),
            tissue_labels=tissue_labels,
        )
        return torch.from_numpy(x_hat).to(device=x_missing.device)