import logging
import torch
import numpy as np
from typing import Optional

from wrappers.imputer import Imputer
from utils.helper import load_yaml
from utils.data.dataset import Data
from utils.configs.model_config import GlobalMeanConfig
from utils.writers.experiment_writer import ExperimentWriter

logger = logging.getLogger(__name__)

class GlobalMeanImputer(Imputer):
    def __init__(self, cfg: GlobalMeanConfig, transpose: bool=False) -> None:
        self.cfg = cfg
        self.transpose = transpose
        self.protein_means = None

    @classmethod
    def from_config(
        cls,
        cfg: GlobalMeanConfig,
        data: Data,
    ):
        _ = data
        return cls(
            cfg=GlobalMeanConfig.model_validate(load_yaml(cfg.model_cfg_path)),
            transpose=data.transpose,
        )
    
    def _compute_protein_means(
        self,
        values: np.ndarray,
    ) -> None:
        logger.info(f"Transpose: {self.transpose}")
        logger.info(f"\n Values shape: {values.shape}")
        if self.transpose:
            self.protein_means = np.nanmean(values, axis=1)
        else:
            self.protein_means = np.nanmean(values, axis=0)
    
    def _impute_protein_means(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        if self.protein_means is None:
            raise RuntimeError("GlobalMeanImputer must be trained, call train(), before imputation.")
        logger.info(
            f"\n Values shape: {values.shape}"
            f"\n Protein means: {self.protein_means.shape}"
        )
        imputed = values.copy()
        nan_positions = np.isnan(imputed)
        protein_indices = np.where(nan_positions)[0] if self.transpose else np.where(nan_positions)[1]
        imputed[nan_positions] = np.take(self.protein_means, protein_indices)
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
    ) -> None:
        _, _, _ = x_true, mask_train, artificial_mask_train
        _, _, _, _ = x_val, x_true_val, mask_val, artificial_mask_val
        _ = experiment_writer
        self._compute_protein_means(values=x_train.detach().cpu().numpy())

    def predict(
        self,
        x_missing: torch.Tensor,
        observed_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        _ = observed_mask
        input = np.full_like(x_missing.detach().cpu().numpy(), fill_value=np.nan)
        x_hat = self._impute_protein_means(values=input)
        return torch.from_numpy(x_hat).to(device=x_missing.device)
    
    def impute(
        self,
        x_missing: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        _ = mask
        x_hat = self._impute_protein_means(values=x_missing.detach().cpu().numpy())
        return torch.from_numpy(x_hat).to(device=x_missing.device)