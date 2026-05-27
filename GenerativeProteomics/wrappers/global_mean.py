import torch
from typing import Optional
import numpy as np

from wrappers.imputer import Imputer
from utils.helper import load_yaml
from utils.data.dataset import Data
from utils.configs.model_config import GlobalMeanConfig
from utils.writers.experiment_writer import ExperimentWriter

class GlobalMeanImputer(Imputer):
    def __init__(self, cfg: GlobalMeanConfig) -> None:
        self.cfg = cfg
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
        )
    
    def _compute_protein_means(
        self,
        values: np.ndarray,
    ) -> None:
        self.protein_means = np.nanmean(values, axis=0)
    
    def _impute_protein_means(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        if self.protein_means is None:
            raise RuntimeError("GlobalMeanImputer must be trained, train(), before imputation.")
        imputed = values.copy()
        nan_positions = np.isnan(imputed)
        imputed[nan_positions] = np.take(self.protein_means, np.where(nan_positions)[1])
        return imputed
    
    def train(
        self,
        x_train: torch.Tensor,
        x_true: torch.Tensor,
        mask_train: torch.Tensor,
        experiment_writer: ExperimentWriter,
        x_val: Optional[torch.Tensor],
        x_true_val: Optional[torch.Tensor],
        mask_val: Optional[torch.Tensor],
    ) -> None:
        _, _, _, _, _, _ = x_true, mask_train, x_val, x_true_val, mask_val
        _ = experiment_writer
        self._compute_protein_means(values=x_train.detach().cpu().numpy())
    
    def predict(
        self,
        x_missing: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        _ = mask
        x_hat = self._impute_protein_means(values=x_missing.detach().cpu().numpy())
        return torch.from_numpy(x_hat).to(device=x_missing.device)