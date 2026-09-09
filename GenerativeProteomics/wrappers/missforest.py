import torch
import shutil
import logging
from typing import Optional

from missingpy import MissForest

from utils.helper import load_yaml
from utils.data.dataset import Data
from utils.writers.experiment_writer import ExperimentWriter
from utils.configs.model_config import MissForestConfig

logger = logging.getLogger(__name__)

class MissForestRImputer():
    def __init__(
        self,
        cfg: MissForestConfig,
    ) -> None:
        if shutil.which("R") is None:
            raise RuntimeError(
                "R executable not found. You must install R (https://cran.r-project.org/) "
                "and add it to your system PATH to use this model."
            )

        try:
            self.n_tree = cfg.n_tree
            self.max_iter = cfg.max_iter
            logger.info(
                f"\n Number of trees: {self.n_tree}"
                f"\n Maximum of iterations: {self.max_iter}"
            )

            self.missforest = MissForest(
                n_estimators=self.n_tree,
                max_iter=self.max_iter,
                max_features=None,
                criterion="squared_error" 
            )
        except Exception as e:
            logger.exception("The 'missForest' R package is not installed. Please install it in your R environment.")
            raise

    @classmethod
    def from_config(
        cls,
        cfg: MissForestConfig,
        data: Data,
    ) -> "MissForestRImputer":
        _ = data
        return cls(
            cfg=MissForestConfig.model_validate(load_yaml(cfg.model_cfg_path)),
        )
        
    def train(
        self,
        x_train: torch.tensor,
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
        _, _, _ = x_true, mask_train, artificial_mask_train  # MissForest operates on the full matrix, thus mask is not required
        _, _, _, _ = x_val, x_true_val, mask_val, artificial_mask_val
        _ = experiment_writer
        _ = tissue
        x_train = x_train.detach().cpu().numpy()
        self.missforest = MissForest(
            n_estimators=self.n_tree,
            max_iter=self.max_iter,
            max_features=None,
            criterion=("squared_error")  
        )
        self.missforest.fit(x_train)

    def impute(
        self,
        x_missing: torch.Tensor,
        mask: Optional[torch.Tensor],
        tissue: Optional[torch.Tensor],
    ) -> torch.Tensor:
        _ = mask  # MissForest operates on the full matrix, thus mask is not required
        _ = tissue
        x_missing = x_missing.detach().cpu().numpy()
        x_pred = self.missforest.transform(x_missing)
        return torch.tensor(x_pred, device=x_missing.device)