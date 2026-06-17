import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from typing import Optional

from utils.helper import load_yaml
from utils.data.dataset import Data
from utils.data.normalizer_registry import get_output_activation

from wrappers.imputer import Imputer

from models.Gain.gain import Gain

from utils.configs.model_config import GainConfig, FillStrategy
from utils.configs.model_entry_config import ModelEntryConfig
from utils.configs.training_config import GainTrainingConfig
from utils.writers.experiment_writer import ExperimentWriter

class GainImputer(Imputer):
    def __init__(
        self,
        input_dim: int,
        gain_hypers: GainConfig,
        generator_output_activation: nn.Module,
        training_cfg: GainTrainingConfig,
    ) -> None:
        super().__init__()
        self.gain = Gain(
            input_dim=input_dim,
            hypers=gain_hypers,
            training_cfg=training_cfg,
            generator_output_activation=generator_output_activation,
        )
        self.training_cfg=training_cfg
    
    @classmethod
    def from_config(
        cls,
        cfg: GainConfig,
        data: Data,
    ) -> "GainImputer":
        return cls(
            input_dim=data.input_dim,
            gain_hypers=GainConfig.model_validate(load_yaml(cfg.model_cfg_path)),
            training_cfg=GainTrainingConfig.model_validate(load_yaml(cfg.training_cfg_path)),
            generator_output_activation=get_output_activation(data.normalizer)
        )
    
    @classmethod
    def get_fill_strategy(
        self, 
        model_entry: ModelEntryConfig,
    ) -> FillStrategy:
        cfg = GainConfig.model_validate(load_yaml(model_entry.model_cfg_path))
        return cfg.fill_strategy  # default "zero" defined in GainConfig

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
        artificial_mask_val: torch.Tensor,
    ) -> None:
        _ = experiment_writer
        dataset = TensorDataset(x_true, x_train, mask_train, artificial_mask_train)
        train_loader = DataLoader(dataset, batch_size=self.training_cfg.batch_size, shuffle=True)

        val_loader = None
        if x_val is not None:
            val_dataset = TensorDataset(x_true_val, x_val, mask_val, artificial_mask_val)
            val_loader = DataLoader(val_dataset, batch_size=self.training_cfg.batch_size, shuffle=False)

        self.gain.fit(train_loader=train_loader, val_loader=val_loader)
    
    def impute(
        self,
        x_missing: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        test_dataset = TensorDataset(x_missing, mask)
        test_loader = DataLoader(test_dataset, batch_size=self.training_cfg.batch_size, shuffle=False)
        x_imputed = self.gain.impute(test_loader)
        return x_imputed
    
    def predict(
        self,
        x_missing: torch.Tensor,
        observed_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        test_dataset = TensorDataset(x_missing, observed_mask)
        test_loader = DataLoader(test_dataset, batch_size=self.training_cfg.batch_size, shuffle=False)
        x_pred = self.gain.predict(test_loader)
        return x_pred
    
    def evaluate_discriminator(
        self,
        x_missing: torch.Tensor,
        mask_eval: torch.Tensor,
        mask_observed: torch.Tensor,
        positive_label: int, 
    ) -> dict[str, float]:
        results = self.trainer.compute_precision_recall_discriminator(
            x=x_missing, 
            mask=mask_eval,
            mask_observed=mask_observed,
            positive_label=positive_label,             
        )
        return results