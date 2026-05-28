import torch
from typing import Optional
from torch.utils.data import TensorDataset, DataLoader

from utils.helper import load_yaml
from utils.data.dataset import Data
from wrappers.imputer import Imputer
from models.AutoEncoder.autoencoder import AutoEncoder
from utils.configs.model_config import AutoEncoderConfig
from utils.configs.training_config import AutoEncoderTrainingConfig
from utils.writers.experiment_writer import ExperimentWriter

class AutoEncoderImputer(Imputer):
    def __init__(
        self,
        input_dim: int,
        hypers: AutoEncoderConfig,
        training_cfg: AutoEncoderTrainingConfig,
    ) -> "AutoEncoderImputer":
        super().__init__()
        self.ae = AutoEncoder(
            input_dim=input_dim,
            hypers=hypers,
            training_cfg=training_cfg
        )
        self.training_cfg=training_cfg
    
    @classmethod
    def from_config(
        cls,
        cfg: AutoEncoderConfig,
        data: Data,
    ) -> "AutoEncoderImputer":
        return cls(
            input_dim=data.input_dim,
            hypers=AutoEncoderConfig.model_validate(load_yaml(cfg.model_cfg_path)),
            training_cfg=AutoEncoderTrainingConfig.model_validate(load_yaml(cfg.training_cfg_path)),
        )
    
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
        _ = experiment_writer
        dataset = TensorDataset(x_true, x_train, mask_train)
        train_loader = DataLoader(dataset, batch_size=self.training_cfg.batch_size, shuffle=True)

        val_loader = None
        if x_val is not None:
            val_dataset = TensorDataset(x_true_val, x_val, mask_val)
            val_loader = DataLoader(val_dataset, batch_size=self.training_cfg.batch_size, shuffle=False)

        self.ae.fit(train_loader=train_loader, val_loader=val_loader)

    def predict(
        self,
        x_missing: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        _ = mask
        test_loader = DataLoader(x_missing, batch_size=self.training_cfg.batch_size, shuffle=False)
        x_pred = self.ae.predict(test_loader)
        return x_pred