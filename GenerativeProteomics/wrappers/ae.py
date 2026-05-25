import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from utils.helper import load_yaml
from utils.data.helper import convert_tensors_dtype
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
        x_train: torch.tensor,
        x_true: torch.tensor,
        mask_train: np.ndarray,
        experiment_writer: ExperimentWriter,
    ) -> None:
        x_train = convert_tensors_dtype(x_train, dtype=torch.float32)
        dataset = TensorDataset(x_true, x_train, mask_train)
        train_loader = DataLoader(dataset, batch_size=self.training_cfg.batch_size)
        self.ae.fit(train_loader=train_loader)

    def impute(
        self,
        x_missing: torch.tensor,
        mask: np.ndarray,
    ) -> np.ndarray:
        x_missing = convert_tensors_dtype(x_missing, dtype=torch.float32)
        dataset = TensorDataset(x_missing, mask)
        test_loader = DataLoader(dataset, batch_size=self.training_cfg.batch_size)
        x_out = self.ae.predict(test_loader)
        return x_out