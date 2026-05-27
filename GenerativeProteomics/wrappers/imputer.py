import torch
from typing import Optional
from abc import ABC, abstractmethod

from utils.data.dataset import Data
from utils.writers.experiment_writer import ExperimentWriter

class Imputer(ABC):
    @abstractmethod
    def from_config(
        self,
        cfg,
        data: Optional[Data]
    ):
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def predict(
        self,
        x_missing: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        pass