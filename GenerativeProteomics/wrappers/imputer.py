import torch
import numpy as np
from abc import ABC, abstractmethod

from utils.writers.experiment_writer import ExperimentWriter

class Imputer(ABC):
    @abstractmethod
    def train(
        self,
        x_train: torch.tensor,
        x_true: torch.tensor,
        mask_train: np.ndarray,
        experiment_writer: ExperimentWriter,
    ) -> None:
        pass

    @abstractmethod
    def impute(
        self,
        x_missing: torch.tensor,
        mask: np.ndarray,
    ) -> np.ndarray:
        pass