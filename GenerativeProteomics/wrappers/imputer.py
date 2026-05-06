import torch
import numpy as np
from abc import ABC, abstractmethod

from utils.data.dataset import Data
from utils.writers.experiment_writer import ExperimentWriter

class Imputer(ABC):
    @abstractmethod
    def train(
        self,
        data: Data,
        x_train: torch.tensor,
        mask_train: np.ndarray,
        experiment_writer: ExperimentWriter,
    ) -> None:
        pass

    @abstractmethod
    def impute(
        self,
        data: Data,
        x_missing: torch.tensor,
        mask_eval: np.ndarray,
    ) -> np.ndarray:
        pass