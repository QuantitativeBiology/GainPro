import numpy as np
from abc import ABC, abstractmethod

from utils.data.dataset import Data
from utils.writers.experiment_writer import ExperimentWriter

class EvaluationStrategy(ABC):
    @abstractmethod
    def run(
        self,
        imputer_factory,
        data: Data,
        experiment_writer: ExperimentWriter,
        **kwargs
    ) -> None:
        pass
    
    def _compute_rmse(
        self,
        x_true: np.ndarray,
        x_pred: np.ndarray,
        mask: np.ndarray,
    ) -> float:
        """Compute RMSE between true and predicted values on masked positions."""
        mse = np.sum((x_true * mask - x_pred * mask) ** 2) / np.sum(mask)
        return np.sqrt(mse)
    
    def _denormalize(
        self,
        x: np.ndarray,
        max_norm: np.ndarray,
        min_norm: np.ndarray,
    ) -> np.ndarray:
        """Reverse min-max normalization."""
        return x * (max_norm - min_norm) + min_norm
    
    def _inverse_log2p1(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """Reverse log2(x + 1) transformation."""
        x_log = np.power(2, x) - 1
        return x_log