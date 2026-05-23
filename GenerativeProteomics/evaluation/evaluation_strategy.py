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