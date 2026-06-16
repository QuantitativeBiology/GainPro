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
        artificial_mask_train: torch.Tensor,
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
        observed_mask: Optional[torch.Tensor],
    ) -> torch.Tensor: 
        """Generate predictions for all entries in the input tensor.

        Args:
            x_missing (torch.Tensor): Input tensor of shape (batch_size, features)
                where missing positions contain placeholder values according with the `fill_strategy`.
            observed_mask (Optional[torch.Tensor]): Boolean tensor of the same shape
                as `x_missing`, where `True` indicates observed positions and `False` indicates 
                missing (unknown) positions.

        Returns:
            torch.Tensor: Output tensor of the same shape as `x_missing` containing
                predictions for all entries, including both observed and missing positions.
        """
        pass

    @abstractmethod
    def impute(
        self,
        x_missing: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Impute missing entries in the input tensor.

        Args:
            x_missing (torch.Tensor): Input tensor of shape (batch_size, features)
                where missing positions contain placeholder values according with 
                the `fill_strategy`.
            observed_mask (Optional[torch.Tensor]): Boolean tensor of the same shape
                as `x_missing`, where `True` indicates observed positions and `False` 
                indicates missing (unknown) positions.

        Returns:
            torch.Tensor: Output tensor of the same shape as `x_missing` where
                missing positions are replaced with predicted values and observed
                positions remain unchanged.
        """
        pass