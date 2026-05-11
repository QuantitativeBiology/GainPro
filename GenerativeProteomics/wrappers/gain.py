import torch
import numpy as np

from wrappers.imputer import Imputer
from models.GainPro.gain import Gain
from models.GainPro.trainer import Trainer
from utils.data.dataset import Data
from utils.model_hypers import GainHypers
from utils.train_hypers import TrainHypers
from utils.writers.experiment_writer import ExperimentWriter

class GainImputer(Imputer):
    def __init__(
        self,
        input_dim: int,
        # tissue_dim: int,
        gain_hypers: GainHypers,
        train_hypers: TrainHypers,
    ) -> "GainImputer":
        self.gain = Gain(
            input_dim=input_dim,
            # tissue_dim=tissue_dim,
            hidden_dim=gain_hypers.hidden_dim,
            num_hidden_layers_generator=gain_hypers.num_hidden_layers_generator,
            num_hidden_layers_discriminator=gain_hypers.num_hidden_layers_discriminator
        )
        self.trainer = Trainer(
            model=self.gain,
            train_hypers=train_hypers
        )

    def train(
        self,
        data: Data,
        x_train: torch.tensor,
        mask_train: torch.tensor,
        experiment_writer: ExperimentWriter,
    ) -> None:
        self.trainer.train(
            x_train=x_train,
            observed_mask=mask_train,
            tissue_ids=data.tissue,
            num_tissues=len(data.tissue_mapping),
            experiment_writer=experiment_writer,
        )

    def impute(
        self,
        data: Data,
        x_missing: torch.tensor,
        mask_eval: torch.tensor,
    ) -> np.ndarray:
        x_hat = self.trainer.impute(
            x_missing=x_missing,
            mask=mask_eval,
            tissue_ids=data.tissue,
            num_tissues=len(data.tissue_mapping),
        )
        return x_hat
    
    def evaluate_discriminator(
        self,
        data: Data,
        x_missing: torch.tensor,
        mask_eval: torch.tensor,
        positive_label: int, 
    ) -> dict[str, float]:
        results = self.trainer.compute_precision_recall_discriminator(
            x=x_missing, 
            mask=mask_eval,
            observed_mask=data.observed_mask,
            tissue_ids=data.tissue,
            num_tissues=len(data.tissue_mapping),
            positive_label=positive_label,             
        )
        return results