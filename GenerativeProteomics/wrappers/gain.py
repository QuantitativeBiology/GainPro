import torch
import numpy as np

from models.GainPro.gain import Gain
from models.GainPro.trainer import Trainer
from utils.model_hypers import GainHypers
from utils.train_hypers import TrainHypers

class GainImputer():
    def __init__(
        self,
        input_dim: int,
        gain_hypers: GainHypers,
        train_hypers: TrainHypers,
    ) -> "GainImputer":
        self.gain = Gain(
            input_dim=input_dim,
            num_hidden_layers_generator=gain_hypers.num_hidden_layers_generator,
            num_hidden_layers_discriminator=gain_hypers.num_hidden_layers_discriminator
        )

        self.trainer = Trainer(
            model=self.gain,
            train_hypers=train_hypers
        )

    def train(
        self,
        x_train: torch.tensor,
        observed_mask: torch.tensor,
        hint_rate: float=0.5,
    ) -> None:
        self.trainer.train(
            x_train=x_train,
            observed_mask=observed_mask,
            hint_rate=hint_rate,
        )

    def impute(
        self,
        x_missing,
        mask,
    ) -> np.ndarray:
        x_hat = self.trainer.impute(
            x_missing=x_missing,
            mask=mask,
        )
        return x_hat