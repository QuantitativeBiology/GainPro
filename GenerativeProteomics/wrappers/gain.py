import torch
import torch.nn as nn
import numpy as np

from utils.helper import load_yaml
from utils.data.normalizer_registry import get_output_activation

from wrappers.imputer import Imputer

from models.GainPro.gain import Gain
from models.GainPro.trainer import Trainer

from utils.configs.model_config import GainConfig
from utils.configs.training_config import GainTrainingConfig
from utils.writers.experiment_writer import ExperimentWriter

class GainImputer(Imputer):
    def __init__(
        self,
        input_dim: int,
        gain_hypers: GainConfig,
        generator_output_activation: nn.Module,
        training_cfg: GainTrainingConfig,
    ) -> "GainImputer":
        self.gain = Gain(
            input_dim=input_dim,
            hidden_dim=gain_hypers.hidden_dim,
            num_hidden_layers_generator=gain_hypers.num_hidden_layers_generator,
            num_hidden_layers_discriminator=gain_hypers.num_hidden_layers_discriminator,
            generator_output_activation=generator_output_activation,
        )
        self.trainer = Trainer(
            model=self.gain,
            training_hypers=training_cfg
        )
    
    @classmethod
    def from_config(
        cls,
        cfg,
        data,
    ) -> "GainImputer":
        return cls(
            input_dim=data.input_dim,
            gain_hypers=GainConfig.model_validate(load_yaml(cfg.model_cfg_path)),
            training_hypers=GainTrainingConfig.model_validate(load_yaml(cfg.training_cfg_path)),
            generator_output_activation=get_output_activation(data.normalizer)
        )

    def train(
        self,
        x_train: torch.Tensor,
        x_true: torch.Tensor,
        mask_train: torch.Tensor,
        experiment_writer: ExperimentWriter,
    ) -> None:
        self.trainer.train(
            x_train=x_train,
            x_true=x_true,
            observed_mask=mask_train,
            experiment_writer=experiment_writer,
        )

    def predict(
        self,
        x_missing: torch.Tensor,
        mask: torch.Tensor,
    ) -> np.ndarray:
        x_hat = self.trainer.impute(
            x_missing=x_missing,
            mask=mask,
        )
        return x_hat
    
    def evaluate_discriminator(
        self,
        x_missing: torch.Tensor,
        mask_eval: torch.Tensor,
        mask_observed: torch.Tensor,
        positive_label: int, 
    ) -> dict[str, float]:
        results = self.trainer.compute_precision_recall_discriminator(
            x=x_missing, 
            mask=mask_eval,
            mask_observed=mask_observed,
            positive_label=positive_label,             
        )
        return results