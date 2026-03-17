import torch
import torch.nn as nn

from utils.model_hypers import ModelHypers
from models.GainPro.generator import Generator
from models.GainPro.discriminator import Discriminator

class Gain(nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_hypers: ModelHypers,
    ) -> "Gain":
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim

        self.generator = Generator(
            input_dim=self.input_dim,
            num_hidden_layers=model_hypers.generator_n_hidden_layers
        )
        self.generator.to(device=self.device)

        self.discriminator = Discriminator(
            input_dim=self.input_dim,
            num_hidden_layers=model_hypers.discriminator_n_hidden_layers
        )
        self.discriminator.to(device=self.device)

        # self._init_weights()

    def _init_weights(self):
        for name, param in self.generator.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)

        for name, param in self.discriminator.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)

    def forward(
        self,
        x
    ):
        pass