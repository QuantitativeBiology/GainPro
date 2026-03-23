import torch
import torch.nn as nn

from models.GainPro.generator import Generator
from models.GainPro.discriminator import Discriminator

class Gain(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_hidden_layers_generator: int,
        num_hidden_layers_discriminator: int,
    ) -> "Gain":
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.num_hidden_layers_generator = num_hidden_layers_generator
        self.num_hidden_layers_discriminator = num_hidden_layers_discriminator

        self.generator = Generator(
            input_dim=self.input_dim,
            num_hidden_layers=self.num_hidden_layers_generator
        )
        self.generator.to(device=self.device)

        self.discriminator = Discriminator(
            input_dim=self.input_dim,
            num_hidden_layers=self.num_hidden_layers_discriminator
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