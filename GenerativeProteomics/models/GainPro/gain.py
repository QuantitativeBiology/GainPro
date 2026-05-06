import torch
import torch.nn as nn

from models.GainPro.generator import Generator
from models.GainPro.discriminator import Discriminator

class Gain(nn.Module):
    def __init__(
        self,
        input_dim: int,
        tissue_dim: int,
        hidden_dim: int=None,
        num_hidden_layers_generator: int=1,
        num_hidden_layers_discriminator: int=1,
    ) -> "Gain":
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.tissue_dim = tissue_dim
        self.hidden_dim = self.input_dim if hidden_dim is None else hidden_dim
        self.num_hidden_layers_generator = num_hidden_layers_generator
        self.num_hidden_layers_discriminator = num_hidden_layers_discriminator

        self.generator = Generator(
            input_dim=self.input_dim,
            tissue_dim=self.tissue_dim,
            hidden_dim=self.hidden_dim,
            num_hidden_layers=self.num_hidden_layers_generator
        )
        self.generator.to(device=self.device)

        self.discriminator = Discriminator(
            input_dim=self.input_dim,
            tissue_dim=self.tissue_dim,
            hidden_dim=self.hidden_dim,
            num_hidden_layers=self.num_hidden_layers_discriminator
        )
        self.discriminator.to(device=self.device)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.generator.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)

        for name, param in self.discriminator.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)

        print("After weight initialization")
        for name, p in self.discriminator.named_parameters():
            print(f"  D {name} | mean: {p.data.mean():.6f} | std: {p.data.std():.6f}")
            break
        for name, p in self.generator.named_parameters():
            print(f"  G {name} | mean: {p.data.mean():.6f} | std: {p.data.std():.6f}")
            break

    def forward(
        self,
        x
    ):
        pass