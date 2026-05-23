import torch
import torch.nn as nn
import logging
from torchinfo import summary

from models.GainPro.generator import Generator
from models.GainPro.discriminator import Discriminator

logger = logging.getLogger(__name__)

class Gain(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int=None,
        num_hidden_layers_generator: int=1,
        num_hidden_layers_discriminator: int=1,
        generator_output_activation: nn.Module=None,
    ) -> "Gain":
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.hidden_dim = int(self.input_dim / 2) if hidden_dim is None else hidden_dim
        logger.debug(
            f"\n Input dim: {self.input_dim}"
            f"\n Hidden dim: {self.hidden_dim}"
        )
        self.num_hidden_layers_generator = num_hidden_layers_generator
        self.num_hidden_layers_discriminator = num_hidden_layers_discriminator

        self.generator = Generator(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_hidden_layers=self.num_hidden_layers_generator,
            generator_output_activation=generator_output_activation,
        )
        self.generator.to(device=self.device)

        self.discriminator = Discriminator(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_hidden_layers=self.num_hidden_layers_discriminator
        )
        self.discriminator.to(device=self.device)

        self._init_weights()

        # logger.debug(f"\n Generator: {summary(self.generator)}")
        # logger.debug(f"\n Discriminator: {summary(self.discriminator)}")

    def _init_weights(self):        
        for m in self.generator.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.discriminator.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_hat = self.generator(x)
        mask_hat = self.discriminator(x)
        return x_hat, mask_hat