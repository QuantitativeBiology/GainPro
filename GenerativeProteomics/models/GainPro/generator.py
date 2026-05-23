import logging
import torch.nn as nn

logger = logging.getLogger(__name__)

class Generator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int=None,
        num_hidden_layers: int=1,
        generator_output_activation: nn.Module=None,
    ) -> "Generator":
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = int(self.input_dim / 2) if hidden_dim is None else hidden_dim
        logger.debug(
            f"\n Input dim: {self.input_dim}"
            f"\n Hidden dim: {self.hidden_dim}"
        )

        self.layers = nn.ModuleList()

        # The generator receives the corrupted data and the mask.
        # Input = concat(x_noise, mask)
        # x_noise: shape (batch, input_dim)
        # mask: shape (batch, input_dim)
        # After concatenation -> (batch, 2 * input_dim)
        self.layers.append(nn.Linear(self.input_dim * 2, self.hidden_dim))
        self.layers.append(nn.BatchNorm1d(self.hidden_dim))
        self.layers.append(nn.LeakyReLU()) 

        for _ in range(num_hidden_layers):
            self.layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.layers.append(nn.BatchNorm1d(self.hidden_dim))
            self.layers.append(nn.LeakyReLU())
        
        self.layers.append(nn.Linear(self.hidden_dim, self.input_dim))

        # Safely default to Identity if nothing is passed
        self.layers.append(
            generator_output_activation if generator_output_activation is not None else nn.Identity()
        )       
    
    def forward(
        self,
        x
    ):
        for layer in self.layers:
            x = layer(x)
        return x