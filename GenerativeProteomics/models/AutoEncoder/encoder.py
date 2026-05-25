import torch
import torch.nn as nn

from models.AutoEncoder.layers import MLP

class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        latent_dim: int,
        use_batch_norm: bool=True,
    ) -> "Encoder":
        super().__init__()

        self.net = MLP(
            layer_sizes=[input_dim] + hidden_dims + [latent_dim],
            use_batch_norm=use_batch_norm,
            activation=nn.LeakyReLU,
            output_activation=nn.LeakyReLU,
        )
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(x)