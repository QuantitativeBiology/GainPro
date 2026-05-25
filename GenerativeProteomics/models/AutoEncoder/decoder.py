import torch
import torch.nn as nn

from models.AutoEncoder.layers import MLP

class Decoder(nn.Module):
    def __init__(
        self,
        output_dim: int,
        hidden_dims: list[int],
        latent_dim: int,
        use_batch_norm: bool=True,
    ) -> "Decoder":
        super().__init__()

        self.net = MLP(
            layer_sizes=[latent_dim] + hidden_dims + [output_dim],
            use_batch_norm=use_batch_norm,
            activation=nn.LeakyReLU,
            output_activation=None,
        )
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(x)