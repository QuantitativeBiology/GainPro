import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int=None,
        num_hidden_layers: int=1,
    ) -> "Discriminator":
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = int(self.input_dim / 2) if hidden_dim is None else hidden_dim

        self.layers = nn.ModuleList()

        # The discriminator receives the imputed data from the generator and the hint.
        # Input = concat(x_hat, hint)
        # x_hat: shape (batch, input_dim)
        # hint: shape (batch, input_dim)
        # After concatenation -> (batch, 2 * input_dim)
        self.layers.append(nn.Linear(self.input_dim * 2, self.hidden_dim))
        self.layers.append(nn.BatchNorm1d(self.hidden_dim))
        self.layers.append(nn.LeakyReLU())

        for _ in range(num_hidden_layers):
            self.layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.layers.append(nn.BatchNorm1d(self.hidden_dim))
            self.layers.append(nn.LeakyReLU())
        
        self.layers.append(nn.Linear(self.hidden_dim, self.input_dim))
    
    def forward(
        self,
        x: torch.Tensor,
        hint: torch.Tensor,
    ) -> torch.Tensor:
        input = torch.cat((x, hint), 1).float()
        x = input
        for layer in self.layers:
            x = layer(x)
        mask_hat = x
        return mask_hat