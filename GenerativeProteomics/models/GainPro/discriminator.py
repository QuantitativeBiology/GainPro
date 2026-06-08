import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int=None,
        num_hidden_layers: int=1,
    ) -> None:
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

        self._init_weights()

    def _init_weights(self) -> None:
        linear_indices = [i for i, m in enumerate(self.layers) if isinstance(m, nn.Linear)]
        last_linear_idx = linear_indices[-1] if linear_indices else None

        for i, m in enumerate(self.layers):
            if isinstance(m, nn.Linear):
                if i == last_linear_idx:
                    nn.init.xavier_normal_(m.weight)
                else:
                    nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        x_imputed: torch.Tensor,
        hint: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat((x_imputed, hint), 1)
        for layer in self.layers:
            x = layer(x)
        mask_hat = x
        return mask_hat