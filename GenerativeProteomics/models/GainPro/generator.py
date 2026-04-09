import torch.nn as nn

class Generator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_hidden_layers: int = 1,
    ) -> "Generator":
        super().__init__()

        self.input_dim = input_dim

        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(self.input_dim * 2, self.input_dim))
        # The generator receives both the corrupted data and the mask.
        # Input = concat(data_with_noise, mask)
        # data_with_noise : shape (batch, input_dim)
        # mask            : shape (batch, input_dim)
        # After concatenation -> (batch, 2 * input_dim)
        self.layers.append(nn.LeakyReLU()) 

        for _ in range(num_hidden_layers):
            self.layers.append(nn.Linear(self.input_dim, self.input_dim))
            self.layers.append(nn.LeakyReLU())
        
        self.layers.append(nn.Linear(self.input_dim, self.input_dim))
    
    def forward(
        self,
        x
    ):
        for layer in self.layers:
            x = layer(x)
        return x