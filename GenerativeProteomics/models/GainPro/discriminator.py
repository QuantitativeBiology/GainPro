import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_hidden_layers: int = 1,
    ) -> "Discriminator":
        super().__init__()

        self.input_dim = input_dim

        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(self.input_dim * 2, self.input_dim))
        # The discriminator receives both the imputed data from the generator
        # and the hint.
        # Input = concat(data_imputed, hint)
        # data_imputed    : shape (batch, input_dim)
        # hint            : shape (batch, input_dim)
        # After concatenation -> (batch, 2 * input_dim)
        self.layers.append(nn.ReLU())

        for _ in range(num_hidden_layers):
            self.layers.append(nn.Linear(self.input_dim, self.input_dim))
            self.layers.append(nn.ReLU())
        
        self.layers.append(nn.Linear(self.input_dim, self.input_dim))
        # self.layers.append(nn.Sigmoid())
    
    def forward(
        self,
        x
    ):
        for layer in self.layers:
            x = layer(x)
        return x