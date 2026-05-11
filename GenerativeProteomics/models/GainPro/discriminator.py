import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        # tissue_dim: int,
        hidden_dim: int=None,
        num_hidden_layers: int=1,
    ) -> "Discriminator":
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = int(self.input_dim/2) if hidden_dim is None else hidden_dim
        print("Hidden dim", self.hidden_dim)

        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(self.input_dim * 2, self.hidden_dim))
        # self.layers.append(nn.Linear(self.input_dim * 2 + tissue_dim, self.hidden_dim))
        # The discriminator receives the imputed data from the generator, the sample's tissue
        # and the hint.
        # Input = concat(data_imputed, tissue, hint)
        # tissue: one-hot encoding of tissue, shape (batch, tissue_dim)
        # data_imputed: shape (batch, input_dim)
        # hint: shape (batch, input_dim)
        # After concatenation -> (batch, 2 * input_dim + tissue_dim)
        self.layers.append(nn.LeakyReLU())

        for _ in range(num_hidden_layers):
            self.layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.layers.append(nn.LeakyReLU())
        
        self.layers.append(nn.Linear(self.hidden_dim, self.input_dim))
    
    def forward(
        self,
        x
    ):
        for layer in self.layers:
            x = layer(x)
        return x