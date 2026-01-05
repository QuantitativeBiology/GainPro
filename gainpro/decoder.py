import torch
import torch.nn as nn


class Decoder(nn.Module):
    """Feedforward decoder that maps the latent representation into the input space of the data"""
    def __init__(self, latent_dim: int, hidden_dim: int, target_dim: int, dropout_rate: float, num_hidden_layers: int):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()

        # Input Layer
        self.layers.append(nn.Linear(latent_dim, hidden_dim))
        self.layers.append(nn.BatchNorm1d(hidden_dim))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout_rate))

        # Hidden Layers
        for i in range(num_hidden_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))
        
        # Output Layer
        self.layers.append(nn.Linear(hidden_dim, target_dim))


    def forward(self, x):
        # return self.decoder(x)
        for layer in self.layers:
            x = layer(x)
        
        return x
