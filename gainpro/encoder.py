import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Feedforward encoder that maps input to latent representation."""
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, dropout_rate: float, num_hidden_layers: int):
        super(Encoder, self).__init__()

        self.layers = nn.ModuleList()

        # Input Layer
        self.layers.append(nn.Linear(input_dim, hidden_dim, dtype=torch.float32))
        self.layers.append(nn.BatchNorm1d(hidden_dim))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout_rate))

        # Hidden Layers
        for i in range(num_hidden_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32))
            self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))
        
        # Output Layer
        self.layers.append(nn.Linear(hidden_dim, latent_dim, dtype=torch.float32))
    
    def forward(self, x):
        # return self.encoder(x)
        for layer in self.layers:
            x = layer(x)
        
        return x