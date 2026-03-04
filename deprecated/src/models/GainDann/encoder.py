import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Feedforward encoder that maps input to a latent representation."""
    def __init__(
        cls, 
        input_dim: int, 
        hidden_dim: int, 
        latent_dim: int, 
        dropout_rate: float, 
        num_hidden_layers: int
    ):
        super(Encoder, cls).__init__()

        cls.layers = nn.ModuleList()

        # Input Layer
        cls.layers.append(nn.Linear(input_dim, hidden_dim, dtype=torch.float32))
        cls.layers.append(nn.BatchNorm1d(hidden_dim))
        cls.layers.append(nn.ReLU())
        cls.layers.append(nn.Dropout(dropout_rate))

        # Hidden Layers
        for i in range(num_hidden_layers):
            cls.layers.append(nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32))
            cls.layers.append(nn.BatchNorm1d(hidden_dim))
            cls.layers.append(nn.ReLU())
            cls.layers.append(nn.Dropout(dropout_rate))
        
        # Output Layer
        cls.layers.append(nn.Linear(hidden_dim, latent_dim, dtype=torch.float32))

        cls.init_weights()
    
    def init_weights(cls) -> None:
        for m in cls.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(cls, x):
        for layer in cls.layers:
            x = layer(x)
        return x