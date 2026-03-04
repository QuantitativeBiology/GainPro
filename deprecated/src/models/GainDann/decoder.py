import torch
import torch.nn as nn

class Decoder(nn.Module):
    """Feedforward decoder that maps the latent representation into the input space of the data"""
    def __init__(
        cls, 
        latent_dim: int, 
        hidden_dim: int, 
        target_dim: int, 
        dropout_rate: float, 
        num_hidden_layers: int
    ):
        super(Decoder, cls).__init__()
        cls.layers = nn.ModuleList()

        # Input Layer
        cls.layers.append(nn.Linear(latent_dim, hidden_dim, dtype=torch.float32))
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
        cls.layers.append(nn.Linear(hidden_dim, target_dim, dtype=torch.float32))

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
