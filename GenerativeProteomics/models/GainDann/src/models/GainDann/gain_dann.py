import torch
import torch.nn as nn

from models.GainDann.encoder import Encoder
from models.GainDann.decoder import Decoder
from models.GainDann.grl import GradientReversalLayer
from models.GainDann.domain_classifier import DomainClassifier
from models.GainDann.helper import initialize_gain

class GAIN_DANN(nn.Module):
    def __init__(
        cls,
        protein_names: list[str], 
        input_dim: int, 
        latent_dim: int,
        n_class: int, 
        hypers: dict
    ):
        super(GAIN_DANN, cls).__init__()

        cls._protein_names = protein_names

        cls._input_dim = input_dim
        cls._hidden_dim = input_dim // 2
        cls._latent_dim = latent_dim
        cls._target_dim = input_dim
        cls._n_class = n_class

        cls.encoder = Encoder(
            input_dim=cls._input_dim, 
            hidden_dim=cls._hidden_dim,
            latent_dim=cls._latent_dim, 
            dropout_rate=hypers.dropout_rate, 
            num_hidden_layers=hypers.hidden_layers
        )
        
        # gradient reversal layer
        cls.grl = GradientReversalLayer()

        cls.domain_classifier = DomainClassifier(
            input_dim=cls._latent_dim, 
            hidden_dim=128, 
            n_class=cls._n_class
        )
        
        # gain
        cls.gain = initialize_gain(cls._latent_dim)
        
        cls.decoder = Decoder(
            latent_dim=cls._latent_dim, 
            hidden_dim=cls._hidden_dim, 
            target_dim=cls._target_dim, 
            dropout_rate=hypers.dropout_rate,
            num_hidden_layers=hypers.hidden_layers
        )
    
    def get_protein_names(cls) -> list[str]:
        return cls._protein_names
    
    def get_input_dim(cls) -> int:
        return cls._input_dim
    
    def get_latent_dim(cls) -> int:
        return cls._latent_dim
    
    def get_n_class(cls) -> int:
        return cls._n_class

    def forward(
        cls, 
        x: torch.tensor, 
        mask: torch.tensor, 
        lambd: float=1.0
    ):
        x_zero_filled = x.clone()

        with torch.no_grad():
            x_zero_filled[torch.isnan(x_zero_filled)] = 0

        z = cls.encoder(x_zero_filled)
        z_aux_gain = z.clone().requires_grad_(True)
        with torch.no_grad():
            z_imputed_aux = cls.gain.generate_sample(z_aux_gain, mask)
        z_imputed = z * mask + z_imputed_aux * (1 - mask)
        x_imputed = cls.decoder(z_imputed)

        # if cls.training:
        z_domain = cls.grl(z, lambd)
        y_hat = cls.domain_classifier(z_domain) # logit
        return z, z_imputed_aux, y_hat, x_imputed
        # return x_imputed

    @torch.no_grad()
    def predict(
        cls,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict all entries
        """
        mask = torch.zeros_like(x)  # everything treated as missing, so forward (after generate sample) will impute all entries
        _,_,_, x_imputed = cls.forward(x, mask)
        return x * mask + x_imputed * (1 - mask)

    @torch.no_grad()
    def impute(
        cls,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Impute missing entries
        """
        mask = ~torch.isnan(x)
        mask = mask.int()
        _,_,_, x_imputed = cls.forward(x, mask)
        res = torch.where(mask == 1, x, x_imputed)
        return res
