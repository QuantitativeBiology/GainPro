import torch
import torch.nn as nn

from gainpro.encoder import Encoder
from gainpro.decoder import Decoder
from gainpro.domain_classifier import DomainClassifier
from gainpro.grl import GradientReversalLayer
from gainpro.model import Network
from gainpro.hypers import Params
from gainpro.output import Metrics


#-----------------------------------#
#          DANN GAIN model          #
#-----------------------------------#
class GainDann(nn.Module):
    def __init__(self, protein_names: list[str], 
                input_dim: int, latent_dim: int, n_class: int, num_hidden_layers: int, 
                dann_params: dict, gain_params: Params, gain_metrics: Metrics):
        super(GainDann, self).__init__()

        self.protein_names = protein_names

        self.input_dim = input_dim
        self.hidden_dim = dann_params["hidden_dim"]
        self.latent_dim = latent_dim
        self.target_dim = input_dim
        self.n_class = n_class

        self.encoder = Encoder(input_dim=self.input_dim, hidden_dim=self.hidden_dim, latent_dim=self.latent_dim, 
                               dropout_rate=dann_params["dropout_rate"], num_hidden_layers=num_hidden_layers)
        
        # gradient reversal layer
        self.grl = GradientReversalLayer()

        self.domain_classifier = DomainClassifier(self.latent_dim, n_class=self.n_class)
        
        # gain
        self.gain = Network(hypers=gain_params, 
                            net_G= nn.Sequential(
                                nn.Linear(self.latent_dim * 2, self.latent_dim),
                                nn.ReLU(),
                                nn.Linear(self.latent_dim, self.latent_dim),
                                nn.ReLU(),
                                nn.Linear(self.latent_dim, self.latent_dim),
                                nn.Sigmoid(),
                            ), 
                            net_D= nn.Sequential(
                                nn.Linear(self.latent_dim * 2, self.latent_dim),
                                nn.ReLU(),
                                nn.Linear(self.latent_dim, self.latent_dim),
                                nn.ReLU(),
                                nn.Linear(self.latent_dim, self.latent_dim),
                                nn.Sigmoid(),
                            ),
                            metrics=gain_metrics)
        
        self.decoder = Decoder(latent_dim=self.latent_dim, hidden_dim=self.hidden_dim, target_dim=self.target_dim, 
                               dropout_rate=dann_params["dropout_rate"], num_hidden_layers=num_hidden_layers)



    def forward(self, x: torch.tensor):
        """
            Forward pass of GainDann.
            
            Args:
                - x (torch.tensor): Dataset to be imputed
            
            Returns:
                - torch.tensor: Dataset with the missing values imputed
        """

        # this part is needed, since both the encoder and decoder have
        # a dropout layer and the encoder has also a normalization layer
        # these layers need at least 2 samples in order to perform their
        # calculations. thus we set the model into inference model, disabling
        # these layers by calling `eval()`. besides that we need to call
        # `eval()` on each module individually due to how we train the model
        # from scratch. we needed to train gain from "scratch" and not call its
        # `train()` function.
        self.encoder.eval()
        self.decoder.eval()
        self.domain_classifier.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        x = x.to(device)
        x_filled = x.clone()
        x_filled[torch.isnan(x_filled)] = 0 # x filled with zeros in the place of missing values

        mask = (~torch.isnan(x)).float()
        mask = mask.to(device)

        with torch.no_grad():

            # 1. Encode
            x_encoded = self.encoder(x_filled)
            x_grl = self.grl(x_encoded) # as a matter of fact, this is not needed, this layer is important for the training process

            # 2. Gain
            sample = self.gain.generate_sample(x_grl, mask)
            x_imputed = x_encoded * mask + sample * (1 - mask)

            # 2.1. Domain Classifier
            x_domain = self.domain_classifier(x_encoded)
            x_domain = torch.argmax(x_domain, dim=1)

            # 3. Decoder
            x_hat = self.decoder(x_imputed)
            x_hat = x_hat.detach().cpu()
            
        return x_hat, x_domain

    def __str__(self):
        s = "\n === Gain Dann model ===\n"
        s += f"Input dim: {self.input_dim} \n"
        s += f"Target dim: {self.target_dim} \n"
        s += f"Number of classes: {self.n_class} \n"
        return s