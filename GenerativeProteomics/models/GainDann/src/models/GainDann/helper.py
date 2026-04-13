import os
import errno
import json
import logging

import torch
import torch.nn as nn

from ..GainPro.model import GainPro
from ..GainPro.output import Metrics
from ..GainPro.hypers import Hypers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_gain(latent_dim: int) -> GainPro:
    gain_params = Hypers()
    gain_metrics = Metrics(gain_params)
    gain = GainPro(hypers=gain_params, 
            net_G= nn.Sequential(
                nn.Linear(latent_dim * 2, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, latent_dim),
                nn.Sigmoid(),
            ),
            net_D= nn.Sequential(
                nn.Linear(latent_dim * 2, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, latent_dim),
                nn.Sigmoid(),
            ),
            metrics=gain_metrics)
    return gain

# def load_model(model_dir: str) -> GAIN_DANN:
#     logger.info("Loading model...")

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     json_path = f"{model_dir}/metadata.json"
#     try:
#         with open(json_path, "r") as f:
#             metadata = json.load(f)
#     except FileNotFoundError:
#         raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), json_path)
#     except json.JSONDecodeError as e:
#         logger.error(f"Error decoding JSON: {e}")
#         raise

#     gain_params = Hypers()
#     gain_metrics = Metrics(gain_params)
#     dann_params = {"hidden_dim": metadata["params"]["hidden_dim"], 
#                     "dropout_rate": metadata["params"]["dropout_rate"]}

#     # load model
#     model = GAIN_DANN(protein_names=metadata.get("protein_names"), 
#                       input_dim=metadata.get("input_dim"), 
#                       latent_dim=metadata.get("latent_dim"), 
#                       n_class=metadata.get("n_class"), 
#                       num_hidden_layers=metadata["params"]["num_hidden_layers"], 
#                       dann_params=dann_params, 
#                       gain_params=gain_params, 
#                       gain_metrics=gain_metrics)
    
#     model_path = f"{model_dir}/model.pt"
#     if not os.path.isfile(model_path):
#         raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), model_path)
#     model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
#     model.to(device)

#     logger.info("Model loaded.")
#     return model