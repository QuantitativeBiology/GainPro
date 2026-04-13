import torch
import torch.nn as nn
from pathlib import Path

from GenerativeProteomics.models.GainPro.gainpro_old import GainPro
from hypers import Hypers
from GenerativeProteomics.models.GainPro.metrics import Metrics

def initialize_gain(
    input_dim: int,
    hidden_dim: int,
    header_params,
    missing_file: str, #todo não devia estar a receber informação de i/o, mas por agora fica assim
    output_folder: str,
    output_file: str=None,
    ref_file: str=None,
) -> GainPro:
    net_G = nn.Sequential(
        nn.Linear(input_dim * 2, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, input_dim),
        nn.Sigmoid(),
    )

    net_D = nn.Sequential(
        nn.Linear(input_dim * 2, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, input_dim),
        nn.Sigmoid(),
    )

    dataset_name = Path(missing_file).stem

    params = Hypers(
        input=missing_file,
        output=dataset_name,
        ref=ref_file,
        output_folder=output_folder,
        header=None,
        num_iterations=2001,
        batch_size=128,
        alpha=10,
        miss_rate=0.1,
        hint_rate=0.9,
        lr_D=0.001,
        lr_G=0.001,
        override=0,
        output_all=0,
    )
    params.update_hypers(header=header_params)
    metrics = Metrics(params)

    model = GainPro(hypers=params, net_G=net_G, net_D=net_D, metrics=metrics)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return model
