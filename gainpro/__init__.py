"""
GainPro
=======

A PyTorch implementation of Generative Adversarial Imputation Networks (GAIN)
for imputing missing values in proteomics datasets.

Main Classes
------------
- Data: Handles datasets with missing values, preprocessing, masking, and scaling
- Params: Manages hyperparameters for model training
- Network: Core GAIN model architecture and training logic
- Metrics: Tracks performance metrics during training and evaluation
- ImputationManagement: Factory for managing different imputation strategies

Example Usage
-------------
>>> from gainpro import Data, Params, Network, Metrics
>>> import torch
>>> import pandas as pd
>>>
>>> # Load dataset
>>> dataset_df = pd.read_csv("your_dataset.csv")
>>> dataset = dataset_df.values
>>>
>>> # Configure parameters
>>> params = Params(
...     input="your_dataset.csv",
...     output="imputed.csv",
...     num_iterations=2001,
... )
>>>
>>> # Set up and train model
>>> data = Data(dataset=dataset, miss_rate=0.2, hint_rate=0.9)
>>> # ... define net_G, net_D ...
>>> network = Network(hypers=params, net_G=net_G, net_D=net_D, metrics=Metrics(params))
>>> network.train(data=data, missing_header=dataset_df.columns.tolist())
"""

__version__ = "0.2.0"
__author__ = "QuantitativeBiology"

# Core classes
from .dataset import Data
from .hypers import Params
from .model import Network
from .output import Metrics
from .imputation_management import ImputationManagement

# Utilities
from . import utils

# GAIN-DANN components (optional, for advanced usage)
from .gain_dann_model import GainDann
from .params_gain_dann import ParamsGainDann

__all__ = [
    # Core classes
    "Data",
    "Params", 
    "Network",
    "Metrics",
    "ImputationManagement",
    # Utilities
    "utils",
    # GAIN-DANN
    "GainDann",
    "ParamsGainDann",
    # Version info
    "__version__",
    "__author__",
]
