# GainPro

[![PyPi Version](https://img.shields.io/pypi/v/gainpro?label=PyPi&color=blue&style=flat&logo=pypi)](https://pypi.org/project/gainpro/)
[![Colab](https://img.shields.io/badge/Google_Colab-0061F2?style=flat&logo=googlecolab&color=blue&label=Colab&colorB=grey)](https://colab.research.google.com/drive/1ihtmsv_UvEz74YrLHZvATu1y2qH4X9-r?usp=sharing)
[![Documentation](https://img.shields.io/badge/docs-read%20the%20docs-blue)](https://generativeproteomics.readthedocs.io/en/latest/)
[![HuggingFace](https://img.shields.io/badge/Hugging_Face-grey?style=flat&logo=huggingface&color=grey)](https://huggingface.co/QuantitativeBiology)

**GainPro** is a PyTorch implementation of Generative Adversarial Imputation Networks (GAIN) [[1]](#1) for imputing missing iBAQ values in proteomics datasets. The package provides a unified command-line interface with multiple imputation methods including basic GAIN, GAIN-DANN (domain-adaptive), and pre-trained GAIN-DANN models from HuggingFace.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Command-Line Usage](#command-line-usage)
- [Python API](#python-api)
- [Repository Structure](#repository-structure)
- [DANN & GAIN Hybrid](#dann--gain-hybrid)
- [References](#references)

## Features

- **Basic GAIN**: Simple Generator + Discriminator architecture for general-purpose imputation
- **GAIN-DANN**: Domain-adaptive imputation with Encoder/Decoder architecture
- **Pre-trained Models**: Easy access to HuggingFace pre-trained GAIN-DANN models
- **Median Imputation**: Simple baseline method
- **Flexible CLI**: Unified `gainpro` command with intuitive subcommands
- **Python API**: Full programmatic access to all functionality


## Installation

### From PyPI (Recommended)

The package is available on PyPI. Install it using:

```bash
pip install gainpro
```

### From Source

1. Clone the repository:
   ```bash
   git clone https://github.com/QuantitativeBiology/GainPro.git
   cd GainPro
   ```

2. Create a Python environment (recommended):
   ```bash
   conda create -n gainpro python=3.10
   conda activate gainpro
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install the package in development mode:
   ```bash
   pip install -e .
   ```

## Quick Start

After installation, you can use the `gainpro` command-line interface:

```bash
# Basic GAIN imputation
gainpro gain -i data.csv

# With reference dataset for evaluation
gainpro gain -i data.csv --ref reference.csv

# Using a configuration file
gainpro gain --parameters configs/params_gain.json
```

## Command-Line Usage

GainPro provides a unified CLI with the following subcommands:


### `gainpro gain` - Basic GAIN Imputation

The basic GAIN command performs imputation using a Generator + Discriminator architecture.

**Basic usage:**
```bash
gainpro gain -i data.csv
```

**With options:**
```bash
gainpro gain -i data.csv -o imputed.csv --ofolder ./results/ --it 3000
```

**Using a configuration file:**
```bash
gainpro gain --parameters configs/params_gain.json
```

**With reference dataset for evaluation:**
```bash
gainpro gain -i data.csv --ref reference.csv
```

**Note:** When run without a reference, the command performs two phases:
1. **Evaluation run**: Conceals a percentage of values (10% by default) during training, calculates RMSE, and creates `test_imputed.csv` for accuracy estimation
2. **Imputation run**: Trains on the entire dataset and creates `imputed.csv`

**Common options:**
- `-i, --input`: Path to input file (CSV, TSV, or Parquet)
- `-o, --output`: Name of output file (default: `imputed`)
- `--ref`: Path to reference (complete) dataset for evaluation
- `--ofolder`: Output folder path (default: `./results`)
- `--it`: Number of training iterations (default: 2001)
- `--batchsize`: Batch size (default: 128)
- `--miss`: Missing rate for evaluation (0-1, default: 0.1)
- `--hint`: Hint rate (0-1, default: 0.9)
- `--lrd`: Learning rate for discriminator (default: 0.001)
- `--lrg`: Learning rate for generator (default: 0.001)
- `--parameters`: Path to JSON configuration file
- `--override`: Override previous output files (1) or append (0, default)
- `--outall`: Output all metrics (1) or minimal output (0, default)

### `gainpro train` - Train GAIN-DANN Model

Train a domain-adaptive GAIN-DANN model:

```bash
gainpro train --config configs/params_gain_dann.json --save
```

### `gainpro impute` - Impute with Trained Model

Use a trained GAIN-DANN checkpoint for imputation:

```bash
gainpro impute --checkpoint checkpoints/your_model --input data.csv --output imputed.csv
```

### `gainpro download` - HuggingFace Pre-trained GAIN-DANN Models

Download and use pre-trained GAIN-DANN models from HuggingFace:

```bash
gainpro download --input data.csv --output imputed.csv
```

**Note:** This command is specifically designed for GAIN-DANN models. It requires the HuggingFace repository to contain:
- `config.json` - Model configuration
- `pytorch_model.bin` - Model weights  
- `modeling_gain_dann.py` - Model architecture file with `GainDANN` class

The model must follow the GAIN-DANN interface (returning `(x_reconstructed, x_domain)` from forward pass). Other model types are not currently supported.

### `gainpro median` - Median Imputation

Simple median imputation baseline:

```bash
gainpro median --input data.csv --output imputed.csv
```

### Getting Help

For detailed help on any command:
```bash
gainpro --help
gainpro gain --help
gainpro train --help
```

**Legacy command:** The `gain` command is still available but deprecated. Use `gainpro gain` instead.


## Python API

GainPro can also be used programmatically through its Python API:

```python
from gainpro import utils, Network, Params, Metrics, Data
import torch
import pandas as pd

# Load your dataset
dataset_path = "your_dataset.tsv"
dataset_df = utils.build_protein_matrix(dataset_path)  # For TSV files
# dataset_df = pd.read_csv(dataset_path)  # For CSV files
dataset = dataset_df.values
missing_header = dataset_df.columns.tolist()

# Define your parameters
params = Params(
    input=dataset_path,
    output="imputed.csv",
    ref=None,
    output_folder=".",
    num_iterations=2001,
    batch_size=128,
    alpha=10,
    miss_rate=0.1,
    hint_rate=0.9,
    lr_D=0.001,
    lr_G=0.001,
    override=1,
    output_all=1,
)

# Define model architecture
input_dim = dataset.shape[1]
h_dim = input_dim
net_G = torch.nn.Sequential(
    torch.nn.Linear(input_dim * 2, h_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim, h_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim, input_dim),
    torch.nn.Sigmoid()
)
net_D = torch.nn.Sequential(
    torch.nn.Linear(input_dim * 2, h_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim, h_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim, input_dim),
    torch.nn.Sigmoid()
)

# Set up the model and data
metrics = Metrics(params)
network = Network(hypers=params, net_G=net_G, net_D=net_D, metrics=metrics)
data = Data(dataset=dataset, miss_rate=0.2, hint_rate=0.9, ref=None)

# Run evaluation and training
network.evaluate(data=data, missing_header=missing_header)
network.train(data=data, missing_header=missing_header)
print("Final Matrix:\n", metrics.data_imputed)
```

For more examples, see the `use-case` directory.

## Repository Structure

Main components of the repository:

- **`.github/workflows`**: CI/CD workflows for automated testing
- **`datasets/`**: Sample datasets with missing values from PRIDE for testing
- **`gainpro/`**: Core package source code
  - `gainpro.py`: Main CLI interface (unified command with subcommands)
  - `model.py`: Basic GAIN model implementation
  - `gain_dann_model.py`: GAIN-DANN model implementation
  - Other core modules (dataset, hypers, output, etc.)
- **`configs/`**: Configuration files for different models
- **`docs/source/`**: Documentation source files for ReadTheDocs
- **`tests/`**: Unit tests to assess model functionality
- **`use-case/`**: Examples demonstrating package usage
  - Installation examples
  - Test execution examples
  - HuggingFace model usage examples

## Demo

The repository includes a breast cancer diagnostic dataset [[2]](#2) in `datasets/breast/`:

- `breast.csv`: Complete dataset
- `breastMissing_20.csv`: Same dataset with 20% missing values
- `parameters.json`: Example configuration file

**Quick demo commands:**

```bash
# Simple imputation
gainpro gain -i ./datasets/breast/breastMissing_20.csv

# With reference for evaluation
gainpro gain -i ./datasets/breast/breastMissing_20.csv --ref ./datasets/breast/breast.csv

# Using configuration file
gainpro gain --parameters ./datasets/breast/parameters.json
```

For detailed metric analysis, either:
- Set `--outall 1` to output all metrics
- Use the Python API in an IPython console to access the `metrics` object (e.g., `metrics.loss_D`, `metrics.loss_G`, `metrics.rmse_train`)


## DANN & GAIN Hybrid

The repository includes a hybrid model combining Domain Adversarial Neural Networks (DANN) with GAIN for domain-adaptive imputation. This is particularly useful when you have multiple datasets from different domains and want to learn domain-invariant representations.

**Training a GAIN-DANN model:**
```bash
gainpro train --config configs/params_gain_dann.json --save
```

**Using a trained model:**
```bash
gainpro impute --checkpoint checkpoints/your_model --input data.csv --output imputed.csv
```

For detailed information about the GAIN-DANN architecture and training procedure, see the documentation.


## References
<a id="1">[1]</a> 
J. Yoon, J. Jordon & M. van der Schaar (2018). GAIN: Missing Data Imputation using Generative Adversarial Nets <br>
<a id="2">[2]</a> 
https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
