# Generative Proteomics

todo compare with previous README

## Table of Contents

- [Generative Proteomics](#generative-proteomics)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Quickstart](#quickstart)
  - [Configuration Files](#configuration-files)
    - [Datasets](#datasets)
    - [Models](#models)
    - [Train](#train)
    - [Benchmark](#benchmark)
  - [Imputation Models](#imputation-models)
    - [GAIN](#gain)
    - [Autoencoder](#autoencoder)
    - [MissForest](#missforest)
    - [Tissue Mean \& Global Mean](#tissue-mean--global-mean)
    - [Model Comparison](#model-comparison)
  - [Usage](#usage)
  - [Reproducing Results](#reproducing-results)
  - [Project Structure](#project-structure)
  - [Contributing](#contributing)

## Overview

todo

2–3 sentences explaining: what problem this solves, what approach it takes, and who it is for.

## Installation

```bash
git clone https://github.com/QuantitativeBiology/GainPro.git
cd GainPro

# Create and activate a virtual environment (recommended)
python -m venv .gainpro
source .gainpro/bin/activate  

# Install dependencies
pip install -r requirements.txt
```

**Requirements:** Python $\geq$ 3.10

## Quickstart

```bash
cd GainPro
source .gainpro/bin/activate
cd GenerativeProteomics

# Run benchmark
python main.py benchmark \
    --config ../configs/benchmark/protogain/holdout/benchmark_miss10.yaml \
    --dataset ../configs/datasets/PXD030304/PXD030304_no_control_multi_peptide_50pct_tissue/PXD030304_no_control_multi_peptide_50pct_tissue_transpose.yaml
```

## Configuration Files

All configuration lives in `configs/`. Each file controls a distinct phase of the pipeline.

| File | Purpose | Pipeline phase |
| ------ | --------- | --------- |
| `dataset.yaml` | Dataset path and preprocessing steps | All phases |
| `model.yaml` | Per-model hyperparameters | Train and benchmark |
| `train.yaml` | Training loop: epochs, batch size, optimizer | Train and benchmark |
| `benchmark.yaml` | Evaluation runs: number of runs, seeds, evaluation strategy, model(s) to run | Benchmark |

> **Note:** GAIN requires two additional fields in `train.yaml`: `hint_rate` and `alpha`.

### Datasets

```yaml
# configs/datasets/...
name: string            # Name for readability, not used by the pipeline

dataset_path: string    # Path to the CSV/TSV/anndata file, relative to `GenerativeProteomics`

log_transform: bool     # ⚠️ Apply log-transform before imputation.
                        # You are responsible for ensuring this matches
                        # your data's scale — the model is unaware of it.

normalizer: string      # Options: ["auto", "none", "minmax", "standard"]. 
                        # "auto" lets each model use its default normalizer.
                        # Override with: "minmax", "standard".
```

A full example is available [PXD030304_no_control_multi_peptide_50pct_tissue.yaml](configs/datasets/PXD030304/PXD030304_no_control_multi_peptide_50pct_tissue/PXD030304_no_control_multi_peptide_50pct_tissue.yaml).

> **Default normalizers per model:**
>
> | Model | Default |
> | ------- | --------- |
> | GAIN | minmax |
> | Autoencoder | standard |
> | MissForest | none |
> | Tissue Mean | none |
> | Global Mean | none |

All fields and their types are validated on load via a [Pydantic BaseModel](GenerativeProteomics/utils/configs/dataset_config.py).
Passing an invalid value raises a `ValidationError` before the pipeline starts.

> **Custom normalizers:** extend the base class in [`normalizer.py`](GenerativeProteomics/utils/data/normalizer.py),
> register it in [`normalizer_registry`](GenerativeProteomics/utils/data/normalizer_registry.py), and add the new key to the `normalizer` field in
> [`dataset_config.py`](GenerativeProteomics/utils/configs/dataset_config.py).

### Models

todo

```yaml
```

### Train

todo

```yaml
```

### Benchmark

todo

```yaml
```

## Imputation Models

todo: add model descriptions and overall improvement

### GAIN

Generative Adversarial Imputation Network. GAN-based approach that learns the data distribution.
Used PyTorch Lightning for training.

### Autoencoder

Used PyTorch Lightning for training.

### MissForest

- **Best for:** Smaller datasets

### Tissue Mean & Global Mean

Simple statistical baselines. Tissue mean imputes using the per-tissue (group) average; global mean uses the dataset-wide average.

- **Best for:** Sanity checks, fast baselines

### Model Comparison

| Model | Accuracy | Speed | Mixed types | Notes |
| ------- | ---------- | ------- | ------------- | ------- |
| GAIN | | | ❌ | |
| Autoencoder | | | ❌ | GPU recommended |
| MissForest | | | ✅ | Good default |
| Tissue Mean | | | ✅ | Baseline only |
| Global Mean | | | ✅ | Baseline only |

## Usage

todo: incomplete

```bash
# For a full list of CLI options:
cd GenerativeProteomics
python main.py --help

# For a specific command:
python main.py benchmark --help
```

## Reproducing Results

todo: incomplete

> todo: Pin environment with `pip freeze > requirements-lock.txt` for exact reproducibility.

## Project Structure

todo: incomplete

```text
GainPro/
├── configs/
│   ├── benchmark.yaml
│   ├── datasets.yaml
│   ├── models.yaml
│   └── train.yaml
├── data/
│   └── ...
├── models/
│   ├── GainPro/
│   ├── AutoEncoder/
│   └── baselines.py
├── results/
├── main.py
├── requirements.txt
└── README.md
```

## Contributing

todo: update how

1. Fork the repo and create a feature branch: `git checkout -b feat/your-feature`
2. Commit your changes and open a pull request.
3. For new imputation models, add an entry to `models.yaml` and the [Model Comparison](#model-comparison) table.
