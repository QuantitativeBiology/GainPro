import errno
import os
from pathlib import Path
import yaml
import torch
import numpy as np
import logging
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

from models.GainDann.gain_dann import GAIN_DANN
from src.models.hypers import Hypers
from src.utils.data.preprocessing_gaindann import GainDannPreprocessor
from src.utils.data.data_utils import Data
from train.trainer import Trainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_config(config: dict) -> None:
    required_keys = {
        "data": dict,
        "train": dict,
        "model": dict,
        "gain": dict,
    }
    for key, t in required_keys.items():
        if key not in config.keys():
            raise Exception(f"{key} is missing")
        if not isinstance(config[key], t):
            raise TypeError(f"Expected {key} to be {t.__name__}, instead is {type(config[key])}")
    
    inner_required_keys = {
        "data": {
            "raw_path": str,
            "prepared_dir": str,
        },
        "train": {
            "epochs": int,
            "early_stop_patience": int,
            "batch_size": int,
            "learning_rate": float,
            "weight_decay": float,
        },
        "model": {
            "hidden_layers": int,
            "dropout_rate": float,
            "alpha_weight": float,
            "beta_weight": float,
            "gamma_weight": float,
        },
        "gain": {
            "miss_rate": float,
            "hint_rate": float,
        }
    }

    for outer_key in inner_required_keys.keys():
        for inner_key, t in inner_required_keys[outer_key].items():
            if inner_key not in inner_required_keys[outer_key].keys():
                raise Exception(f"{outer_key}{inner_key} is missing")
            if outer_key == "data" and (inner_key == "raw_path" or inner_key == "prepared_dir"):
                # todo one of those can be null but not at the same time
                if config[outer_key][inner_key] is None:
                    continue
            if not isinstance(config[outer_key][inner_key], t):
                raise TypeError(f"Expected {inner_key} to be {t}, instead is {type(config[outer_key][inner_key])}")

def read_config(config_path: str) -> dict:
    config_path = Path(config_path)

    if not config_path.exists() or not config_path.is_file():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), config_path.name)
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError:
        raise yaml.YAMLError(f"{config_path} is an invalid YAML configuration file")

    validate_config(config)
    return config

def run_train(
    config_path: Path,
    save: bool,
    run_dir: Path = None,
) -> None:
    config = read_config(config_path)
    hypers = Hypers(config)

    preprocessor = GainDannPreprocessor(Path(hypers.prepared_dir))
    reference, missing, mask, domain, domain_mapped = preprocessor.run()
    data = Data(reference, missing, mask, domain, domain_mapped)
    
    # Train and validation sets
    name_dataset = Path(hypers.prepared_dir).parent.stem
    train_idx = np.load(Path(f"data/splits/{name_dataset}/train_idx.npy"))
    val_idx = np.load(Path(f"data/splits/{name_dataset}/val_idx.npy"))

    reference_train = data.reference.iloc[train_idx]
    missing_train = data.missing.iloc[train_idx]
    mask_train = data.mask.iloc[train_idx]
    domain_mapped_train = data.domain_mapped.iloc[train_idx]

    reference_train = torch.tensor(reference_train.values, dtype=torch.float32)
    missing_train = torch.tensor(missing_train.values, dtype=torch.float32)
    domain_train = torch.tensor(domain_mapped_train.values).flatten()
    mask_train = torch.tensor(mask_train.values, dtype=torch.int)

    reference_val = data.reference.iloc[val_idx]
    missing_val = data.missing.iloc[val_idx]
    mask_val = data.mask.iloc[val_idx]
    domain_mapped_val = data.domain_mapped.iloc[val_idx]

    reference_val = torch.tensor(reference_val.values, dtype=torch.float32)
    missing_val = torch.tensor(missing_val.values, dtype=torch.float32)
    domain_val = torch.tensor(domain_mapped_val.values).flatten()
    mask_val = torch.tensor(mask_val.values, dtype=torch.int)

    train_dataset = TensorDataset(reference_train, missing_train, domain_train, mask_train)
    train_labels = torch.tensor([y for _, _, y, _ in train_dataset]) 
    class_samples_count = torch.bincount(train_labels)
    weights = 1. / class_samples_count
    sample_weights = weights[train_labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=hypers.batch_size, drop_last=True, sampler=sampler)

    val_dataset = TensorDataset(reference_val, missing_val, domain_val, mask_val)
    val_loader = DataLoader(val_dataset, batch_size=hypers.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GAIN_DANN(
        protein_names=data.protein_names, 
        input_dim=data.n_proteins,
        latent_dim=data.n_proteins,
        n_class=data.n_domains,
        hypers=hypers
    )
    model.to(device)

    trainer = Trainer(model=model, hypers=hypers, save_model=save, run_dir=run_dir)
    trainer.fit(train_loader, val_loader)
    trainer.metrics.to_csv(Path(f"{run_dir}/training/metrics.csv"))