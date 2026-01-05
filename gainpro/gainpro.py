"""
GainPro - Unified command-line interface for GAIN-based imputation methods.

This module provides a consolidated CLI for:
- Basic GAIN imputation (simple Generator + Discriminator)
- Training GAIN-DANN models (with domain adaptation)
- Imputing with trained models
- Downloading and using pre-trained HuggingFace models
- Using simple imputation methods (median, etc.)
"""

import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import time
import cProfile
import pstats

# Optional visualization imports (only needed for certain features)
try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    import umap.umap_ as umap
except ImportError:
    # These are optional dependencies for visualization features
    sns = None
    plt = None
    umap = None

# Model imports - use try/except to handle both relative and absolute imports
try:
    # Try relative imports first (when run as package module)
    from .gain_dann_model import GainDann
    from .model import Network
    from .hypers import Params
    from .output import Metrics
    from .params_gain_dann import ParamsGainDann
    from .data_utils import Data as DataDANN
    from .dataset import Data
    from .train import GainDannTrain
    from . import utils
    from .imputation_management import ImputationManagement
except ImportError:
    # Fall back to absolute imports (when run as script or in test environment)
    from gainpro.gain_dann_model import GainDann
    from gainpro.model import Network
    from gainpro.hypers import Params
    from gainpro.output import Metrics
    from gainpro.params_gain_dann import ParamsGainDann
    from gainpro.data_utils import Data as DataDANN
    from gainpro.dataset import Data
    from gainpro.train import GainDannTrain
    from gainpro import utils
    from gainpro.imputation_management import ImputationManagement
from huggingface_hub import hf_hub_download
from sklearn.preprocessing import LabelEncoder
import sys

import logging
import click
import os

try:
    import inquirer
    INQUIRER_AVAILABLE = True
except ImportError:
    INQUIRER_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_logging(debug: bool):
    """Configure logging level."""
    logger.setLevel(logging.DEBUG if debug else logging.INFO)


def load_model_metadata(checkpoint_dir: str):
    """Load model metadata from checkpoint directory."""
    json_path = os.path.join(checkpoint_dir, "metadata.json")
    if not os.path.exists(json_path):
        raise click.ClickException(f"Metadata file not found: {json_path}")
    
    try:
        with open(json_path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise click.ClickException(f"Error decoding JSON: {e}")


def load_trained_model(checkpoint_dir: str, metadata: dict):
    """Load a trained GAIN-DANN model from checkpoint."""
    gain_params = Params()
    gain_metrics = Metrics(gain_params)
    
    dann_params = {
        "hidden_dim": metadata["params"]["hidden_dim"],
        "dropout_rate": metadata["params"]["dropout_rate"]
    }
    
    model = GainDann(
        metadata["protein_names"],
        metadata["input_dim"],
        latent_dim=metadata["latent_dim"],
        n_class=metadata["n_class"],
        num_hidden_layers=metadata["params"]["num_hidden_layers"],
        dann_params=dann_params,
        gain_params=gain_params,
        gain_metrics=gain_metrics
    )
    
    model_path = os.path.join(checkpoint_dir, "model.pt")
    if not os.path.isfile(model_path):
        raise click.ClickException(f"Model file not found: {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model


# Click group for main command
@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug logging",
)
@click.pass_context
def cli(ctx, debug):
    """
    GainPro - Unified GAIN-based imputation tools for proteomics data.
    
    This tool provides various imputation methods:
    - Basic GAIN (Generator + Discriminator)
    - GAIN-DANN (Domain-adaptive with Encoder/Decoder)
    - Pre-trained HuggingFace models
    - Simple statistical methods (median)
    """
    ctx.ensure_object(dict)
    ctx.obj['debug'] = debug
    setup_logging(debug)


@cli.command()
@click.option(
    "--config",
    "config_file",
    type=click.Path(exists=True),
    default="configs/params_gain_dann.json",
    help="Path to GAIN-DANN configuration file",
    show_default=True,
)
@click.option(
    "--save/--no-save",
    default=False,
    help="Save trained model checkpoint",
)
def train(config_file, save):
    """
    Train a GAIN-DANN model on your dataset.
    
    Example:
    
        gainpro train --config configs/params_gain_dann.json --save
    """
    logger.info("Starting GAIN-DANN training...")
    
    params_gain_dann = ParamsGainDann.read_hyperparameters(config_file)
    params = params_gain_dann.to_dict()
    
    # Read dataset - use DataDANN for GAIN-DANN models
    if params_gain_dann.path_dataset_missing:
        logger.info(f"Loading dataset with missing values: {params_gain_dann.path_dataset_missing}")
        dataset_missing = pd.read_csv(params_gain_dann.path_dataset_missing, index_col=0)
        dataset_missing = dataset_missing.iloc[:, 8500:]  # TODO: Remove start_col hardcoding
        data = DataDANN(
            dataset_path=params_gain_dann.path_dataset,
            dataset_missing=dataset_missing,
            start_col=8500
        )
    else:
        data = DataDANN(
            dataset_path=params_gain_dann.path_dataset,
            miss_rate=params["miss_rate"],
            start_col=8500
        )
    
    logger.info(f"Early stop patience: {params_gain_dann['early_stop_patience']}")
    train_obj = GainDannTrain(
        data,
        params,
        early_stop_patience=params_gain_dann["early_stop_patience"],
        save_model=save
    )
    train_obj.train()
    
    click.echo("Training completed successfully!")


@cli.command()
@click.option(
    "--checkpoint",
    "checkpoint_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Path to model checkpoint directory",
)
@click.option(
    "--input",
    "input_file",
    type=click.Path(exists=True),
    required=True,
    help="Path to input CSV file with missing values",
)
@click.option(
    "--output",
    "output_file",
    type=click.Path(),
    default="imputed.csv",
    help="Path to output CSV file",
    show_default=True,
)
@click.option(
    "--miss-rate",
    type=float,
    default=0.1,
    help="Missing rate for evaluation (only used if no missing values in input)",
    show_default=True,
)
def impute(checkpoint_dir, input_file, output_file, miss_rate):
    """
    Impute missing values using a trained GAIN-DANN model.
    
    Example:
    
        gainpro impute --checkpoint checkpoints/2024-01-01_12:00 --input data.csv --output imputed.csv
    """
    logger.info("Loading trained model...")
    
    metadata = load_model_metadata(checkpoint_dir)
    model = load_trained_model(checkpoint_dir, metadata)
    model.eval()
    
    # Read dataset
    df = pd.read_csv(input_file, index_col=0)
    
    # Reduce to common proteins
    common_proteins = set(model.protein_names) & set(df.columns)
    df = df.loc[:, list(common_proteins)]
    logger.info(f"Using {len(common_proteins)} common proteins")
    
    # Create data object - use DataDANN for GAIN-DANN models
    data = DataDANN(dataset=df, miss_rate=miss_rate, start_col=0)
    incomplete_data = data.dataset_missing
    
    # Pad with NaNs for model compatibility
    padded_data = incomplete_data.reindex(columns=model.protein_names)
    
    # Transform to tensor
    model_input = torch.tensor(padded_data.values, dtype=torch.float32)
    
    # Impute
    logger.info("Running imputation...")
    with torch.no_grad():
        imputed_data, _ = model(model_input)
    
    # Retrieve only original proteins
    imputed_data = pd.DataFrame(imputed_data.numpy(), index=incomplete_data.index, columns=model.protein_names)
    imputed_data = imputed_data.loc[:, list(common_proteins)]
    
    # Scale back to real space
    imputed_data_real_values = data.scaler.inverse_transform(imputed_data.values)
    imputed_data_real = pd.DataFrame(
        imputed_data_real_values,
        index=imputed_data.index,
        columns=imputed_data.columns
    )
    
    # Save output
    imputed_data_real.to_csv(output_file)
    click.echo(f"Imputation completed! Results saved to {output_file}")


@cli.command()
@click.option(
    "--input",
    "input_file",
    type=click.Path(exists=True),
    required=True,
    help="Path to input CSV file with missing values",
)
@click.option(
    "--output",
    "output_file",
    type=click.Path(),
    default="imputed.csv",
    help="Path to output CSV file",
    show_default=True,
)
@click.option(
    "--model-id",
    default="QuantitativeBiology/GAIN_DANN_model",
    help="HuggingFace model repository ID (must be a GAIN-DANN model with modeling_gain_dann.py file)",
    show_default=True,
)
def download(input_file, output_file, model_id):
    """
    Download a pre-trained GAIN-DANN model from HuggingFace and perform imputation.
    
    NOTE: This command is specifically designed for GAIN-DANN models. It expects:
    - A config.json file with model configuration
    - A pytorch_model.bin file with model weights
    - A modeling_gain_dann.py file with the model architecture class
    
    The model must implement the GainDANN interface with forward() returning 
    (x_reconstructed, x_domain) tuples. Other model types are not supported.
    
    Example:
    
        gainpro download --input data.csv --output imputed.csv
    """
    logger.info(f"Downloading GAIN-DANN model from HuggingFace: {model_id}")
    logger.warning(
        "NOTE: This command only works with GAIN-DANN models that include "
        "modeling_gain_dann.py. Other model types are not supported."
    )
    
    save_dir = "./GAIN_DANN_model"
    os.makedirs(save_dir, exist_ok=True)
    
    # Download files from HuggingFace
    # NOTE: This is specific to GAIN-DANN model structure
    logger.info("Downloading model files...")
    try:
        config_path = hf_hub_download(
            repo_id=model_id,
            filename="config.json",
            cache_dir=save_dir
        )
        weights_path = hf_hub_download(
            repo_id=model_id,
            filename="pytorch_model.bin",
            cache_dir=save_dir
        )
        model_path = hf_hub_download(
            repo_id=model_id,
            filename="modeling_gain_dann.py",
            cache_dir=save_dir
        )
    except Exception as e:
        raise click.ClickException(
            f"Failed to download GAIN-DANN model files from {model_id}. "
            f"Ensure the repository contains config.json, pytorch_model.bin, "
            f"and modeling_gain_dann.py files. Error: {e}"
        )
    
    # Add directory to Python path to import the model
    directory = os.path.dirname(model_path)
    if directory not in sys.path:
        sys.path.append(directory)
    
    # Import model classes (GAIN-DANN specific)
    try:
        from modeling_gain_dann import GainDANNConfig, GainDANN
    except ImportError as e:
        raise click.ClickException(
            f"Failed to import GAIN-DANN model classes from modeling_gain_dann.py. "
            f"This command only works with GAIN-DANN models. Error: {e}"
        )
    
    logger.info("Loading model configuration...")
    
    # Load config
    with open(config_path) as f:
        cfg = json.load(f)
    
    # Read input data
    df = pd.read_csv(input_file)
    
    # Extract data (assuming first column might be index/ID, skip it)
    if df.shape[1] > 1:
        data_df = df.iloc[:, 1:] if 'Project' in df.columns else df
    else:
        data_df = df
    
    input_dim = data_df.shape[1]
    logger.info(f"Input dimension: {input_dim}")
    
    cfg['input_dim'] = input_dim
    config = GainDANNConfig(**cfg)
    model = GainDANN(config)
    
    # Load model weights
    logger.info("Loading model weights...")
    state_dict = torch.load(weights_path, map_location="cpu")
    
    # Add "model." prefix to keys for HuggingFace compatibility
    renamed_state_dict = {f"model.{k}": v for k, v in state_dict.items()}
    model.load_state_dict(renamed_state_dict)
    model.eval()
    
    # Prepare data
    if 'Project' in data_df.columns:
        label_encoder = LabelEncoder()
        data_df['Project'] = label_encoder.fit_transform(data_df['Project'])
    
    x = torch.tensor(data_df.values, dtype=torch.float32)
    
    # Perform imputation (GAIN-DANN specific: returns reconstructed data and domain predictions)
    logger.info("Running imputation...")
    with torch.no_grad():
        try:
            x_reconstructed, x_domain = model(x)
        except (ValueError, TypeError) as e:
            raise click.ClickException(
                f"Model output format not recognized. This command expects GAIN-DANN models "
                f"that return (x_reconstructed, x_domain) tuples. Error: {e}"
            )
    
    # Convert to DataFrame and save
    result_df = pd.DataFrame(x_reconstructed.numpy(), columns=data_df.columns if hasattr(data_df, 'columns') else None)
    
    # If original DataFrame had index, preserve it
    if df.index.name is not None or len(df.index) != len(result_df):
        result_df.index = df.index
    
    result_df.to_csv(output_file, index=True)
    
    # Save domain predictions (GAIN-DANN specific feature)
    domain_file = output_file.replace(".csv", "_domain.csv")
    pd.DataFrame(x_domain.numpy()).to_csv(domain_file, index=False)
    
    click.echo(f"Imputation completed! Results saved to {output_file}")
    click.echo(f"Domain predictions saved to {domain_file}")


@cli.command()
@click.option(
    "--input",
    "input_file",
    type=click.Path(exists=True),
    required=True,
    help="Path to input CSV file with missing values",
)
@click.option(
    "--output",
    "output_file",
    type=click.Path(),
    default="imputed.csv",
    help="Path to output CSV file",
    show_default=True,
)
def median(input_file, output_file):
    """
    Perform median imputation on the dataset.
    
    Replaces missing values with the median of each column.
    
    Example:
    
        gainpro median --input data.csv --output imputed.csv
    """
    logger.info("Performing median imputation...")
    
    try:
        df = pd.read_csv(input_file)
        df_imputed = df.copy()
        
        # Select numeric columns only
        num_cols = df_imputed.select_dtypes(include=[np.number]).columns
        
        # Replace 0 with NaN (common representation of missing values)
        df_imputed[num_cols] = df_imputed[num_cols].astype(float).replace(0, np.nan)
        
        # Calculate column medians
        col_medians = df_imputed[num_cols].median().fillna(0.0)
        
        # Fill missing values with medians
        df_imputed[num_cols] = df_imputed[num_cols].fillna(col_medians)
        
        # Save output
        df_imputed.to_csv(output_file, index=False)
        click.echo(f"Median imputation completed! Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Error during median imputation: {e}")
        raise click.ClickException(f"Imputation failed: {e}")


@cli.command()
@click.option(
    "-i",
    "--input",
    "missing_file",
    type=click.Path(exists=True),
    required=True,
    help="Path to missing data file",
)
@click.option(
    "-o",
    "--output",
    "output_file",
    default="imputed",
    help="Name of output file",
    show_default=True,
)
@click.option(
    "--ref",
    "ref_file",
    type=click.Path(exists=True),
    help="Path to a reference (complete) dataset",
)
@click.option(
    "--ofolder",
    "output_folder",
    default=lambda: os.path.join(os.getcwd(), "results"),
    help="Path to output folder",
    show_default=True,
)
@click.option(
    "--it",
    "num_iterations",
    type=int,
    default=2001,
    help="Number of iterations",
    show_default=True,
)
@click.option(
    "--batchsize",
    "batch_size",
    type=int,
    default=128,
    help="Batch size",
    show_default=True,
)
@click.option(
    "--alpha",
    type=float,
    default=10.0,
    help="Alpha parameter",
    show_default=True,
)
@click.option(
    "--miss",
    "miss_rate",
    type=float,
    default=0.1,
    help="Missing rate",
    show_default=True,
)
@click.option(
    "--hint",
    "hint_rate",
    type=float,
    default=0.9,
    help="Hint rate",
    show_default=True,
)
@click.option(
    "--lrd",
    "lr_D",
    type=float,
    default=0.001,
    help="Learning rate for the discriminator",
    show_default=True,
)
@click.option(
    "--lrg",
    "lr_G",
    type=float,
    default=0.001,
    help="Learning rate for the generator",
    show_default=True,
)
@click.option(
    "--parameters",
    "parameters_file",
    type=click.Path(exists=True),
    help="Load a parameters.json file",
)
@click.option(
    "--override",
    type=int,
    default=0,
    help="Override previous files (1 to override, 0 otherwise)",
    show_default=True,
)
@click.option(
    "--outall",
    "output_all",
    type=int,
    default=0,
    help="Output all files (1 to output all, 0 otherwise)",
    show_default=True,
)
@click.option(
    "--model",
    type=str,
    help="Custom imputation model name (must be registered via ImputationManagement)",
)
def gain(
    missing_file,
    output_file,
    ref_file,
    output_folder,
    num_iterations,
    batch_size,
    alpha,
    miss_rate,
    hint_rate,
    lr_D,
    lr_G,
    parameters_file,
    override,
    output_all,
    model,
):
    """
    Perform basic GAIN (Generative Adversarial Imputation Network) imputation.
    
    This command uses a simple Generator + Discriminator architecture for
    general-purpose missing value imputation. For domain-adaptive imputation,
    use 'gainpro train' and 'gainpro impute' commands.
    
    Examples:
    
        gainpro gain -i data.csv
        gainpro gain -i data.csv --ref reference.csv --it 3000
        gainpro gain --parameters configs/params_gain.json
    """
    start_time = time.time()
    
    with cProfile.Profile() as profile:
        # Load parameters from file if provided
        if parameters_file is not None:
            params = Params.read_hyperparameters(parameters_file)
            missing_file = params.input
            output_file = params.output
            ref_file = params.ref
            output_folder = params.output_folder
            num_iterations = params.num_iterations
            batch_size = params.batch_size
            alpha = params.alpha
            miss_rate = params.miss_rate
            hint_rate = params.hint_rate
            lr_D = params.lr_D
            lr_G = params.lr_G
            override = params.override
            output_all = params.output_all
        else:
            params = Params(
                missing_file,
                output_file,
                ref_file,
                output_folder,
                None,
                num_iterations,
                batch_size,
                alpha,
                miss_rate,
                hint_rate,
                lr_D,
                lr_G,
                override,
                output_all,
            )

        # Create output folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Handle custom model via ImputationManagement
        if model is not None:
            df_missing = pd.read_csv(missing_file)
            imputation_management = ImputationManagement(model, df_missing, missing_file)
            imputation_management.run_model(model)
            click.echo("Imputation completed using custom model.")
            return

        # Load and process input data
        if missing_file.endswith(".csv"):
            df_missing = pd.read_csv(missing_file)
            missing = df_missing.values
            missing_header = df_missing.columns.tolist()
            params.update_hypers(header=missing_header)
        elif missing_file.endswith(".tsv"):
            df_missing = utils.build_protein_matrix(missing_file)
            missing = df_missing.values
            missing_header = df_missing.columns.tolist()
            params.update_hypers(header=missing_header)
        elif missing_file.endswith(".parquet"):
            df_missing = utils.handle_parquet(missing_file)
            missing = df_missing.to_numpy()
            missing_header = df_missing.columns
            params.update_hypers(header=missing_header)
        else:
            raise click.ClickException("Unsupported file format. Supported: .csv, .tsv, .parquet")

        # Build model architecture
        dim = missing.shape[1]
        train_size = missing.shape[0]
        h_dim1 = dim
        h_dim2 = dim

        net_G = nn.Sequential(
            nn.Linear(dim * 2, h_dim1),
            nn.ReLU(),
            nn.Linear(h_dim1, h_dim2),
            nn.ReLU(),
            nn.Linear(h_dim2, dim),
            nn.Sigmoid(),
        )

        net_D = nn.Sequential(
            nn.Linear(dim * 2, h_dim1),
            nn.ReLU(),
            nn.Linear(h_dim1, h_dim2),
            nn.ReLU(),
            nn.Linear(h_dim2, dim),
            nn.Sigmoid(),
        )

        # Initialize network and metrics
        metrics = Metrics(params)
        network = Network(hypers=params, net_G=net_G, net_D=net_D, metrics=metrics)

        # Train with or without reference
        if ref_file is not None:
            logger.info("Training with reference dataset...")
            df_ref = pd.read_csv(ref_file)
            ref = df_ref.values
            ref_header = df_ref.columns.tolist()

            if dim != ref.shape[1]:
                raise click.ClickException(
                    f"Mismatch in number of features: input has {dim}, reference has {ref.shape[1]}"
                )
            elif train_size != ref.shape[0]:
                raise click.ClickException(
                    f"Mismatch in number of samples: input has {train_size}, reference has {ref.shape[0]}"
                )

            data = Data(missing, miss_rate, hint_rate, ref)
            network.train_ref(data, missing_header)
        else:
            logger.info("Training without reference (evaluation + training mode)...")
            data = Data(missing, miss_rate, hint_rate)
            network.evaluate(data, missing_header)
            network.train(data, missing_header)

        # Save execution time
        run_time = time.time() - start_time
        file_path = os.path.join(output_folder, "run_time.csv")

        if override == 1:
            pd.DataFrame([run_time]).to_csv(file_path, index=False)
        else:
            if os.path.exists(file_path):
                with open(file_path, "a") as f:
                    f.write(str(run_time) + "\n")
            else:
                pd.DataFrame([run_time]).to_csv(file_path, index=False)

        click.echo(f"\n--- Execution time: {run_time:.2f} seconds ---\n")
        
        # Save profiling results
        results = pstats.Stats(profile)
        results.sort_stats(pstats.SortKey.TIME)
        results.dump_stats(os.path.join(output_folder, "results.prof"))


# Legacy function for backward compatibility
def list_checkpoints():
    """List available checkpoint directories."""
    checkpoint_root = "checkpoints"
    if not os.path.exists(checkpoint_root):
        return []
    return sorted([
        d for d in os.listdir(checkpoint_root)
        if os.path.isdir(os.path.join(checkpoint_root, d))
    ])


def select_checkpoint_interactively():
    """Interactively select a checkpoint."""
    checkpoints = list_checkpoints()
    if not checkpoints:
        raise FileNotFoundError("No checkpoints found.")
    
    if not INQUIRER_AVAILABLE:
        click.echo("Available checkpoints:")
        for i, cp in enumerate(checkpoints, 1):
            click.echo(f"  {i}. {cp}")
        raise click.ClickException(
            "Please specify checkpoint with --checkpoint flag. "
            "Install 'inquirer' for interactive selection."
        )
    
    question = [
        inquirer.List("timestamp", message="Select a checkpoint to load", choices=checkpoints)
    ]
    answer = inquirer.prompt(question)
    return answer["timestamp"]


# Main entry point
def main():
    """Main entry point for the gainpro CLI."""
    cli()


# Backward compatibility wrapper for 'gain' command
def gain_main():
    """
    Backward compatibility entry point for the deprecated 'gain' command.
    
    This wrapper invokes 'gainpro gain' subcommand to maintain backward compatibility.
    """
    import sys
    
    # Show deprecation warning
    click.echo(
        "⚠️  WARNING: The 'gain' command is deprecated. "
        "Please use 'gainpro gain' instead.\n",
        err=True
    )
    
    # Modify sys.argv to route to the 'gain' subcommand
    # When called as 'gain', sys.argv[0] is the entry point script
    # We need to insert 'gain' as the subcommand argument
    original_argv = sys.argv[:]
    # Insert 'gain' as the first argument (subcommand) if not already present
    if len(sys.argv) > 1 and sys.argv[1] != 'gain':
        sys.argv = [sys.argv[0], 'gain'] + sys.argv[1:]
    elif len(sys.argv) == 1:
        # Just 'gain' was called, show help
        sys.argv = [sys.argv[0], 'gain', '--help']
    # If sys.argv[1] is already 'gain', keep it as is
    
    try:
        cli()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
