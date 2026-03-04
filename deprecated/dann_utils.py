import numpy as np
import pandas as pd
import torch
import json
import os
from datetime import datetime
import logging
from typing import Any, Dict
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_model(model: torch.nn.Module, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "model.pt")
    torch.save(model.state_dict(), path)
    logger.info(f"\nModel saved to {path}")

def save_metadata(metadata: Dict[str, Any], save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    # path = os.path.join(save_dir, f"metadata_{datetime.now().strftime('%d-%m_%H:%M')}.json")
    path = os.path.join(save_dir, f"metadata.json")

    # Convert non-serializable items
    serializable_metadata = {}
    for key, value in metadata.items():
        if hasattr(value, "dict"):  # pydantic models
            serializable_metadata[key] = value.dict()
        elif isinstance(value, (int, float, str, list, dict, bool, type(None))):
            serializable_metadata[key] = value
        else:
            serializable_metadata[key] = str(value)

    with open(path, "w") as f:
        json.dump(serializable_metadata, f, indent=2)
        
    logger.info(f"\nMetadata saved to {path}")

def load_model(model: torch.nn.Module, load_path: str) -> torch.nn.Module:
    model.load_state_dict(torch.load(load_path))
    logger.info(f"\nModel loaded from {load_path}")
    return model

def inverse_transform_output(imputed_data: torch.tensor, scaler) -> torch.Tensor:
    """
    Transforms the imputed data from standardized space back to the original proteomics scale.

    Args:
        - imputed_data (torch.Tensor): The imputed dataset in standardized space.
        - scaler (StandardScaler): The fitted scaler used for normalization.

    Returns:
        - torch.Tensor: The imputed data in the original proteomics scale.
    """
    imputed_data = scaler.inverse_transform(imputed_data)
    return torch.tensor(imputed_data, dtype=torch.float32)

def plot_loss(epochs: list, train_loss, val_loss, save_dir: str, title: str=None):
    plt.figure(figsize=(10,8))
    if title is not None:
        plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Task-Specific Loss')
    plt.plot(epochs, train_loss, label="Train", color="#fc8b64")
    plt.plot(epochs, val_loss, label="Validation", color="#909cc5")
    plt.legend()
    path = os.path.join(save_dir, "imgs")
    os.makedirs(path, exist_ok=True)
    if title is not None:
        plt.savefig(f"{path}/{title}-{datetime.now().strftime('%d-%m_%H:%M')}.png")
    else:
        plt.savefig(f"{path}/loss-{datetime.now().strftime('%d-%m_%H:%M')}.png")
    plt.show() # todo tirar isto daqui
    # plt.close()

def plot_pearson_and_rmse_stats_per_project(pearson_stats: pd.DataFrame, rmse_stats: pd.DataFrame, save_dir: str):
    """
    Plot Pearson correlation and RMSE statistics for each project.
    
    Args:
        - pearson_stats (pd.DataFrame): DataFrame containing Pearson correlation per project.
        - rmse_stats (pd.DataFrame): DataFrame containing RMSE per project.
        - save_dir (str): Directory to store the plot.
    """
    for stats in ["Pearson", "RMSE"]:
        fig, ax = plt.subplots(figsize=(10,8))
        
        if stats == "Pearson":
            bar = ax.barh(pearson_stats["project"], pearson_stats["Pearson mean across folds"], align="center", color="#fc8b64")
    
            for bar, value in zip(bar, pearson_stats["Pearson mean across folds"]):
                ax.text(value + 0.01, bar.get_y() + bar.get_height()/2,
                        f"{value:.3f}", va='center', fontsize=8)
                
            new_labels = [
                f"{proj}\n({cnt} samples)"
                for proj, cnt in zip(pearson_stats["project"], pearson_stats["count"])
            ]

            ax.set_xticks(range(len(new_labels)))
            ax.set_xticklabels(new_labels)
        
        else:
            bar = ax.barh(rmse_stats["project"], rmse_stats["RMSE mean across folds"], align="center", color="#fc8b64")
    
            for bar, value in zip(bar, rmse_stats["RMSE mean across folds"]):
                ax.text(value + 0.01, bar.get_y() + bar.get_height()/2,
                        f"{value:.3f}", va='center', fontsize=8)

        ax.set_title(f"Average {stats} across folds by Project", loc="left")
        ax.set_ylabel("Project")
    
        plt.grid(True, which='major', color='lightgray', linewidth=0.5, axis="x")
        ax.set_axisbelow(True)
        plt.tick_params(axis='both', which='both', direction='out')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.invert_yaxis()

        plt.tight_layout()
        path = os.path.join(save_dir, "imgs")
        os.makedirs(path, exist_ok=True)
        plt.savefig(f"{path}/average-{str.lower(stats)}-across-folds-{datetime.now().strftime('%d-%m_%H:%M')}.png")
        # plt.show() # todo tirar isto daqui
        plt.close()
    
    # Pearson and RMSE all in on plot
    projects = pearson_stats["project"]
    pearson_vals = pearson_stats["Pearson mean across folds"]
    rmse_vals = rmse_stats["RMSE mean across folds"]

    y = np.arange(len(projects))  # numeric positions for projects
    bar_height = 0.4  # total height for each metric's bar

    fig, ax = plt.subplots(figsize=(10, 8))

    # Pearson bars (slightly lower)
    bar1 = ax.barh(y - bar_height/2, pearson_vals, height=bar_height, color="#fc8b64", label="Pearson")

    # RMSE bars (slightly higher)
    bar2 = ax.barh(y + bar_height/2, rmse_vals, height=bar_height, color="#6495ed", label="RMSE")

    # Add values on bars
    for bar, value in zip(bar1, pearson_vals):
        ax.text(value + 0.01, bar.get_y() + bar.get_height()/2, f"{value:.3f}", va='center', fontsize=8)

    for bar, value in zip(bar2, rmse_vals):
        ax.text(value + 0.01, bar.get_y() + bar.get_height()/2, f"{value:.3f}", va='center', fontsize=8)

    ax.axvline(x=1, color='red', linestyle="-")

    # Titles and labels
    ax.set_title("Average Pearson and RMSE across folds by Project")
    ax.set_xlabel("Average value")
    ax.set_ylabel("Project")
    ax.set_yticks(y)
    ax.set_yticklabels(projects)

    # Style
    plt.grid(True, which='major', color='lightgray', linewidth=0.5, axis="x")
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.invert_yaxis()
    ax.legend()
    # for i in range(int(max(max(rmse_vals), max(pearson_vals))*10)+1):
    #     if i % 2 == 0:  # alternate
    #         ax.axvspan(i*0.1, (i+1)*0.1, facecolor="lightgray", alpha=0.2)

    # Save
    plt.tight_layout()
    path = os.path.join(save_dir, "imgs")
    os.makedirs(path, exist_ok=True)
    plt.savefig(f"{path}/average-pearson-rmse-across-folds-{datetime.now().strftime('%d-%m_%H:%M')}.png")
    # plt.show() # todo tirar isto daqui
    plt.close()