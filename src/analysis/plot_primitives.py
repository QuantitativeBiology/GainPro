from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

FIGSIZE = (10,8)
TRAIN_COLOR = "#fc8b64"
VAL_COLOR = "#909cc5"

def plot_domain_accuracies(
    train_domain_accuracies: list, 
    val_domain_accuracies: list,
    run_dir: Path,
) -> None:
    epochs = np.arange(1, len(train_domain_accuracies) + 1)

    plt.figure(figsize=FIGSIZE)
    plt.xlabel('Epoch')
    plt.ylabel('Domain Accuracy')
    plt.plot(epochs, train_domain_accuracies, label="Train", color=TRAIN_COLOR)
    plt.plot(epochs, val_domain_accuracies, label="Validation", color=VAL_COLOR)
    plt.legend()
    # plt.grid(True, which='major', color='lightgray', linewidth=0.5)
    plt.tick_params(axis='both', which='both', direction='out')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)

    plt.savefig(f"{run_dir}/domain_accuracy.png")
    plt.close()

def plot_rmses(
    train_decoder_rmse_loss: list, 
    val_decoder_rmse_loss: list, 
    run_dir: Path,
) -> None:
    epochs = np.arange(1, len(train_decoder_rmse_loss) + 1)

    plt.figure(figsize=FIGSIZE)
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.plot(epochs, train_decoder_rmse_loss, label="Train", color=TRAIN_COLOR)
    plt.plot(epochs, val_decoder_rmse_loss, label="Validation", color=VAL_COLOR)
    plt.legend()
    # plt.grid(True, which='major', color='lightgray', linewidth=0.5)
    plt.tick_params(axis='both', which='both', direction='out')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)
    
      
    plt.savefig(f"{run_dir}/decoder_rmse.png")
    plt.close()

def plot_gain_losses(
    train_gain_mse_losses: list, 
    val_gain_mse_losses: list,
    run_dir: Path,
) -> None:
    epochs = np.arange(1, len(train_gain_mse_losses) + 1)

    plt.figure(figsize=FIGSIZE)
    plt.xlabel('Epoch')
    plt.ylabel('Gain Loss (Imputation latent space)')
    plt.plot(epochs, train_gain_mse_losses, label="Train", color=TRAIN_COLOR)
    plt.plot(epochs, val_gain_mse_losses, label="Validation", color=VAL_COLOR)
    plt.legend()
    # plt.grid(True, which='major', color='lightgray', linewidth=0.5)
    plt.tick_params(axis='both', which='both', direction='out')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)
      
    plt.savefig(f"{run_dir}/gain_losses.png")
    plt.close()

def plot_task_specific_losses(
    train_task_specific_losses: list, 
    val_task_specific_losses: list,
    run_dir: Path,
) -> None:
    epochs = np.arange(1, len(train_task_specific_losses) + 1)

    plt.figure(figsize=FIGSIZE)
    plt.xlabel('Epoch')
    plt.ylabel('Task-Specific Loss')
    # plt.ylabel(r'$\mathcal{L}_\text{Task-Specific}$')
    plt.plot(epochs, train_task_specific_losses, label="Train", color=TRAIN_COLOR)
    plt.plot(epochs, val_task_specific_losses, label="Validation", color=VAL_COLOR)
    plt.legend()
    # plt.grid(True, which='major', color='lightgray', linewidth=0.5)
    plt.tick_params(axis='both', which='both', direction='out')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)
 
    plt.savefig(f"{run_dir}/task_specific_losses.png")
    plt.close()

def plot_domain_adversarial_losses(
    train_domain_classifier_losses: list, 
    val_domain_classifier_losses: list,
    run_dir: Path,
) -> None:  
    epochs = np.arange(1, len(train_domain_classifier_losses) + 1)

    plt.figure(figsize=FIGSIZE)
    plt.xlabel('Epoch')
    plt.ylabel('Adversarial Loss')
    # plt.ylabel(r'$\mathcal{L}_\text{domain}$')
    plt.plot(epochs, train_domain_classifier_losses, label="Train", color=TRAIN_COLOR)
    plt.plot(epochs, val_domain_classifier_losses, label="Validation", color=VAL_COLOR)
    plt.legend()
    # plt.grid(True, which='major', color='lightgray', linewidth=0.5)
    plt.tick_params(axis='both', which='both', direction='out')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)
    
    plt.savefig(f"{run_dir}/domain_adversarial_losses.png")
    plt.close()

def plot_model_losses(
    train_model_losses: list, 
    val_model_losses: list,
    run_dir: Path,
) -> None: 
    epochs = np.arange(1, len(train_model_losses) + 1)

    plt.figure(figsize=FIGSIZE)
    plt.xlabel('Epoch')
    plt.ylabel('Model Loss')
    # plt.ylabel(r'$\mathcal{L}_\mathsf{model}$', fontsize=12)
    plt.plot(epochs, train_model_losses, label="Train", color=TRAIN_COLOR)
    plt.plot(epochs, val_model_losses, label="Validation", color=VAL_COLOR)
    plt.legend()
    # plt.grid(True, which='major', color='lightgray', linewidth=0.5)
    plt.tick_params(axis='both', which='both', direction='out')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)
    
    plt.savefig(f"{run_dir}/model_losses.png")
    plt.close()