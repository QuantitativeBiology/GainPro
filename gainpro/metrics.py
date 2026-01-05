import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MetricsTracker:
    """
    Tracks metrics across epochs and folds.
    Provides logging, averaging, and plotting helpers.
    """
    def __init__(self, name="metrics"):
        self.name = name
        self.metrics = { # metrics per epoch
            "loss_train": [],
            "loss_val": [],
            "rmse_train": [],
            "rmse_val": [],
            "domain_acc_train": [],
            "domain_acc_val": [],

            "loss_gain_train": [],
            "loss_gain_val": [],

            "loss_domain_classifier_train": [],
            "loss_domain_classifier_val": [],

            "loss_model_train": [],
            "loss_model_val": [],
        }
        self.fold_history = []  # store per-fold histories if needed
        self.avg_metrics = dict()

    def update(self, mode: str, 
               loss=None, loss_gain=None, loss_domain_classifier=None, loss_model=None,
               rmse=None, domain_acc=None):
        """Update metrics for the current epoch."""
        if mode == "Train":
            if loss is not None: self.metrics["loss_train"].append(loss)
            if rmse is not None: self.metrics["rmse_train"].append(rmse)
            if domain_acc is not None: self.metrics["domain_acc_train"].append(domain_acc)
            if loss_gain is not None: self.metrics["loss_gain_train"].append(loss_gain)
            if loss_domain_classifier is not None: self.metrics["loss_domain_classifier_train"].append(loss_domain_classifier)
            if loss_model is not None: self.metrics["loss_model_train"].append(loss_model)
        else:
            if loss is not None: self.metrics["loss_val"].append(loss)
            if rmse is not None: self.metrics["rmse_val"].append(rmse)
            if domain_acc is not None: self.metrics["domain_acc_val"].append(domain_acc)
            if loss_gain is not None: self.metrics["loss_gain_val"].append(loss_gain)
            if loss_domain_classifier is not None: self.metrics["loss_domain_classifier_val"].append(loss_domain_classifier)
            if loss_model is not None: self.metrics["loss_model_val"].append(loss_model)

    def new_fold(self):
        """Start a new fold and save previous fold’s history."""
        self.fold_history.append(self.metrics)
        self.metrics = {k: [] for k in self.metrics}

    def get_last(self):
        """Return last values of tracked metrics (for logging)."""
        return {k: (v[-1] if v else None) for k, v in self.metrics.items()}

    def get_last_epoch_metric(self, key):
        return self.metrics[key][-1]
    
    def print_metrics_epoch(self, mode=str):
        if mode == "Train":
            print(f"    GAIN Loss (MSE) {self.get_last_epoch_metric('loss_gain_train'):.4f}")
            print(f"    Domain Accuracy: {self.get_last_epoch_metric('domain_acc_train'):.4f}")
            print(f"    Domain Loss: {self.get_last_epoch_metric('loss_domain_classifier_train'):.4f}")
            print(f"    Model Loss: {self.get_last_epoch_metric('loss_model_train'):.4f}")
            print(f"    Reconstruction Error (RMSE): {self.get_last_epoch_metric('rmse_train'):.4f}")
            print(f"    Train Loss (Task-specific): {self.get_last_epoch_metric('loss_train'):.4f}")
        else:
            print(f"    GAIN Loss (MSE) {self.get_last_epoch_metric('loss_gain_val'):.4f}")
            print(f"    Domain Accuracy: {self.get_last_epoch_metric('domain_acc_val'):.4f}")
            print(f"    Domain Loss: {self.get_last_epoch_metric('loss_domain_classifier_val'):.4f}")
            print(f"    Model Loss: {self.get_last_epoch_metric('loss_model_val'):.4f}")
            print(f"    Reconstruction Error (RMSE): {self.get_last_epoch_metric('rmse_val'):.4f}")
            print(f"    Val Loss (Task-specific): {self.get_last_epoch_metric('loss_val'):.4f}")
            

    def mean_over_folds(self):
        """Compute mean metric across folds, handling early stopping (variable lengths)."""
        """
        #todo corrigir esta documentacao
        Compute the mean loss at each epoch across multiple folds.

        Each element in `losses` represents the loss values per epoch for one fold.
        Since folds may stop early due to Early Stopping, the folds can have different lengths.
        This function averages over all available folds at each epoch index.
        
        Args:
            - losses (list[list]): A list where each sublist contains the loss values per epoch for one fold. 
        """

        def mean_over_epochs(metric):
            max_epoch = max(len(l) for l in metric)
            means = []
            for epoch in range(max_epoch):
                values = []
                for fold in metric:
                    if len(fold) > epoch:
                        values.append(fold[epoch])
                means.append(np.mean(values))
            return np.array(means)

        for key in self.fold_history[0].keys():
            folds = [fold[key] for fold in self.fold_history]
            self.avg_metrics[key] = mean_over_epochs(folds)
        

    def to_dataframe(self):
        """Convert current history to pandas DataFrame."""
        return pd.DataFrame(self.metrics)
    
    def plot_domain_accuracies(self, save_dir: str):
        epochs = np.arange(1, len(self.avg_metrics["domain_acc_train"]) + 1)

        plt.figure(figsize=(10,8))
        plt.xlabel('Epoch')
        plt.ylabel('Domain Accuracy')
        plt.plot(epochs, self.avg_metrics["domain_acc_train"], label="Train", color="#fc8b64")
        plt.plot(epochs, self.avg_metrics["domain_acc_val"], label="Validation", color="#909cc5")
        plt.legend()
        plt.grid(True, which='major', color='lightgray', linewidth=0.5)
        plt.tick_params(axis='both', which='both', direction='out')
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_axisbelow(True)
        path = os.path.join(save_dir, "imgs")
        os.makedirs(path, exist_ok=True)
        plt.savefig(f"{path}/domain-acc-{datetime.now().strftime('%d-%m_%H:%M')}.png")
        plt.show() # todo tirar isto daqui
        # plt.close()

    def plot_rmses(self, save_dir: str):
        epochs = np.arange(1, len(self.avg_metrics["rmse_train"]) + 1)

        plt.figure(figsize=(10,8))
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.plot(epochs, self.avg_metrics["rmse_train"], label="Train", color="#fc8b64")
        plt.plot(epochs, self.avg_metrics["rmse_val"], label="Validation", color="#909cc5")
        plt.legend()
        plt.grid(True, which='major', color='lightgray', linewidth=0.5)
        plt.tick_params(axis='both', which='both', direction='out')
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_axisbelow(True)
        path = os.path.join(save_dir, "imgs")
        os.makedirs(path, exist_ok=True)
        plt.savefig(f"{path}/rmse-{datetime.now().strftime('%d-%m_%H:%M')}.png")
        plt.show() # todo tirar isto daqui
        # plt.close()
    
    def plot_task_specific_losses(self, save_dir: str):
        epochs = np.arange(1, len(self.avg_metrics["loss_train"]) + 1)

        plt.figure(figsize=(10,8))
        plt.xlabel('Epoch')
        plt.ylabel('Task-Specific Loss')
        plt.plot(epochs, self.avg_metrics["loss_train"], label="Train", color="#fc8b64")
        plt.plot(epochs, self.avg_metrics["loss_val"], label="Validation", color="#909cc5")
        plt.legend()
        plt.grid(True, which='major', color='lightgray', linewidth=0.5)
        plt.tick_params(axis='both', which='both', direction='out')
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_axisbelow(True)
        path = os.path.join(save_dir, "imgs")
        os.makedirs(path, exist_ok=True)
        plt.savefig(f"{path}/loss-{datetime.now().strftime('%d-%m_%H:%M')}.png")
        plt.show() # todo tirar isto daqui
        # plt.close()
    
    def plot_adversarial_losses(self, save_dir: str):
        # Domain classifier loss
        epochs = np.arange(1, len(self.avg_metrics["domain_acc_train"]) + 1)

        plt.figure(figsize=(10,8))
        plt.xlabel('Epoch')
        plt.ylabel('Adversarial Loss')
        plt.plot(epochs, self.avg_metrics["domain_acc_train"], label="Train", color="#fc8b64")
        plt.plot(epochs, self.avg_metrics["domain_acc_val"], label="Validation", color="#909cc5")
        plt.legend()
        plt.grid(True, which='major', color='lightgray', linewidth=0.5)
        plt.tick_params(axis='both', which='both', direction='out')
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_axisbelow(True)
        path = os.path.join(save_dir, "imgs")
        os.makedirs(path, exist_ok=True)
        plt.savefig(f"{path}/adv-loss-{datetime.now().strftime('%d-%m_%H:%M')}.png")
        plt.show() # todo tirar isto daqui
        # plt.close()

    def plot_model_losses(self, save_dir: str):
        epochs = np.arange(1, len(self.avg_metrics["loss_model_train"]) + 1)

        plt.figure(figsize=(10,8))
        plt.xlabel('Epoch')
        plt.ylabel('Model Loss')
        plt.plot(epochs, self.avg_metrics["loss_model_train"], label="Train", color="#fc8b64")
        plt.plot(epochs, self.avg_metrics["loss_model_val"], label="Validation", color="#909cc5")
        plt.legend()
        plt.grid(True, which='major', color='lightgray', linewidth=0.5)
        plt.tick_params(axis='both', which='both', direction='out')
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_axisbelow(True)
        path = os.path.join(save_dir, "imgs")
        os.makedirs(path, exist_ok=True)
        plt.savefig(f"{path}/model-loss-{datetime.now().strftime('%d-%m_%H:%M')}.png")
        plt.show() # todo tirar isto daqui
        # plt.close()