from pathlib import Path
import pandas as pd

from .plot_primitives import plot_domain_accuracies, plot_domain_adversarial_losses, plot_gain_losses, plot_model_losses, plot_task_specific_losses, plot_rmses

class Plotter:
    def __init__(cls, run_dir: str):
        cls.run_dir = Path(run_dir)

    def _load_training_metrics(cls) -> pd.DataFrame:
        metrics_df = pd.read_csv(
            Path(f"{cls.run_dir}/training/metrics.csv")
        )
        return metrics_df

    def plot_training(cls) -> None:
        metrics_df = cls._load_training_metrics()
        
        plot_domain_accuracies(
            metrics_df["train_domain_accuracy"],
            metrics_df["val_domain_accuracy"],
            Path(f"{cls.run_dir}/plots/training")
        )

        plot_domain_adversarial_losses(
            metrics_df["train_domain_classifier_loss"],
            metrics_df["val_domain_classifier_loss"],
            Path(f"{cls.run_dir}/plots/training")
        )

        plot_gain_losses(
            metrics_df["train_gain_mse_loss"],
            metrics_df["val_gain_mse_loss"],
            Path(f"{cls.run_dir}/plots/training")
        )

        plot_rmses(
            metrics_df["train_decoder_rmse_loss"],
            metrics_df["val_decoder_rmse_loss"],
            Path(f"{cls.run_dir}/plots/training")
        )
        
        plot_task_specific_losses(
            metrics_df["train_task_specific_loss"],
            metrics_df["val_task_specific_loss"],
            Path(f"{cls.run_dir}/plots/training")
        )

        plot_model_losses(
            metrics_df["train_model_loss"],
            metrics_df["val_model_loss"],
            Path(f"{cls.run_dir}/plots/training")
        )

