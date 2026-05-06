import torch
import numpy as np

from utils.data.dataset import Data
from utils.writers.experiment_writer import ExperimentWriter

class TissueMeanImputer():
    """
    Imputation model using protein-wise mean per tissue type for imputation.
    For each missing protein, imputes the mean of that protein computed
    exclusively from samples of the same tissue type.
    """
    def __init__(self) -> "TissueMeanImputer":
        self.protein_means = None
        self.tissue_means = dict()

    def compute_tissue_means(
        self,
        x,
        tissue_labels: np.ndarray,
    ) -> None:
        x_aux = x.copy()
        for tissue in np.unique(tissue_labels):
            tissue_mask = tissue_labels == tissue
            tissue_data = x_aux[tissue_mask, :]
            tissue_mean = np.nanmean(tissue_data, axis=0)
            self.tissue_means[tissue] = tissue_mean

    def impute_tissue_means(
        self,
        x,
        tissue_labels: np.ndarray,
    ):
        imputed = x.copy()
        global_mean = np.nanmean(x, axis=0)
        for tissue, tissue_mean in self.tissue_means.items():
            tissue_mask = tissue_labels == tissue
            tissue_rows = imputed[tissue_mask, :]
            nan_positions = np.isnan(tissue_rows)
            
            fill_values = np.where(np.isnan(tissue_mean), global_mean, tissue_mean)

            tissue_rows[nan_positions] = np.take(fill_values, np.where(nan_positions)[1])
            imputed[tissue_mask, :] = tissue_rows
        return imputed
    
    def train(
        self,
        data: Data,
        x_train: torch.tensor,
        mask_train: torch.tensor,
        experiment_writer: ExperimentWriter,
    ) -> None:
        _, _, _ = data, mask_train, experiment_writer
        x_train = x_train.detach().cpu().numpy()
        x_tissue = data.tissue.detach().cpu().numpy()
        self.compute_tissue_means(x_train, x_tissue)

    def impute(
        self,
        data: Data,
        x_missing: torch.tensor,
        mask_eval: torch.tensor,
    ):
        _ = mask_eval
        x_missing = x_missing.detach().cpu().numpy()
        x_tissue = data.tissue.detach().cpu().numpy()
        x_hat = self.impute_tissue_means(x_missing, x_tissue)
        return x_hat