import numpy as np

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
        x_train: np.ndarray,
        x_tissue,
    ) -> None:
        self.compute_tissue_means(x_train, x_tissue)

    def impute(
        self,
        x_missing: np.ndarray,
        x_tissue,
    ):
        x_hat = self.impute_tissue_means(x_missing, x_tissue)
        return x_hat