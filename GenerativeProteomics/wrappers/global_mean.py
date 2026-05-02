import numpy as np
from datetime import datetime

class GlobalMeanImputer():
    def __init__(self) -> "GlobalMeanImputer":
        self.protein_means = None

    def compute_protein_means(
        self,
        df,
    ) -> None:
        self.protein_means = np.nanmean(df, axis=0)
    
    def impute_protein_means(
        self,
        df,
    ):
        imputed = df.copy()
        nan_positions = np.isnan(imputed)
        imputed[nan_positions] = np.take(self.protein_means, np.where(nan_positions)[1])
        return imputed
    
    def train(
        self,
        x_train: np.ndarray,
    ) -> None:
        """
        Args:
            - x_train (np.ndarray): Array with missing values.
        """
        self.compute_protein_means(
            df=x_train,
        )
    
    def impute(
        self,
        x_missing: np.ndarray,
    ):
        x_hat = self.impute_protein_means(
            df=x_missing,
        )
        return x_hat