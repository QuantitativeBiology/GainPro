import shutil
import numpy as np

from missingpy import MissForest

from utils.model_hypers import MissForestHypers

class MissForestRImputer():
    def __init__(
        self,
        missforest_hypers: MissForestHypers,
    ) -> "MissForestRImputer":
        if shutil.which("R") is None:
            raise RuntimeError(
                "R executable not found. You must install R (https://cran.r-project.org/) "
                "and add it to your system PATH to use this model."
            )

        try:
            self.n_tree = missforest_hypers.n_tree
            self.max_iter = missforest_hypers.max_iter
            print("Number of trees:", self.n_tree)
            print("Maximum iterations:", self.max_iter)

            self.missforest = MissForest(
                n_estimators=self.n_tree,
                max_iter=self.max_iter,
                max_features=None,
                criterion=("squared_error")  
            )
        except Exception as e:  
            print("The 'missForest' R package is not installed. Please install it in your R environment.")
            raise e
        
    def train(
        self,
        x_train: np.ndarray,
    ) -> None:
        self.missforest = MissForest(
            n_estimators=self.n_tree,
            max_iter=self.max_iter,
            max_features=None,
            criterion=("squared_error")  
        )
        self.missforest.fit(x_train)

    def impute(
        self,
        x_missing: np.ndarray,
    ):
        x_hat = self.missforest.transform(x_missing)
        return x_hat