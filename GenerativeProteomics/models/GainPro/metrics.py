import numpy as np
import pandas as pd
from pathlib import Path

class Metrics:
    def __init__(
        self,
    ) -> "Metrics":
        
        self.train_metrics = self.init_epoch()
        self.val_metrics = self.init_epoch()

    def init_epoch(self) -> dict[str, list]:
        return {
            "generator_loss": [],
            "discriminator_loss": [],
            "rmse": [],
        }
    
    def init_fold(self) -> None:
        self.train_metrics = self.init_epoch()
        self.val_metrics = self.init_epoch()

    def get_train_metrics(self) -> dict[str, list]:
        return self.train_metrics
    
    def get_val_metrics(self) -> dict[str, list]:
        return self.val_metrics

    def to_dataframe(self) -> pd.DataFrame:
        train_metrics = pd.DataFrame(self.train_metrics)
        val_metrics = pd.DataFrame(self.val_metrics)

        epochs = np.arange(1, len(train_metrics) + 1)
        train_metrics.index = epochs
        val_metrics.index = epochs

        train_metrics.index.name = "epochs"
        val_metrics.index.name = "epochs"

        dfs = [train_metrics, val_metrics]
        return  pd.concat(dfs, axis=1, join="outer")
    
    def to_csv(cls, file_path: Path) -> None:
        df = cls.to_dataframe()
        df.to_csv(file_path)
    
    def to_dataframe(
        self
    ) -> pd.DataFrame:
        df = pd.DataFrame({
            key: value
            for key, value in self.__dict__.items()
            if isinstance(value, np.ndarray)
        })
        df.index.name = "epoch"
        return df

    def __str__(self) -> str:
        return (
            f"Train: {self.train_metrics}\n"
            f"Test: {self.val_metrics}\n"
        )