import numpy as np
import pandas as pd
from pathlib import Path

class Metrics:
    def __init__(
        self,
    ) -> "Metrics":
        
        self.train_metrics = self.init_epoch()
        self.test_metrics = self.init_epoch()

        # fix: os nomes das losses deviam ser d_loss and not loss_d
        # self.discriminator_loss = np.zeros(num_epochs)
        # self.loss_D_evaluate = np.zeros(num_epochs)

        # self.generator_loss = np.zeros(num_epochs)
        # self.loss_G_evaluate = np.zeros(num_epochs)

        # self.loss_MSE_train = np.zeros(num_epochs)
        # self.loss_MSE_train_evaluate = np.zeros(num_epochs) #todo what does mse_train_evaluate mean?
        # self.loss_RMSE_train = np.zeros(num_epochs)

        # self.loss_MSE_test = np.zeros(num_epochs)
        # self.loss_RMSE_test = np.zeros(num_epochs)

        # self.cpu = np.zeros(num_epochs)
        # self.cpu_evaluate = np.zeros(num_epochs)

        # self.ram = np.zeros(num_epochs)
        # self.ram_evaluate = np.zeros(num_epochs)

        # self.ram_percentage = np.zeros(num_epochs)
        # self.ram_percentage_evaluate = np.zeros(num_epochs)

    def init_epoch(self) -> dict[str, list]:
        return {
            "generator_loss": [],
            "discriminator_loss": [],
            "rmse": [],
        }

    def get_train_metrics(self) -> dict[str, list]:
        return self.train_metrics
    
    def get_val_metrics(self) -> dict[str, list]:
        return self.test_metrics

    def to_dataframe(self) -> pd.DataFrame:
        train_metrics = pd.DataFrame(self.train_metrics)
        val_metrics = pd.DataFrame(self.test_metrics)

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
            f"Test: {self.test_metrics}\n"
        )