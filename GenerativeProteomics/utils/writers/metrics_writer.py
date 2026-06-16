import pandas as pd
from pathlib import Path

from models.Gain.metrics import Metrics

class MetricsWriter:
    def __init__(
        self,
        out_dir: Path,
    ) -> "MetricsWriter":
        self.out_dir = out_dir

    def _metrics_to_df(
        self,
        metrics: dict,
        fold_id: int=None,
    ) -> pd.DataFrame:
        
        METRIC_KEYS = [
            "generator_loss",
            "generator_rmse",
            "generator_entropy",
            "discriminator_loss",
            "rmse",
        ]
                
        num_epochs = len(metrics[METRIC_KEYS[2]])

        records = []
        for ep in range(num_epochs):
            if fold_id != None:
                record = {
                    "fold": fold_id,
                    "epoch": ep,
                }
            else:
                record = {
                    "epoch": ep,
                }
            record.update({k: metrics[k][ep] for k in METRIC_KEYS})
            records.append(record)

        return pd.DataFrame(records)

    def log_metrics(
        self,
        metrics: Metrics,
        fold_id: int=None,
    ) -> None:

        train_df = self._metrics_to_df(
            fold_id=fold_id,
            metrics=metrics.get_train_metrics(),
        )
        val_df = self._metrics_to_df(
            fold_id=fold_id,
            metrics=metrics.get_val_metrics(),
        )

        train_path = self.out_dir / "train.csv"
        val_path = self.out_dir / "validation.csv"

        for df, path in [(train_df, train_path), (val_df, val_path)]:
            if path.exists():
                df.to_csv(path, mode="a", header=False, index=False)
            else:
                df.to_csv(path, index=False)