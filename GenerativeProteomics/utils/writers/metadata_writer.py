import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

class MetadataWriter:
    def __init__(
        self,
        out_dir: Path = None,
        start_time: datetime = None,
        end_time: datetime = None,
        fold_id: int = None,
    ) -> "MetadataWriter":
        self.out_dir = out_dir

        self.fold_id = fold_id
        self.start_time = start_time
        self.end_time = end_time
        self.duration = None
    
    def set_out_dir(
        self,
        out_dir: Path
    ) -> None:
        self.out_dir = out_dir
    
    def set_start_time(
        self,
        start_time: datetime,
    ) -> None:
        self.start_time = start_time

    def set_end_time(
        self,
        end_time: datetime,
    ) -> None:
        self.end_time = end_time

    # @property
    def duration_seconds(
        self
    ) -> float:
        return (self.end_time - self.start_time).total_seconds()
    
    def save_metadata(
        self
    ) -> None:
        
        df = pd.DataFrame([
            {
                "fold": self.fold_id,
                "start time": self.start_time.strftime("%Y-%m-%d_%H:%M:%S"),
                "end time": self.end_time.strftime("%Y-%m-%d_%H:%M:%S"),
                "duration(s)": self.duration_seconds(),
                "duration(hours)": timedelta(seconds=self.duration_seconds()),
                # todo "epoch count": self.epoch_count,
            }
        ])
        
        path = self.out_dir / "metadata.csv"
        if path.exists():
            df.to_csv(path, mode="a", header=False, index=False)
        else:
            df.to_csv(path, index=False)

