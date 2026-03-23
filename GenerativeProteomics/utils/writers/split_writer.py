import numpy as np
from pathlib import Path

class SplitWriter:
    def __init__(
        self,
        split_dir: Path,
    ) -> "SplitWriter":
        self.split_dir = split_dir

    def save_splits(
        self,
        out_dir: Path,
        train_idx: np.array,
        val_idx: np.array=None,
        test_idx: np.array=None,
    ) -> None:
        np.save(out_dir/ "train_idx.npy", train_idx)
        # np.save(self.split_dir / "val_idx.npy", val_idx)
        if test_idx is not None:
            np.save(out_dir / "test_idx.npy", test_idx)

    def save_fold_splits(
        self,
        fold_id: int,
        train_idx: np.array,
        test_idx: np.array,
        val_idx: np.array=None,
    ) -> None:
        fold_dir = self.split_dir / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        self.save_splits(fold_dir, train_idx, val_idx, test_idx)