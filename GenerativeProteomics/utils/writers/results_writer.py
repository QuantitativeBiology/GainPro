import numpy as np
import pandas as pd
from pathlib import Path

class ResultWriter:
    def __init__(
        self,
        preds_dir: Path,
        evaluation_dir: Path,
    ) -> "ResultWriter":
        self.evaluation_dir = evaluation_dir
        self.preds_dir = preds_dir 
    
    def save_predictions(
        self,
        sample_ids: list,
        feature_names: list,
        true_values: np.ndarray,
        pred_values: np.ndarray,
        observed_mask: np.ndarray,
        artificial_missing_mask: np.ndarray,
        fold_id: int=None,
        group_mapping: dict=None,
        group_ids=None
    ) -> None:
        safe_group_mapping = group_mapping or {}
        records = []
        assert group_ids is None or len(group_ids) == len(sample_ids), \
            f"group_ids ({len(group_ids)}) != sample_ids ({len(sample_ids)})"
        for i, sample_id in enumerate(sample_ids):
            for j, feature in enumerate(feature_names):
                if group_ids is None:
                    mapped_gid = None
                else:
                    raw_gid = group_ids[i]
                    mapped_gid = safe_group_mapping.get(raw_gid, raw_gid)

                records.append({
                    "fold": fold_id,
                    "sample_id": sample_id,
                    "feature": feature,
                    "true_value": true_values[i, j],
                    "predicted_value": pred_values[i, j],
                    "observed_mask": int(observed_mask[i, j]),
                    "artificial_missing_mask": int(artificial_missing_mask[i, j]),
                    "group_id": mapped_gid,
                })

        df = pd.DataFrame(records)
        if fold_id is not None:
            fold_dir = self.preds_dir / f"fold_{fold_id}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(fold_dir / "predictions.csv", index=False)
        else:
            df = df.drop(columns=["fold"])
            df.to_csv(self.preds_dir / "predictions.csv", index=False)

    def save_test_rmse(
        self,
        out_dir: Path,
        rmse,
        fold_id: int = None,
    ) -> None:
        
        df = pd.DataFrame([
            {
                "fold": fold_id,
                "test rmse": rmse
            }
        ])
        
        path = out_dir / "rmse_test.csv"
        if path.exists():
            df.to_csv(path, mode="a", header=False, index=False)
        else:
            df.to_csv(path, index=False)
    
    def save_metrics(
        self,
        out_dir: Path,
    ) -> None:
        pass