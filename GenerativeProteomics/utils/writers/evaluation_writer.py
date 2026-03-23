import pandas as pd
from pathlib import Path

class EvaluationWriter:
    def __init__(
        self,
        evaluation_dir: Path,
    ) -> "EvaluationWriter":
        self.evaluation_dir = evaluation_dir

    def set_evaluation_dir(
        self,
        eval_dir: Path,
    ) -> None:
        self.evaluation_dir = eval_dir

    def save_hold_out_cv(
        self,
        mask: pd.DataFrame,
        true_matrix: pd.DataFrame,
        pred_matrix: pd.DataFrame,
        rmse: float,
    ) -> None:
        mask.to_csv(f"{self.evaluation_dir}/mask.csv", index=True)
        true_matrix.to_csv(f"{self.evaluation_dir}/true_matrix.csv", index=True)
        pred_matrix.to_csv(f"{self.evaluation_dir}/pred_matrix.csv", index=True)
        
        df = pd.DataFrame([rmse], columns=["rmse"])
        df.to_csv(f"{self.evaluation_dir}/rmse.csv", index=True)

    def save_kfold_cv(
        self,
        true_matrix: pd.DataFrame,
        pred_matrix: pd.DataFrame,
        rmse: float,
    ) -> None:
        true_matrix.to_csv(f"{self.evaluation_dir}/true_matrix.csv", index=True)
        pred_matrix.to_csv(f"{self.evaluation_dir}/pred_matrix.csv", index=True)
        
        df = pd.DataFrame([rmse], columns=["rmse"])
        df.to_csv(f"{self.evaluation_dir}/rmse.csv", index=True)


