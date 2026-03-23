import pandas as pd
from pathlib import Path

from utils.data.dataset import Data

class DatasetWriter:
    def _save_df(
        cls,
        df: pd.DataFrame, 
        out_path: Path
    ) -> None:
        df.to_csv(out_path, index=True)
        df_out = pd.read_csv(out_path, index_col=0)
        assert df.shape == df_out.shape, f"Expected shape {df_out.shape}, but got {df.shape}."

    def save_data(
        cls,
        data: Data,
        out_dir: Path,
    ) -> None:

        cls._save_df(
            pd.DataFrame(
                data.reference.detach().cpu().numpy(),
                index=data.sample_names,
                columns=data.feature_names
            ), 
            f"{out_dir}/reference.csv"
        )
        cls._save_df(
            pd.DataFrame(
                data.missing.detach().cpu().numpy(),
                index=data.sample_names,
                columns=data.feature_names
            ), 
            f"{out_dir}/missing.csv"
        )
        cls._save_df(
            pd.DataFrame(
                data.observed_mask.detach().cpu().numpy(),
                index=data.sample_names,
                columns=data.feature_names
            ),  
            f"{out_dir}/observed_mask.csv")
        cls._save_df(
            pd.DataFrame(
                data.artificial_missing_mask.detach().cpu().numpy(),
                index=data.sample_names,
                columns=data.feature_names
            ),  
            f"{out_dir}/artificial_missing_mask.csv"
        )
        # todo save cell line and/or condition and/or tissue
        # cls._save_df(data.cell_line, f"{out_dir}/cell_line.csv")
        # cls._save_df(data.cell_line_mapping, f"{out_dir}/cell_line_mapping.csv")
    
    def save_data_metadata(
        cls,
        original_missingness: float,
        miss_rate: float,
        current_missingness: float,
        induction_strategy: str,
        seed: int,
        out_dir: Path
    ) -> None:
        df = pd.DataFrame([
            {
                "original missingness": original_missingness,
                "induced missingness rate": miss_rate,
                "current missingness": current_missingness,
                "seed": seed,
                "induction strategy": induction_strategy
            }
        ])
        path = out_dir / "metadata.csv"
        df.to_csv(path, index=False)