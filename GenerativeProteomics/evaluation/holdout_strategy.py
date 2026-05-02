import torch
import numpy as np
from datetime import datetime

from utils.data.dataset import Data
from utils.writers.experiment_writer import ExperimentWriter
from evaluation.evaluation_strategy import EvaluationStrategy


class HoldoutStrategy(EvaluationStrategy):
    def run(
        self,
        imputer_factory,
        data: Data,
        experiment_writer: ExperimentWriter,
        # todo: train/val split
        # test_size: float=0.2,
        # random_state: int=42,
        **kwargs,
    ) -> None:
        holdout_dir = experiment_writer.evaluation_dir / "holdout"
        holdout_dir.mkdir(parents=True, exist_ok=True)
        experiment_writer.metadata_writer.set_out_dir(holdout_dir)
        
        observed_mask = data.observed_mask.detach().cpu().numpy()
        artificial_mask = data.artificial_missing_mask.detach().cpu().numpy()
        
        mask_train = (observed_mask == True) & (artificial_mask == True)
        mask_train_tensor = torch.tensor(mask_train, device=data.device)
        mask_eval = (observed_mask == True) & (artificial_mask == False)
        mask_eval_tensor = torch.tensor(mask_eval, device=data.device)
        
        x_train_tensor = data.reference.detach()
        x_train = x_train_tensor.cpu().numpy()
        x_true = data.reference.detach().cpu().numpy()
        x_true = np.nan_to_num(x_true, nan=0)
        x_missing_tensor = data.missing.detach()
        x_missing = x_missing_tensor.cpu().numpy()
        x_missing_tissue = data.tissue.detach().cpu().numpy()
        
        input_dim = data.reference.shape[1]
        imputer = imputer_factory(input_dim)
        experiment_writer.metadata_writer.set_start_time(datetime.now())
        if imputer.__class__.__name__ == "GainImputer":
            imputer.train(
                x_train=x_train_tensor, 
                observed_mask=mask_train_tensor,
            )
            x_pred = imputer.impute(
                x_missing=x_missing_tensor, 
                mask=mask_eval_tensor
            )
        elif imputer.__class__.__name__ == "MissForestRImputer":
            imputer.train(x_missing)
            x_pred = imputer.impute(
                x_missing=x_missing,
            )
        elif imputer.__class__.__name__ == "TissueMeanImputer":
            imputer.train(
                x_missing,
                x_tissue=x_missing_tissue,
            )
            x_pred = imputer.impute(
                x_missing=x_missing,
                x_tissue=x_missing_tissue,
            )
        else:
            imputer.train(
                x_train=x_train,
            )
            x_pred = imputer.impute(
                x_missing=x_missing,
            )

        experiment_writer.metadata_writer.set_end_time(datetime.now())
        rmse = self._compute_rmse(x_true, x_pred, mask_eval)
        print("RMSE:", rmse)
        
        max_norm = data.max_norm.values
        min_norm = data.min_norm.values
        x_pred_denorm = self._denormalize(x_pred, max_norm, min_norm)
        x_true_denorm = self._denormalize(x_true, max_norm, min_norm)
        
        x_pred_log2p1_inverse = self._inverse_log2p1(x_pred_denorm)
        x_true_log2p1_inverse = self._inverse_log2p1(x_true_denorm)
        x_true_log2p1_inverse = np.where(
            observed_mask == 0, 
            np.nan, 
            np.power(2, x_true) - 1
        )
        
        experiment_writer.result_writer.save_predictions(
            sample_ids=data.sample_names,
            feature_names=data.feature_names,
            true_values=x_true_log2p1_inverse,
            pred_values=x_pred_log2p1_inverse,
            observed_mask=data.observed_mask,
            artificial_missing_mask=data.artificial_missing_mask,
            group_ids=data.tissue.cpu().numpy(),
            group_mapping=data.tissue_mapping,
        )
        
        experiment_writer.result_writer.save_test_rmse(
            out_dir=holdout_dir,
            rmse=rmse,
        )
        
        experiment_writer.metadata_writer.save_metadata()