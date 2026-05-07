import torch
import numpy as np
from datetime import datetime

from utils.data.dataset import Data
from wrappers.gain import GainImputer
from utils.writers.experiment_writer import ExperimentWriter
from evaluation.evaluation_strategy import EvaluationStrategy


class HoldoutStrategy(EvaluationStrategy):
    def run(
        self,
        imputer_factory,
        data: Data,
        experiment_writer: ExperimentWriter,
        positive_label: int,
    ) -> None:
        evaluation_holdout_dir = experiment_writer.evaluation_dir / "holdout"
        evaluation_holdout_dir.mkdir(parents=True, exist_ok=True)
        experiment_writer.metadata_writer.set_out_dir(evaluation_holdout_dir)
        experiment_writer.result_writer.set_results_dir(results_dir=evaluation_holdout_dir)
        experiment_writer.result_writer.set_prediction_dir(prediction_dir=experiment_writer.preds_dir)
        
        observed_mask = data.observed_mask.detach().cpu().numpy()
        artificial_mask = data.artificial_missing_mask.detach().cpu().numpy()
        
        mask_train = ~(artificial_mask)
        mask_train_tensor = torch.tensor(mask_train, device=data.device)
        mask_eval = (artificial_mask == True)
        mask_eval_tensor = torch.tensor(mask_eval, device=data.device)
        
        x_train = data.reference.detach()
        x_true = data.reference.detach().cpu().numpy()
        x_true = np.nan_to_num(x_true, nan=0)
        x_missing = data.missing.detach()
        
        input_dim = data.reference.shape[1]
        tissue_dim = len(data.tissue_mapping)
        imputer = imputer_factory(input_dim, tissue_dim)
        experiment_writer.metadata_writer.set_start_time(datetime.now())

        imputer.train(
            data=data,
            x_train=x_train,
            mask_train=mask_train_tensor,
            experiment_writer=experiment_writer,
        )
        x_pred = imputer.impute(
            data=data,
            x_missing=x_missing,
            mask_eval=mask_eval_tensor,
        )
        experiment_writer.metadata_writer.set_end_time(datetime.now())

        if isinstance(imputer, GainImputer):
            discriminator_precision_recall = imputer.evaluate_discriminator(
                data=data,
                x_missing=x_missing, 
                mask_eval=mask_eval_tensor,
                positive_label=positive_label, 
            )
            experiment_writer.result_writer.save_precision_recall_discriminator(
                precision_recall=discriminator_precision_recall,
            )

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
            x_true_log2p1_inverse,
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
            rmse=rmse,
        )
        
        experiment_writer.metadata_writer.save_metadata()