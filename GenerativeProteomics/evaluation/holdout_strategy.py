import torch
import logging
import numpy as np
from datetime import datetime

from utils.data.dataset import Data
from wrappers.gain import GainImputer
from utils.writers.experiment_writer import ExperimentWriter
from evaluation.evaluation_strategy import EvaluationStrategy

logger = logging.getLogger(__name__)

class HoldoutStrategy(EvaluationStrategy):
    def run(
        self,
        imputer_factory,
        data: Data,
        experiment_writer: ExperimentWriter,
        positive_label: int=1,
    ) -> None:
        evaluation_holdout_dir = experiment_writer.evaluation_dir / "holdout"
        evaluation_holdout_dir.mkdir(parents=True, exist_ok=True)
        experiment_writer.metadata_writer.set_out_dir(evaluation_holdout_dir)
        experiment_writer.result_writer.set_results_dir(results_dir=evaluation_holdout_dir)
        experiment_writer.result_writer.set_prediction_dir(prediction_dir=experiment_writer.preds_dir)
        
        observed_mask = data.observed_mask.detach().cpu().numpy()
        artificial_mask = data.artificial_missing_mask.detach().cpu().numpy()
        mask_observed_tensor = torch.tensor(observed_mask, device=data.device)
        
        mask_train = (observed_mask == True) & (artificial_mask != True)
        mask_train_tensor = torch.tensor(mask_train, device=data.device)
        mask_eval = (artificial_mask == True)
        mask_eval_tensor = torch.tensor(mask_eval, device=data.device)
        
        x_true = data.reference.detach().cpu().numpy()
        x_true = np.nan_to_num(x_true, nan=0)
        x_true_tensor = torch.tensor(x_true, device=data.device, dtype=torch.float32)
        x_missing = data.missing.detach()
        
        imputer = imputer_factory()
        experiment_writer.metadata_writer.set_start_time(datetime.now())

        imputer.train(
            x_train=x_missing,
            x_true=x_true_tensor,
            mask_train=mask_train_tensor,
            experiment_writer=experiment_writer,
        )
        mask_impute = (mask_eval == True) | (observed_mask == False) # impute artificial hidden entries and original missing entries
        mask_impute_tensor = torch.tensor(mask_impute, device=data.device)
        x_pred = imputer.impute(
            x_missing=x_missing,
            mask=mask_impute_tensor,
        )
        experiment_writer.metadata_writer.set_end_time(datetime.now())

        if isinstance(imputer, GainImputer):
            discriminator_precision_recall = imputer.evaluate_discriminator(
                x_missing=x_missing, 
                mask_observed=mask_observed_tensor,
                mask_eval=mask_eval_tensor,
                positive_label=positive_label, 
            )
            experiment_writer.result_writer.save_precision_recall_discriminator(
                precision_recall=discriminator_precision_recall,
            )

        rmse = self._compute_rmse(x_true, x_pred, mask_eval)
        logging.info(f"RMSE: {rmse}")
        # Sanity check on training data
        train_rmse = self._compute_rmse(x_true, x_pred, mask_train)
        logger.debug(f"RMSE on training data: {train_rmse}")

        x_pred_denorm = data.denormalize(x_pred)
        x_true_denorm = data.denormalize(x_true)

        if data.log_transform:
            x_true_out = data._inverse_log2p1(x_true_denorm)
            x_pred_out = data._inverse_log2p1(x_pred_denorm)
        else:
            x_true_out = x_true_denorm
            x_pred_out = x_pred_denorm

        x_true_out = np.where(observed_mask == 0, np.nan, x_true_out,)
        
        experiment_writer.result_writer.save_predictions(
            sample_ids=data.sample_names,
            feature_names=data.feature_names,
            true_values=x_true_out,
            pred_values=x_pred_out,
            observed_mask=data.observed_mask,
            artificial_missing_mask=data.artificial_missing_mask,
            group_ids=data.tissue.cpu().numpy(),
            group_mapping=data.tissue_mapping,
            transpose=data.transpose,
        )
        
        experiment_writer.result_writer.save_test_rmse(
            rmse=rmse,
        )
        
        experiment_writer.metadata_writer.save_metadata()