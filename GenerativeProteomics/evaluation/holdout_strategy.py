import torch
import logging
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split

from utils.data.dataset import Data
from utils.data.normalizer import MinMaxNormalizer
from utils.metrics.metrics import rmse
from wrappers.gain import GainImputer
from wrappers.ae import AutoEncoderImputer
from utils.writers.experiment_writer import ExperimentWriter
from evaluation.evaluation_strategy import EvaluationStrategy

logger = logging.getLogger(__name__)

class HoldoutStrategy(EvaluationStrategy):
    def run(
        self,
        imputer_factory,
        data: Data,
        experiment_writer: ExperimentWriter,
        val_size: float=0.2,
        seed: int=42,
        positive_label: int=1,
    ) -> None:
        evaluation_holdout_dir = experiment_writer.evaluation_dir / "holdout"
        evaluation_holdout_dir.mkdir(parents=True, exist_ok=True)
        experiment_writer.metadata_writer.set_out_dir(evaluation_holdout_dir)
        experiment_writer.result_writer.set_results_dir(results_dir=evaluation_holdout_dir)
        experiment_writer.result_writer.set_prediction_dir(prediction_dir=experiment_writer.preds_dir)
        
        observed_mask = data.observed_mask.detach().cpu().numpy()

        # artificial_mask: artificially hidden positions used to assess imputation quality
        # Shape: (n_samples, n_features), dtype: bool
        # These entries had known ground-truth values (from observed_mask) that the benchmark
        # deliberately removed to simulate missingness.
        # RMSE is computed here only
        artificial_mask = data.artificial_missing_mask.detach().cpu().numpy()

        # mask_train: observed positions excluding artificially hidden entries
        # Shape: (n_samples, n_features), dtype: bool
        # - (a) observed_mask == True: positions with real observed values (not originally NaN)
        # - (b) artificial_mask != True: exclude positions the benchmark intentionally hid for evaluation
        # Result: only positions that are (a) real data and (b) not held out
        mask_train = (observed_mask == True) & (artificial_mask != True)
        mask_train = torch.tensor(mask_train, device=data.device)
        
        x_true = np.nan_to_num(data.reference.detach().cpu().numpy(), nan=0)
        x_true_tensor = torch.tensor(x_true, device=data.device, dtype=torch.float32)
        x_missing = data.missing.detach()
        tissue = data.tissue.detach() if data.tissue is not None else None

        imputer = imputer_factory()

        if isinstance(imputer, AutoEncoderImputer) or isinstance(imputer, GainImputer):
            train_idx, val_idx = train_test_split(
                np.arange(x_missing.shape[0]),
                test_size=val_size,
                random_state=seed,
            )
        else:
            all_idx = np.arange(x_missing.shape[0])
            train_idx, val_idx = all_idx, all_idx
            
        experiment_writer.metadata_writer.set_start_time(datetime.now())

        logger.debug(
            f"\n Train:"
            f"\n X true: {x_true_tensor}"
            f"\n X missing: {x_missing}"
            f"\n Mask: {mask_train}"
            f"\n Artificial mask: {data.artificial_missing_mask}"
        )

        imputer.train(
            x_train=x_missing[train_idx],
            x_true=x_true_tensor[train_idx],
            mask_train=mask_train[train_idx],
            artificial_mask_train=data.artificial_missing_mask[train_idx],
            x_val=x_missing[val_idx],
            x_true_val=x_true_tensor[val_idx],
            mask_val=mask_train[val_idx],
            artificial_mask_val= data.artificial_missing_mask[val_idx],
            experiment_writer=experiment_writer,
            tissue=tissue,
        )
        x_pred = imputer.impute(
            x_missing=x_missing,
            mask=mask_train,
            tissue=tissue,
        )
        experiment_writer.metadata_writer.set_end_time(datetime.now())
        x_pred = x_pred.detach().cpu().numpy()

        # if isinstance(imputer, GainImputer):
        #     discriminator_precision_recall = imputer.evaluate_discriminator(
        #         x_missing=x_missing, 
        #         mask_observed=mask_observed_tensor,
        #         artificial_mask=mask_eval_tensor,
        #         positive_label=positive_label, 
        #     )
        #     experiment_writer.result_writer.save_precision_recall_discriminator(
        #         precision_recall=discriminator_precision_recall,
        #     )

        test_rmse = rmse(x_true, x_pred, artificial_mask)
        logging.info(f"RMSE: {test_rmse}")
        # Sanity check on training data
        logger.debug(
            f"\n RMSE on training data: {rmse(x_true, x_pred, mask_train.detach().cpu().numpy())}"
            f"\n X: {x_true}"
            f"\n X hat: {x_pred}"
            f"\n Mask: {mask_train}"
        )

        if isinstance(data.normalizer, MinMaxNormalizer):
            logger.debug(
                f"\n Data stats"
                f"\n    Mins Mean: {data.normalizer.mins.mean()}"
                f"\n    Maxs Mean: {data.normalizer.maxs.mean()}"
                f"\n    Ranges Mean: {data.normalizer.ranges.mean()}"
            )

        missing_mask = observed_mask == 0
        logger.debug(
            f"\n Observed entries"
            f"\n In normalized space"
            f"\n    X predicted mean: {x_pred[observed_mask == 1].mean()}"
            f"\n    X true mean: {x_true[observed_mask == 1].mean()}"
            f"\n After inverse transformation"
            f"\n    X predicted mean: {data.inverse_transform(x_pred)[observed_mask == 1].mean()}"
            f"\n    X true mean: {data.inverse_transform(x_true)[observed_mask == 1].mean()}"
            f"\n Missing entries"
            f"\n    X predicted mean: {x_pred[missing_mask].mean()}"
            f"\n    X true mean: {x_true[missing_mask].mean()}" 
            f"\n Artificial entries (held-out for eval)"
            f"\n    X predicted mean: {x_pred[artificial_mask].mean()}"
            f"\n    X true mean: {x_true[artificial_mask].mean()}"
        )

        x_pred_out = data.inverse_transform(x_pred)
        x_true_out = data.inverse_transform(x_true)
        x_true_out = np.where(observed_mask == 0, np.nan, x_true_out)
        
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
        experiment_writer.result_writer.save_test_rmse(rmse=test_rmse)
        experiment_writer.metadata_writer.save_metadata()