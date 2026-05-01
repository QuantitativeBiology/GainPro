import torch
import numpy as np
from datetime import datetime
from sklearn.model_selection import GroupKFold

from utils.data.dataset import Data
from utils.writers.experiment_writer import ExperimentWriter
from evaluation.evaluation_strategy import EvaluationStrategy


class GroupKFoldStrategy(EvaluationStrategy):
    def run(
        self,
        imputer_factory,
        data: Data,
        experiment_writer: ExperimentWriter,
        num_folds: int=5,
        holdout_tissues: list=None,
        **kwargs,
    ):
        """
        Run Group K-Fold evaluation.
        
        Args:
            imputer: Imputer wrapper to evaluate
            data: Data object with reference and missing matrices
            experiment_writer: Writer for outputs
            num_folds: Number of folds for cross-validation
            holdout_tissues: Optional list of tissues to hold out entirely
            
        Returns:
            dict: Evaluation results with aggregated metrics
        """
        groupkfold_dir = experiment_writer.preds_dir / "groupkfold"
        groupkfold_dir.mkdir(parents=True, exist_ok=True)
        experiment_writer.metadata_writer.set_out_dir(groupkfold_dir)

        tissue_name_to_id = {v: k for k, v in data.tissue_mapping.items()}
        
        # Convert holdout_tissues to tissue IDs if they're names
        holdout_tissue_ids: set[int] | None = None
        if holdout_tissues is not None:
            holdout_tissue_ids = set()
            for t in holdout_tissues:
                if isinstance(t, str):
                    if t not in tissue_name_to_id:
                        raise ValueError(
                            f"Unknown tissue name: '{t}'. "
                            f"Available tissues: {list(tissue_name_to_id.keys())}"
                        )
                    holdout_tissue_ids.add(tissue_name_to_id[t])
                elif isinstance(t, int):
                    holdout_tissue_ids.add(t)
                else:
                    raise TypeError(
                        f"holdout_tissues must contain str or int, got {type(t)}"
                    )
            print(f"Holdout tissues: {holdout_tissues}")
            print(f"Holdout tissue IDs: {holdout_tissue_ids}")

        groups = data.tissue.cpu().numpy()

        if holdout_tissue_ids is not None:
            # Single tissue holdout
            folds = _get_folds_from_holdout_tissues(groups, holdout_tissue_ids)
        else:
            # Multiple tissues holdout
            gkf = GroupKFold(n_splits=num_folds)
            folds = list(gkf.split(X=data.reference.cpu().numpy(), y=groups, groups=groups))

        for fold_id, (train_idx, test_idx) in enumerate(folds, start=1):
            print(f"\n\n------------ Fold {fold_id}/{num_folds} ------------\n")

            fold_dir = groupkfold_dir / f"fold_{fold_id}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            experiment_writer.metadata_writer.set_out_dir(fold_dir)

            train_tissues = set(groups[train_idx])
            test_tissues = set(groups[test_idx])
            assert train_tissues.isdisjoint(test_tissues), \
                f"Fold {fold_id}: tissue leakage. Overlap: {train_tissues & test_tissues}"

            holdout_tissue_ids = test_tissues
            holdout_tissue_names = [
                data.tissue_mapping[t] for t in holdout_tissue_ids
            ]

            print(f"Holdout tissue(s): {holdout_tissue_names}")
            print(f"Train samples: {len(train_idx)}, Test samples: {len(test_idx)}\n")

            x_train = data.reference[train_idx, :].detach().cpu().numpy()
            x_true_test = data.reference[test_idx, :].detach().cpu().numpy()
            x_true_test = np.nan_to_num(x_true_test, nan=0)
            x_missing_test = data.missing[test_idx, :].detach().cpu().numpy()

            observed_mask_np = data.observed_mask.detach().cpu().numpy()
            artificial_mask_np = data.artificial_missing_mask.detach().cpu().numpy()
            mask_train = (observed_mask_np==True) & (artificial_mask_np==True)
            mask_train = mask_train[train_idx]
            mask_eval = (observed_mask_np==True) & (artificial_mask_np==False) # artificial hidden entries
            mask_eval = mask_eval[test_idx]

            # Reinitialize imputer
            input_dim = data.reference.shape[1]
            imputer = imputer_factory(input_dim)

            # Train
            experiment_writer.metadata_writer.set_start_time(datetime.now())
            if imputer.__class__.__name__ == "GainImputer":
                imputer.train(
                    x_train=torch.tensor(x_train, device=data.device),
                    observed_mask=torch.tensor(mask_train, device=data.device),
                )
                x_pred_test = imputer.impute(
                    x_missing=torch.tensor(x_missing_test, device=data.device), 
                    mask=torch.tensor(mask_eval, device=data.device)
                )
            else:
                imputer.train(x_train)
                x_pred_test = imputer.impute(x_missing_test)

            experiment_writer.metadata_writer.set_end_time(datetime.now())

            rmse = self._compute_rmse(x_true_test, x_pred_test, mask_eval)
            print("RMSE:", rmse)
            
            max_norm = data.max_norm.values
            min_norm = data.min_norm.values
            x_pred_denorm = self._denormalize(x_pred_test, max_norm, min_norm)
            x_true_denorm = self._denormalize(x_true_test, max_norm, min_norm)
            
            x_pred_log2p1_inverse = self._inverse_log2p1(x_pred_denorm)
            x_true_log2p1_inverse = self._inverse_log2p1(x_true_denorm)
            x_true_log2p1_inverse = np.where(
                observed_mask_np[test_idx] == 0, 
                np.nan, 
                np.power(2, x_true_test) - 1
            )
            
            experiment_writer.result_writer.set_prediction_dir(prediction_dir=fold_dir)
            experiment_writer.result_writer.save_predictions(
                fold_id=fold_id,
                sample_ids=np.array(data.sample_names)[test_idx],
                feature_names=data.feature_names,
                true_values=x_true_log2p1_inverse,
                pred_values=x_pred_log2p1_inverse,
                observed_mask=data.observed_mask,
                artificial_missing_mask=data.artificial_missing_mask,
                group_ids=data.tissue[test_idx].cpu().numpy(),
                group_mapping=data.tissue_mapping,
            )
            
            experiment_writer.result_writer.save_test_rmse(
                out_dir=fold_dir,
                rmse=rmse,
            )
            
            experiment_writer.metadata_writer.save_metadata()

def _get_folds_from_holdout_tissues(
    groups: np.ndarray,
    holdout_tissue_ids: set[int],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Create folds based on specified holdout tissue IDs.
    
    Args:
        groups: Array of tissue IDs for each sample.
        holdout_tissue_ids: Set of tissue IDs to holdout as test set.
        
    Returns:
        List of (trainval_idx, test_idx) tuples.
    """
    folds = []
    for tissue_id in holdout_tissue_ids:
        test_mask = np.isin(groups, tissue_id)
        test_idx = np.where(test_mask)[0]
        trainval_idx = np.where(~test_mask)[0]
        
        if len(test_idx) == 0:
            raise ValueError(
                f"No samples found for holdout tissue IDs: {holdout_tissue_ids}. "
                f"Available tissue IDs in data: {np.unique(groups).tolist()}"
            )
        if len(trainval_idx) == 0:
            raise ValueError(
                f"No training samples found after holding out tissue IDs: {holdout_tissue_ids}"
            )

        folds.append((trainval_idx, test_idx))
    return folds