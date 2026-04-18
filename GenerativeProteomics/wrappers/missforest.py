import torch
import shutil
import numpy as np
import pandas as pd
from datetime import datetime
from multiprocessing import cpu_count

from sklearn.model_selection import StratifiedGroupKFold

from missingpy import MissForest

# r packages
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.conversion import localconverter

from utils.data.dataset import Data
from utils.model_hypers import MissForestHypers
from utils.writers.experiment_writer import ExperimentWriter

class MissForestRImputationModel:
    def __init__(
        self,
        missforest_hypers: MissForestHypers,
    ) -> "MissForestRImputationModel":
        if shutil.which("R") is None:
            raise RuntimeError(
                "R executable not found. You must install R (https://cran.r-project.org/) "
                "and add it to your system PATH to use this model."
            )

        try:
            # locally:
            # rpackages.importr("utils").install_packages(
            #     "missForest",
            #     repos="https://cloud.r-project.org"
            # )
            # rpackages.importr("utils").install_packages(
            #     "doParallel",
            #     repos="https://cloud.r-project.org"
            # )
            # rpackages.importr("utils").install_packages(
            #     "foreach",
            #     repos="https://cloud.r-project.org"
            # )
            self.missforest = importr("missForest")
            self.doParallel = importr("doParallel")
            self.parallel = importr("parallel")
            
            self.n_cores = cpu_count()
            ro.r(f"""
                library(doParallel)
                library(parallel)
                cl <- makeCluster({self.n_cores})
                registerDoParallel(cl)
            """)
            self.n_tree = missforest_hypers.n_tree
            self.max_iter = missforest_hypers.max_iter
            print("Number of trees:", self.n_tree)
            print("Maximum iterations:", self.max_iter)

        except Exception as e:  
            print("The 'missForest' R package is not installed. Please install it in your R environment.")
            raise e
        
    def shutdown_parallel(self):
        try:
            ro.r("""
            if (exists("cl")) {
                stopCluster(cl)
                rm(cl)
            }
            """)
            print("R parallel cluster stopped.")
        except Exception as e:
            print("Error stopping R cluster:", e)

    def train(
        self,
        data: Data,
        dataset_name: str,
        experiment_writer: ExperimentWriter,
    ) -> pd.DataFrame:
        print("MissForest training...\n")

        # pandas -> R
        with localconverter(ro.default_converter + pandas2ri.converter):
            print("Data missing\n", data.missing)
            r_missing_df = ro.conversion.py2rpy(data.missing)
            print("Data reference\n", data.reference)
            r_reference_df = ro.conversion.py2rpy(data.reference)

        print("Running missForest...")

        experiment_writer.metadata_writer.set_out_dir(experiment_writer.train_dir)
        experiment_writer.metadata_writer.set_start_time(datetime.now())
        res = self.missforest.missForest(
            xmis=r_missing_df,
            xtrue=r_reference_df,
            ntree=self.n_tree,
            maxiter=self.max_iter,
            parallelize=ro.StrVector(["forests"]),
            verbose=True,
        )
        experiment_writer.metadata_writer.set_end_time(datetime.now())
        experiment_writer.metadata_writer.save_metadata()

        # R -> pandas
        with localconverter(ro.default_converter + pandas2ri.converter):
            out = ro.conversion.rpy2py(res.rx2("ximp"))
            error = ro.conversion.rpy2py(res.rx2("OOBerror")) # Normalized Root Mean Squared Error (NRMSE)
        
        print("Error", error)
        print("Total missing:", np.isnan(data.missing).sum())
        print("Columns fully missing:", np.where(np.isnan(data.missing).mean(axis=0) == 1)[0])
        print("Zero variance columns:", np.where(np.var(data.missing, axis=0) == 0)[0])
        
        experiment_writer.result_writer.save_test_rmse(rmse=error) # NRMSE

        print("Imputation completed.")

        imputed_dataset = pd.DataFrame(out, index=data.missing.index, columns=data.reference.columns)

        file_path = experiment_writer.preds_dir / f"{dataset_name}_imputed.csv"
        imputed_dataset.to_csv(file_path, index=True)

        self.shutdown_parallel()
        
        return imputed_dataset
    
    def evaluate(
        self,
        data: Data,
        strategy: str,
        experiment_writer: ExperimentWriter,
        num_folds: int = None,
    ) -> None:
        """
        Args:
            - strategy (str): Cross validation strategies. Available: Holdout ("holdout") and K-Fold ("k-fold").
        """
        print("Evaluating missForest...")
        
        if strategy == "holdout":
            self.holdout_cv(
                data=data,
                experiment_writer=experiment_writer
            )
        elif strategy == "group-k-fold":
            self.group_kfold_cv(
                data=data,
                experiment_writer=experiment_writer,
                num_folds=num_folds,
            )
        else:
            raise ValueError(f"Invalid cross validation strategy. Available strategies: Holdout ('holdout') and K-Fold ('k-fold').")
    
    


    def holdout_cv(
        self,
        data: Data,
        experiment_writer: ExperimentWriter,
    ) -> None:

        with localconverter(ro.default_converter + pandas2ri.converter):
            missing_df = pd.DataFrame(
                data.missing.detach().cpu().numpy().astype("float64"),
                columns=data.feature_names
            )
            r_missing_df = ro.conversion.py2rpy(
                missing_df
            )

        hold_out_dir = experiment_writer.evaluation_dir / "holdout"
        hold_out_dir.mkdir(parents=True, exist_ok=True)
        experiment_writer.metadata_writer.set_out_dir(hold_out_dir)
        print("Started missforest calculation")
        experiment_writer.metadata_writer.set_start_time(datetime.now())
        res = self.missforest.missForest(
            xmis=r_missing_df,
            ntree=self.n_tree,
            maxiter=self.max_iter,
            parallelize=ro.StrVector(["forests"]),
            verbose=True,
        )
        experiment_writer.metadata_writer.set_end_time(datetime.now())
        experiment_writer.metadata_writer.save_metadata()
        print("Ended missforest calculation")

        with localconverter(ro.default_converter + pandas2ri.converter):
            imputed = ro.conversion.rpy2py(res.rx2("ximp"))

        print("First imputed\n", imputed)
        print("First imputed to numpy\n", imputed.to_numpy())

        # Compute RMSE on induced missing entries
        artificial_missing_mask = ~data.artificial_missing_mask.detach().cpu().numpy()
        print("mask", artificial_missing_mask)

        print("Computing RMSE...")
        x_true = data.reference
        print("X true\n", x_true)
        x_pred = torch.tensor(imputed.to_numpy(), device=x_true.device)
        print("First imputed tensor\n", x_pred)
        rmse = torch.sqrt(torch.mean((x_true[artificial_missing_mask] - x_pred[artificial_missing_mask]) ** 2))
        rmse = rmse.item()
        print(f"RMSE: {rmse:.4f}")

        x_true_np = data.reference.detach().cpu().numpy()
        x_pred_np = imputed.to_numpy()

        # Invert normalization
        max_norm = data.max_norm.values
        min_norm = data.min_norm.values
        x_pred = x_pred_np * (max_norm - min_norm) + min_norm
        x_true_test = x_true_np * (max_norm - min_norm) + min_norm

        # Invert log2(x + 1) from dataset_builder.py: x = 2^y - 1
        x_true_log2p1_inverse = pd.DataFrame(
            np.power(2, x_true_test)-1,
            index=np.array(data.sample_names),
            columns=data.feature_names,
        )
        x_pred_log2p1_inverse = pd.DataFrame(
            np.power(2, x_pred)-1,
            index=np.array(data.sample_names),
            columns=data.feature_names,
        )
        print("Imputed\n", x_pred_log2p1_inverse)
        
        experiment_writer.evaluation_writer.set_evaluation_dir(
            eval_dir=hold_out_dir
        )

        observed_mask = pd.DataFrame(
            data.observed_mask.detach().cpu().numpy(),
            index=np.array(data.sample_names),
            columns=data.feature_names,
        )

        artificial_missing_mask = pd.DataFrame(
            data.artificial_missing_mask.detach().cpu().numpy(),
            index=np.array(data.sample_names),
            columns=data.feature_names,
        )

        tissue_mask = pd.DataFrame(
            data.tissue.detach().cpu().numpy(),
            index=np.array(data.sample_names),
        )

        experiment_writer.evaluation_writer.save_missforest_predictions(
            true_matrix=x_true_log2p1_inverse,
            observed_mask=observed_mask,
            pred_matrix=x_pred_log2p1_inverse,
            artificial_missing_mask=artificial_missing_mask,
            tissue_matrix=tissue_mask,
            rmse=rmse,
        )

        self.shutdown_parallel()

    def group_kfold_cv(
        self,
        data: Data,
        experiment_writer: ExperimentWriter,
        num_folds: int,
    ) -> None:
        sgkf = StratifiedGroupKFold(n_splits=num_folds)
        groups = data.tissue.cpu().numpy()
        y = groups
        
        folds_dir = experiment_writer.evaluation_dir / "groupkfold"
        folds_dir.mkdir(parents=True, exist_ok=True)

        reference_np = data.reference.cpu().detach().numpy()
        observed_mask_np = data.observed_mask.cpu().detach().numpy()

        for fold_id, (train_idx, test_idx) in enumerate(
            sgkf.split(X=reference_np, y=y, groups=groups), start=1
        ):
            print(f"\n\n------------ Fold {fold_id}/{num_folds} ------------\n")

            fold_dir = folds_dir / f"fold_{fold_id}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            experiment_writer.metadata_writer.set_out_dir(fold_dir)

            train_reference = reference_np[train_idx, :]

            print("Fitting missForest...")
            imputer = MissForest(
                n_estimators=self.n_tree,
                max_iter=self.max_iter,
                max_features=None,
                criterion=("squared_error")  
            )
            experiment_writer.metadata_writer.set_start_time(datetime.now())
            imputer.fit(train_reference)

            # Test each sample as completely unobserved
            _, n_features = reference_np[test_idx, :].shape
            observed_mask_test = observed_mask_np[test_idx, :]

            n_test, n_features = reference_np[test_idx, :].shape

            test_anchor = np.full((n_test, n_features), np.nan)
            rng = np.random.default_rng(seed=42)

            # print("reference", reference_np)
            # print(reference_np.shape)

            for col in range(n_features):
                observed_rows = np.where(observed_mask_test[:, col])[0]
                if len(observed_rows) == 0:
                    continue
                anchor_row = rng.choice(observed_rows, size=1)
                # print("test idx row value", reference_np[test_idx[anchor_row], col])
                test_anchor[anchor_row, col] = reference_np[test_idx[anchor_row], col]

            # print("test anchor", test_anchor)

            test_imputed = imputer.transform(test_anchor)
            experiment_writer.metadata_writer.set_end_time(datetime.now())
            experiment_writer.metadata_writer.save_metadata()
            print("Ended!\n")

            print("Computing the error (RMSE)...")
            test_reference = reference_np[test_idx, :]
            test_observed_mask = observed_mask_np[test_idx, :].astype(bool)

            anchor_mask = ~np.isnan(test_anchor)
            eval_mask = test_observed_mask & ~anchor_mask

            diff = test_reference[eval_mask] - test_imputed[eval_mask]
            rmse = np.sqrt(np.mean(diff ** 2))
            print(f"RMSE (fold {fold_id}): {rmse:.4f}")

            # Invert the normalization
            max_norm = data.max_norm.values
            min_norm = data.min_norm.values
            x_hat = test_imputed * (max_norm - min_norm) + min_norm # inverse normalization
            x_true_test = test_reference * (max_norm - min_norm) + min_norm # inverse normalization

            # Invert the log2(x + 1) from dataset_builder.py: x = 2^y - 1
            x_true_log2p1_inverse = pd.DataFrame(
                np.power(2, x_true_test)-1,
                index=np.array(data.sample_names)[test_idx],
                columns=data.feature_names,
            )
            x_pred_log2p1_inverse = pd.DataFrame(
                np.power(2, x_hat)-1,
                index=np.array(data.sample_names)[test_idx],
                columns=data.feature_names,
            )
            print("imputed", x_pred_log2p1_inverse)

            experiment_writer.evaluation_writer.set_evaluation_dir(
                eval_dir=fold_dir
            )
            print("Saving files...")

            test_idx_np = test_idx.cpu().numpy() if hasattr(test_idx, 'cpu') else np.array(test_idx)

            observed_mask = pd.DataFrame(
                data.observed_mask[test_idx, :].detach().cpu().numpy(),
                index=np.array(data.sample_names)[test_idx_np],
                columns=data.feature_names,
            )

            artificial_missing_mask = pd.DataFrame(
                data.artificial_missing_mask[test_idx, :].detach().cpu().numpy(),
                index=np.array(data.sample_names)[test_idx_np],
                columns=data.feature_names,
            )

            tissue_mask = pd.DataFrame(
                data.tissue[test_idx].detach().cpu().numpy(),
                index=np.array(data.sample_names)[test_idx_np],
                columns=["tissue"]
            )

            experiment_writer.evaluation_writer.save_missforest_predictions(
                true_matrix=x_true_log2p1_inverse,
                observed_mask=observed_mask,
                pred_matrix=x_pred_log2p1_inverse,
                artificial_missing_mask=artificial_missing_mask,
                tissue_matrix=tissue_mask,
                tissue_mapping=data.tissue_mapping,
                rmse=rmse,
            )

            experiment_writer.split_writer.save_fold_splits(
                fold_id=fold_id,
                train_idx=train_idx,
                test_idx=test_idx,
            )
            
        self.shutdown_parallel()