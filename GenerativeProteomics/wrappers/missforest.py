import shutil
import numpy as np
import pandas as pd
from datetime import datetime
from multiprocessing import cpu_count

# r packages
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages
from rpy2.robjects.conversion import localconverter
import torch

from utils.data.dataset import Data
from utils.writers.experiment_writer import ExperimentWriter

class MissForestRImputationModel:
    def __init__(self) -> "MissForestRImputationModel":
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
            ntree=100,
            maxiter=10,
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
        experiment_writer: ExperimentWriter,
        strategy: str,
        num_folds: int = None,
    ) -> None:
        """
        Args:
            - strategy (str): Cross validation strategies. Available: Hold-out ("hold-out") and K-Fold ("k-fold").
        """
        print("Evaluating missForest...")
        
        if strategy == "hold-out":
            self.hold_out_cv(
                data=data,
                experiment_writer=experiment_writer
            )
        elif strategy == "k-fold":
            self.kfold_cv(
                num_folds=num_folds,
                data=data,
                experiment_writer=experiment_writer
            )
        else:
            raise ValueError(f"Invalid cross validation strategy. Available strategies: Hold-out ('hold-out') and K-Fold ('k-fold').")
    
    def hold_out_cv(
        self,
        data: Data,
        experiment_writer: ExperimentWriter,
    ) -> None:
        # Hold-out cross-validation
        with localconverter(ro.default_converter + pandas2ri.converter):
            missing_df = pd.DataFrame(
                data.missing.detach().cpu().numpy().astype("float64"),
                columns=data.feature_names
            )
            r_missing_df = ro.conversion.py2rpy(
                missing_df
            )

        hold_out_dir = experiment_writer.evaluation_dir / "hold-out"
        hold_out_dir.mkdir(parents=True, exist_ok=True)
        experiment_writer.metadata_writer.set_out_dir(hold_out_dir)
        print("started missforest calculation")
        experiment_writer.metadata_writer.set_start_time(datetime.now())
        res = self.missforest.missForest(
            xmis=r_missing_df,
            ntree=3,
            maxiter=2,
            parallelize=ro.StrVector(["forests"]),
            verbose=True,
        )
        experiment_writer.metadata_writer.set_end_time(datetime.now())
        experiment_writer.metadata_writer.save_metadata()
        print("ended missforest calculation")

        with localconverter(ro.default_converter + pandas2ri.converter):
            imputed = ro.conversion.rpy2py(res.rx2("ximp"))

        # imputed_df = pd.DataFrame(
        #     imputed,
        #     # index=data.missing.index,
        #     # columns=data.missing.columns
        # )

        # compute RMSE on induced missing entries
        mask = ~np.isnan(data.reference.detach().cpu().numpy().astype("float64")) & np.isnan(data.missing.detach().cpu().numpy().astype("float64"))
        # x_true = pd.DataFrame(data.reference.cpu()) #log2(x+1) transformed
        # print("x true", x_true)
        # x_pred = imputed_df

        print("Computing rmse...")
        # rmse = np.sqrt(np.mean((x_true[mask] - x_pred[mask]) ** 2))
        x_true_np = data.reference.detach().cpu().numpy()
        x_pred_np = imputed

        diff = x_true_np - x_pred_np
        diff = diff[mask]
        rmse = np.sqrt(np.mean(diff ** 2))
        print("rmse:", rmse)


        # revert the logarithm of base 2
        x_true_log2p1_inverse = pd.DataFrame(
            np.power(2, x_true_np)-1,
        )
        x_pred_log2p1_inverse = pd.DataFrame(
            np.power(2, x_pred_np)-1,
        )
        print("imputed", x_pred_log2p1_inverse)
        
        experiment_writer.evaluation_writer.set_evaluation_dir(
            eval_dir=hold_out_dir
        )
        print("Saving files...")
        experiment_writer.evaluation_writer.save_hold_out_cv(
            mask=pd.DataFrame(mask),
            true_matrix=x_true_log2p1_inverse,
            pred_matrix=x_pred_log2p1_inverse,
            rmse=rmse,
        )

        self.shutdown_parallel()

    def kfold_cv(
        self,
        num_folds: int,
        data: Data,
        experiment_writer: ExperimentWriter,
    ) -> None:
        pass