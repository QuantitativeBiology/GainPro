import pandas as pd
import rpy2.robjects as ro
import shutil
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter


class MissForestRImputationModel:
    def run(self, df: pd.DataFrame) -> pd.DataFrame:

        if shutil.which("R") is None:
            raise RuntimeError(
                "R executable not found. You must install R (https://cran.r-project.org/) "
                "and add it to your system PATH to use this model."
            )

        try:
            missforest = importr("missForest")
        except Exception as e:  
            print("The 'missForest' R package is not installed. Please install it in your R environment.")
            raise e

        # pandas -> R
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_df = ro.conversion.py2rpy(df)

        res = missforest.missForest(
            xmis=r_df,
            ntree=500,
            maxiter=10,
            verbose=False,
        )

        # R -> pandas
        with localconverter(ro.default_converter + pandas2ri.converter):
            out = ro.conversion.rpy2py(res.rx2("ximp"))

        print ("MissForest imputation completed.")

        final_dataset = pd.DataFrame(out, index=df.index, columns=df.columns)

        print(final_dataset)
        final_dataset.to_csv("missforest_imputed_dataset.csv", index=False)

        return final_dataset
