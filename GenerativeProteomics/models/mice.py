import pandas as pd
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer

class IterativeMICEImputationModel:
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        imp = IterativeImputer(max_iter=10, random_state=42)
        out = imp.fit_transform(df)
        print(out)
        return pd.DataFrame(out, index=df.index, columns=df.columns)
    
