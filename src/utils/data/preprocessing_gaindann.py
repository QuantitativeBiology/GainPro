import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from .base_preprocessing import BasePreprocessor

class GainDannPreprocessor(BasePreprocessor):
    def __init__(cls, in_dir: Path=None):
        super().__init__(in_dir)

        cls.scaler = StandardScaler()
    
    def is_fitted(cls):
        return hasattr(cls.scaler, "mean_") and hasattr(cls.scaler, "scale_")

    def normalize(cls, df: pd.DataFrame) -> pd.DataFrame:
        if not cls.is_fitted():
            x = cls.scaler.fit_transform(df.values)
        else:
            x = cls.scaler.transform(df.values)
        return pd.DataFrame(x, index=df.index, columns=df.columns)

    def run(cls) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        
        reference = cls.load_reference()
        missing = cls.load_missing()
        mask = cls.load_mask()
        domain = cls.load_domain()
        domain_mapped = cls.load_domain_mapped()

        reference = cls.normalize(reference)
        missing = cls.normalize(missing)

        return reference, missing, mask, domain, domain_mapped
