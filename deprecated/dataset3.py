import torch
import pandas as pd

class Data:
    def __init__(
        self,
        miss_rate: float,
        reference: pd.DataFrame,
        missing: pd.DataFrame,
        mask: pd.DataFrame,
    ) -> Data:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.miss_rate = miss_rate

        self.reference = reference
        self.missing = missing
        self.mask = mask