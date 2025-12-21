import os
import errno
from pathlib import Path
import pandas as pd
from abc import abstractmethod, ABC

class BasePreprocessor(ABC):
    # Abstract class
    def __init__(cls, in_dir: Path):
    
        if in_dir != None and not in_dir.is_dir():
            raise NotADirectoryError(errno.ENOTDIR, os.strerror(errno.ENOTDIR), in_dir)
        
        cls.in_dir = in_dir
        cls.scaler = None
        
    def load_csv(cls, file_path: Path) -> pd.DataFrame:
        df = pd.read_csv(file_path, index_col=0)  # samples as rows (obs.) and proteins as columns
        return df
    
    def load_reference(cls) -> pd.DataFrame:
        return cls.load_csv(f"{cls.in_dir}/reference.csv")

    def load_missing(cls) -> pd.DataFrame:
        return cls.load_csv(f"{cls.in_dir}/missing.csv")
    
    def load_mask(cls) -> pd.DataFrame:
        return cls.load_csv(f"{cls.in_dir}/mask.csv")
    
    def load_domain(cls) -> pd.DataFrame:
        return cls.load_csv(f"{cls.in_dir}/domain.csv")
    
    def load_domain_mapped(cls) -> pd.DataFrame:
        return cls.load_csv(f"{cls.in_dir}/domain_mapped.csv")
    
    @abstractmethod
    def normalize(cls, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def run(cls):
        pass