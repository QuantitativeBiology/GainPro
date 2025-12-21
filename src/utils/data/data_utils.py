import numpy as np
import pandas as pd
import logging
import torch
# from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Data():
    def __init__(
            cls,
            reference: pd.DataFrame,
            missing: pd.DataFrame=None,
            mask: pd.DataFrame=None,
            domain: pd.DataFrame=None,
            domain_mapped: pd.DataFrame=None,
            # scaler=None
            ):
        
        cls.reference = reference.astype(np.float32)
        cls.missing = missing.astype(np.float32)
        cls.mask = mask
        cls.domain = domain
        cls.domain_mapped = domain_mapped

        cls.n_samples = cls.reference.shape[0]
        cls.n_proteins = cls.reference.shape[1]

        if cls.domain is not None:
            cls.domain_labels = cls.domain_mapped["Domain"]
            cls.n_domains = cls._get_number_domains()
            cls.sample_to_project = cls._get_sample_to_domain() # mapping
        
        cls.samples_names = cls.get_samples_names()
        cls.protein_names = list(cls.reference.columns)

        # cls.scaler = scaler if scaler is not None else StandardScaler()
        # cls.dataset_normalized = cls.normalize_df(cls.reference)

    def _get_number_domains(cls) -> int:
        return len(np.unique(cls.domain_labels))

    def get_samples_names(cls) -> list:
        return cls.reference.index.tolist()
    
    def _get_sample_to_domain(cls):
        return dict(zip(cls.domain.index, cls.domain.values))
    
    def get_scaler(cls):
        return cls.scaler
    
    def to_tensors(cls) -> torch.tensor:
        X = torch.tensor(cls.missing.values, dtype=torch.float32)
        ref = torch.tensor(cls.reference.values, dtype=torch.float32)
        mask = torch.tensor(cls.mask.values, dtype=torch.bool)
        if cls.domain is not None:
            # Map project names to numeric domain labels
            domain_mapped = torch.tensor(cls.domain_mapped, dtype=torch.int)
            return X, ref, mask, domain_mapped
        return X, ref, mask