from typing import List
from pydantic import BaseModel

from utils.configs.validation_config import (
    HoldoutValidationConfig,
    GroupKFoldValidationConfig
)
from utils.configs.model_entry_config import ModelEntryConfig

class BenchmarkConfig(BaseModel):
    n_runs: int
    initial_seed: int

    validation: HoldoutValidationConfig | GroupKFoldValidationConfig
    models: List[ModelEntryConfig]