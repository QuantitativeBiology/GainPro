from pathlib import Path
from typing import Literal
from pydantic import BaseModel

class DatasetConfig(BaseModel):
    name: str
    dataset_path: Path
    normalizer: Literal["minmax", "standard", "none", "auto"] = "auto"
    log_transform: bool | Literal["auto"] = "auto"