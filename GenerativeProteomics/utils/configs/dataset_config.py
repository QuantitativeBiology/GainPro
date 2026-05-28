from pathlib import Path
from typing import Literal
from pydantic import BaseModel

class DatasetConfig(BaseModel):
    name: str
    dataset_path: Path
    normalizer: Literal["minmax", "standard", "none"] = "minmax"
    log_transform: bool = False