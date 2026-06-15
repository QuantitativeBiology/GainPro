from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel, Field

from utils.data.helper import MissingMechanism

class DatasetConfig(BaseModel):
    name: str
    dataset_path: Path
    normalizer: Literal["minmax", "standard", "none", "auto"] = "auto"
    log_transform: bool | Literal["auto"] = "auto"
    missing_mechanism: MissingMechanism = Field(default=MissingMechanism.MNAR)
    steepness: Optional[float] = Field(default=1.0, gt=0.0, le=5.0)