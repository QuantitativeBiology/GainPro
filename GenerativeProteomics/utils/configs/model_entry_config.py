from pathlib import Path
from typing import Optional
from pydantic import BaseModel

class ModelEntryConfig(BaseModel):
    name: str
    model_cfg_path: Optional[Path] = None
    training_cfg_path: Optional[Path] = None