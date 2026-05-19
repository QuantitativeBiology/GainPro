from pathlib import Path
from pydantic import BaseModel

class ModelEntryConfig(BaseModel):
    name: str
    model_config_path: Path
    training_config_path: Path