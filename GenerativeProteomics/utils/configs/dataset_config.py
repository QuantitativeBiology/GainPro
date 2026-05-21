from pathlib import Path
from pydantic import BaseModel

class DatasetConfig(BaseModel):
    name: str
    dataset_path: Path