from typing import Dict
from pydantic import BaseModel, Field
from pathlib import Path

class MergeConfig(BaseModel):
    datasets: Dict[str, Path] = Field(min_length=2)
    output: Path