from typing import Dict
from pathlib import Path
from pydantic import BaseModel, Field

class MergeConfig(BaseModel):
    datasets: Dict[str, Path] = Field(min_length=2)
    output: Path