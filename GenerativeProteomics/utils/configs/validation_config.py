from typing import List
from pydantic import BaseModel

class HoldoutValidationConfig(BaseModel):
    name: str
    missing_levels: List[float]

class GroupKFoldValidationConfig(BaseModel):
    name: str
    missing_levels: List[float]