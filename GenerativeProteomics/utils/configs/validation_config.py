from typing import List
from pydantic import BaseModel

class HoldoutValidationConfig(BaseModel):
    name: str
    target_missing: List[float]

class GroupKFoldValidationConfig(BaseModel):
    name: str
    target_missing: List[float]