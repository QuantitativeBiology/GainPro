from enum import IntEnum
from dataclasses import dataclass, field
from typing import Dict

class EntryState(IntEnum):
    MISSING = 0
    OBSERVED = 1
    ARTIFICIAL_MISSING = 1


@dataclass(frozen=True)
class PlotConfig:
    marker_map: Dict[str, str] = field(default_factory=lambda: {
        "Mean": "^",
        "MissForest": "#",
        "ProtoGain": "*",
        "AutoEncoder": "o",
        "Tissue Mean": "+",
    })