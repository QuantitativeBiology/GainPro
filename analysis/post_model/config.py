from enum import IntEnum
from dataclasses import dataclass, field
from typing import Dict
import seaborn as sns
from matplotlib.lines import Line2D

MODELS = [
    "Mean", 
    "MissForest", 
    "ProtoGain", 
    "AutoEncoder", 
    "Tissue Mean"
]

MARKERS = [m for m in Line2D.markers if m not in ("None", None, " ", "")]

class EntryState(IntEnum):
    MISSING = 0
    OBSERVED = 1
    ARTIFICIAL_MISSING = 1


@dataclass(frozen=True)
class PlotConfig:
    marker_map: Dict[str, str] = field(default_factory=lambda: dict(zip(MODELS, MARKERS)))
    model_color: Dict[str, tuple] = field(default_factory=lambda: dict(zip(
        MODELS,
        sns.color_palette("Paired", n_colors=len(MODELS)),
    )))