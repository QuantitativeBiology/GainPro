from wrappers.imputer import Imputer
from wrappers.gain import GainImputer
from wrappers.ae import AutoEncoderImputer
from wrappers.global_mean import GlobalMeanImputer
from wrappers.tissue_mean import TissueMeanImputer
from wrappers.missforest import MissForestRImputer
from utils.configs.model_config import FillStrategy


class ImputationManager:
    """Owns the imputer registry and is responsible for constructing imputers
    from config"""

    REGISTRY: dict[str, type] = {
        "gain": GainImputer,
        "autoencoder": AutoEncoderImputer,
        "global_mean": GlobalMeanImputer,
        "tissue_mean": TissueMeanImputer,
        "missForest": MissForestRImputer,
    }

    def __init__(self, model_cfg) -> None:
        self.model_cfg = model_cfg
        self.data = None
        self.cls = self.resolve(model_cfg.name)

    def resolve(self, model_name: str) -> type:
        cls = self.REGISTRY.get(model_name)
        if cls is None:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available: {list(self.REGISTRY)}."
            )
        return cls
    
    @property
    def fill_strategy(self) -> FillStrategy:
        if hasattr(self.cls, "get_fill_strategy"):
            return self.cls.get_fill_strategy(self.model_cfg)
        return "none"
    
    def set_data(self, data) -> None:
        self.data = data

    def build(self) -> Imputer:
        """Construct and return a fresh imputer instance from config."""
        if self.data is None:
            raise RuntimeError("Call set_data() before build().")
        return self.cls.from_config(self.model_cfg, self.data)