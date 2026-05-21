from wrappers.imputer import Imputer
from wrappers.gain import GainImputer
from wrappers.ae import AutoEncoderImputer
from wrappers.global_mean import GlobalMeanImputer
from wrappers.tissue_mean import TissueMeanImputer
from wrappers.missforest import MissForestRImputer


class ImputationManager:
    """Owns the imputer registry and is responsible for constructing imputers
    from config"""

    REGISTRY: dict[str, type] = {
        "protogain": GainImputer,
        "autoencoder": AutoEncoderImputer,
        "global_mean": GlobalMeanImputer,
        "tissue_mean": TissueMeanImputer,
        "missForest": MissForestRImputer,
    }

    def __init__(self, model_cfg, data) -> None:
        self.model_cfg = model_cfg
        self.data = data
        self.cls = self.resolve(model_cfg.name)

    def resolve(self, model_name: str) -> type:
        cls = self.REGISTRY.get(model_name)
        if cls is None:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available: {list(self.REGISTRY)}."
            )
        return cls

    def build(self) -> Imputer:
        """Construct and return a fresh imputer instance from config."""
        return self.cls.from_config(self.model_cfg, self.data)