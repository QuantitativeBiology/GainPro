import torch.nn as nn
from utils.data.normalizer import (
    Normalizer, 
    MinMaxNormalizer, 
    StandardNormalizer
)

NORMALIZER_REGISTRY: dict[str, type[Normalizer]] = {
    "minmax": MinMaxNormalizer,
    "standard": StandardNormalizer,
}

NORMALIZER_TO_ACTIVATION: dict[str, nn.Module] = {
    "standard": nn.Identity(),
    "minmax": nn.Sigmoid(),
}

def build_normalizer(name: str) -> Normalizer:
    normalizer = NORMALIZER_REGISTRY.get(name)
    if normalizer is None:
        raise ValueError(
            f"Unknown normalizer '{name}'. "
            f"Available: {list(NORMALIZER_REGISTRY)}."
        )
    return normalizer()  

def get_output_activation(normalizer: Normalizer) -> nn.Module:
    activation = NORMALIZER_TO_ACTIVATION.get(normalizer.name)
    if activation is None:
        raise ValueError(
            f"No output activation registered for normalizer '{normalizer.name}'."
            f"Register it in NORMALIZER_TO_ACTIVATION."
        )
    return activation