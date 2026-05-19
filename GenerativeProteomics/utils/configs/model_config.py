from pydantic import BaseModel, Field

class GainConfig(BaseModel):
    name: str
    num_hidden_layers_generator: int = Field(default=1)
    num_hidden_layers_discriminator: int = Field(default=1)
    hidden_dim: int = Field(default=1024)

class MissForestConfig(BaseModel):
    n_tree: int = Field(default=1)
    max_iter: int = Field(default=1)

class AutoEncoderConfig(BaseModel):
    hidden_dims: list[int]
    latent_dim: int = Field(default=256)