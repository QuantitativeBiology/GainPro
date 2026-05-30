from pydantic import BaseModel, Field, ConfigDict

class GainConfig(BaseModel):
    name: str
    num_hidden_layers_generator: int = Field(default=1)
    num_hidden_layers_discriminator: int = Field(default=1)
    hidden_dim: int = Field(default=1024)

class MissForestConfig(BaseModel):
    name: str
    n_tree: int = Field(default=1)
    max_iter: int = Field(default=1)

class AutoEncoderConfig(BaseModel):
    name: str
    hidden_dims: list[int]
    latent_dim: int = Field(default=256)

class GlobalMeanConfig(BaseModel):
    name: str