from pydantic import BaseModel, Field

class OptimizerConfig(BaseModel):
    lr: float = Field(default=1e-4, ge=0.0)
    weight_decay: float = Field(default=0, ge=0.0)
    
class SchedulerConfig(BaseModel):
    step: int = Field(ge=0.0)
    gamma: float = Field(default=0.1, ge=0.0, le=1.0)

class GainTrainingConfig(BaseModel):
    num_epochs: int = Field(ge=1)
    patience: int = Field(ge=1)
    min_delta: float = Field(ge=0.0)
    batch_size: int = Field(ge=1)
    generator_optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    generator_scheduler: SchedulerConfig | None = None
    discriminator_optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    discriminator_scheduler: SchedulerConfig | None = None
    alpha: float = Field(default=10)
    hint_rate: float = Field(default=0.5, ge=0.0, le=1.0)

class AutoEncoderTrainingConfig(BaseModel):
    num_epochs: int = Field(ge=1)
    patience: int = Field(ge=1)
    min_delta: float = Field(ge=0.0)
    batch_size: int = Field(ge=1)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)