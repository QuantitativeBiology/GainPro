from pydantic import BaseModel, Field

class OptimizerConfig(BaseModel):
    lr: float = Field(default=1e-4)
    weight_decay: float = Field(default=0)
    
class SchedulerConfig(BaseModel):
    step: int
    gamma: float

class GainTrainingConfig(BaseModel):
    num_epochs: int
    patience: int
    min_delta: float
    batch_size: int
    generator_optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    generator_scheduler: SchedulerConfig | None = None
    discriminator_optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    discriminator_scheduler: SchedulerConfig | None = None
    alpha: float = Field(default=10)
    hint_rate: float = Field(default=0.5)

class AutoEncoderTrainingConfig(BaseModel):
    num_epochs: int
    patience: int
    min_delta: float
    batch_size: int
    lr: float
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)