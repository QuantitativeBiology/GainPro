from pydantic import BaseModel, Field

class GainTrainingConfig(BaseModel):
    num_epochs: int
    generator_lr: float = Field(default=1e-4)
    discriminator_lr: float = Field(default=1e-4)
    alpha: float = Field(default=10)
    hint_rate: float = Field(default=0.5)
    
class SchedulerConfig(BaseModel):
    step: int
    gamma: float

class AutoEncoderTrainingConfig(BaseModel):
    num_epochs: int
    lr: float
    scheduler: SchedulerConfig