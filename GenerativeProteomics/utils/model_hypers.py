
class ModelHypers:
    def __init__(
        self,
        model_cfg: dict
    ) -> "ModelHypers":
        self.generator_n_hidden_layers = model_cfg["generator"]["num_layers"]
        self.discriminator_n_hidden_layers = model_cfg["discriminator"]["num_layers"]