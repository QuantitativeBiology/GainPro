
class GainHypers:
    def __init__(
        self,
        model_cfg: dict,
    ) -> "GainHypers":
        self.num_hidden_layers_generator = model_cfg["generator"]["num_layers"]
        self.num_hidden_layers_discriminator = model_cfg["discriminator"]["num_layers"]

class MissForestHypers:
    def __init__(
        self,
        model_cfg: dict,
    ) -> "MissForestHypers":
        self.n_tree = model_cfg["n_tree"]
        self.max_iter = model_cfg["max_iter"]