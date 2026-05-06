class TrainHypers:
    def __init__(
        self,
        train_cfg: dict
    ) -> "TrainHypers":
        self.num_epochs = train_cfg["num_epochs"]
        self.batch_size = train_cfg["batch_size"]

        self.generator_lr = float(train_cfg["generator"]["lr"])
        self.discriminator_lr = float(train_cfg["discriminator"]["lr"])

        self.alpha = train_cfg["alpha"]
        self.hint_rate = train_cfg["hint_rate"]