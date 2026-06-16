import json

class Hypers:

    def __init__( #todo o params não devia estar a receber informação de i/o (ou seja o path dos files, etc...)
        self,
        input=None,
        output="imputed",
        ref=None,
        output_folder=f"../results/",
        # pretrained=None,
        header=None,
        num_iterations=2001,
        batch_size=128,
        alpha=10,
        miss_rate=0.1,
        hint_rate=0.9,
        lr_D=0.001,
        lr_G=0.001,
        override=0,
        output_all=0,
        save=False,
    ):
        self.input = input
        self.output = output
        self.output_folder = output_folder
        self.ref = ref
        # self.pretrained = pretrained
        self.header = header
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.alpha = alpha
        self.miss_rate = miss_rate
        self.hint_rate = hint_rate
        self.lr_D = lr_D
        self.lr_G = lr_G
        self.override = override
        self.output_all = output_all
        self.save = save

    @staticmethod
    def _read_json(json_file):
        with open(json_file, "r") as f:
            params = json.load(f)
        return params

    @classmethod
    def read_hyperparameters(cls, params_json=None):

        params = cls._read_json(params_json)

        print("\n", params)

        input = params["input"]
        output = params["output"]
        ref = params["ref"]
        # pretrained = params["pretrained"]
        output_folder = params["output_folder"]
        num_iterations = params["num_iterations"]
        batch_size = params["batch_size"]
        alpha = params["alpha"]
        miss_rate = params["miss_rate"]
        hint_rate = params["hint_rate"]
        lr_D = params["lr_D"]
        lr_G = params["lr_G"]
        override = params["override"]
        output_all = params["output_all"]
        save = params["save"]

        return cls(
            input,
            output,
            ref,
            # pretrained,
            output_folder,
            None,
            num_iterations,
            batch_size,
            alpha,
            miss_rate,
            hint_rate,
            lr_D,
            lr_G,
            override,
            output_all,
            save
        )

    def update_hypers(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"{key} is not a valid parameter")
