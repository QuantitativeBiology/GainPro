from utils.data.dataset import Data
from utils.model_hypers import ModelHypers
from utils.train_hypers import TrainHypers
from wrappers.gain import GainImputationModel
from wrappers.mean import MeanImputationModel
from wrappers.mice import IterativeMICEImputationModel
from wrappers.missforest import MissForestRImputationModel
from utils.writers.experiment_writer import ExperimentWriter


class ImputationManager:
    def __init__(
        self,
        input_dim: int,
        experiment_writer: ExperimentWriter,
        model_cfg: dict=None,
        train_cfg: dict=None,
    ) -> "ImputationManager":
        self.imputation_methods = {
            "protogain": GainImputationModel(
                input_dim=input_dim,
                model_hypers=ModelHypers(model_cfg),
                train_hypers=TrainHypers(train_cfg)
            ),
            "mean": MeanImputationModel(), 
            "missForest": MissForestRImputationModel(), 
            "mice": IterativeMICEImputationModel(),
        }
        self.experiment_writer = experiment_writer

    def add_method(
        self, 
        model: str, 
        fn,
    ) -> None: 
        if model in self.imputation_methods:
            raise SystemExit ("Method already exists")
        else:
            self.imputation_methods.update({model:fn})

    def run_train(
        self, 
        model_name: str,
        dataset_name: str,
        data: Data,
    ):
        if model_name not in self.imputation_methods:
            raise SystemExit (f"Unknown {model_name} model. \n Models available are: {','.join(self.imputation_methods)}")
        else:
            return self.imputation_methods[model_name].train(
                dataset_name=dataset_name, 
                experiment_writer=self.experiment_writer,
                data=data,
            )
    
    def run_evaluate(
        self,
        model_name: str,
        data: Data,
        strategy: str,
    ):
        if model_name not in self.imputation_methods:
            raise SystemExit (f"Unknown {model_name} model. \n Models available are: {','.join(self.imputation_methods)}")
        else:
            return self.imputation_methods[model_name].evaluate(
                experiment_writer=self.experiment_writer,
                data=data,
                strategy=strategy,
            )









