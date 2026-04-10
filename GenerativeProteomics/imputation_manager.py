from utils.data.dataset import Data
from utils.model_hypers import GainHypers, MissForestHypers
from utils.train_hypers import TrainHypers
from wrappers.gain import GainImputationModel
from wrappers.mean import MeanImputationModel
from wrappers.mice import IterativeMICEImputationModel
from wrappers.missforest import MissForestRImputationModel
from utils.writers.experiment_writer import ExperimentWriter


class ImputationManager:
    def __init__(
        self,
        experiment_writer: ExperimentWriter,
        model_name: str,
        input_dim: int=None,
        model_cfg: dict=None,
        train_cfg: dict=None,
    ) -> "ImputationManager":
        self.initialize_model(
            model_name=model_name,
            input_dim=input_dim,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
        )
        self.experiment_writer = experiment_writer

    def initialize_model(
        self,
        model_name: str,
        input_dim: int=None,
        model_cfg: dict=None,
        train_cfg: dict=None,
    ) -> None:
        if model_name == "protogain":
            self.model = GainImputationModel(
                input_dim=input_dim,
                gain_hypers=GainHypers(model_cfg),
                train_hypers=TrainHypers(train_cfg),
            )
        elif model_name == "mean":
            self.model = MeanImputationModel()
        elif model_name == "missForest":
            self.model = MissForestRImputationModel(
                missforest_hypers=MissForestHypers(model_cfg)
            )
        elif model_name == "mice":
            self.model = IterativeMICEImputationModel()
        else:
            raise ValueError(f"Unknown {model_name} model. Models available are: protogain, mean, missForest and mice.")

    def run_train(
        self,
        dataset_name: str,
        data: Data,
    ):
        return self.model.train(
            dataset_name=dataset_name, 
            experiment_writer=self.experiment_writer,
            data=data,
        )
    
    def run_evaluate(
        self,
        strategy: str,
        idxs_folds: list,
        data: Data,
        num_folds: int = None,
    ):
        return self.model.evaluate(
            strategy=strategy,
            data=data,
            idxs_folds=idxs_folds,
            experiment_writer=self.experiment_writer,
            num_folds=num_folds,
        )









