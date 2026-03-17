from utils.data.dataset import Data
from models.GainPro.gain import Gain
from models.GainPro.trainer import Trainer
from utils.model_hypers import ModelHypers
from utils.train_hypers import TrainHypers
from utils.writers.experiment_writer import ExperimentWriter

class GainImputationModel:
    def __init__(
        self,
        input_dim: int,
        model_hypers: ModelHypers,
        train_hypers: TrainHypers,
    ) -> "GainImputationModel":
        self.gain = Gain(
            input_dim=input_dim,
            model_hypers=model_hypers,
        )

        self.trainer = Trainer(
            model=self.gain,
            train_hypers=train_hypers
        )

    def train(
        self,
        experiment_writer: ExperimentWriter,
    ) -> None:
        pass

    def evaluate(
        self,
        data: Data,
        experiment_writer: ExperimentWriter,
        strategy: str,
        num_folds: int = None,
    ) -> None:
        """
        Args:
            - strategy (str): Cross validation strategies. Available: Hold-out ("hold-out") and K-Fold ("k-fold").
        """
        print("Evaluating gain...")
        
        if strategy == "hold-out":
            self.trainer.evaluate(
                data=data,
                strategy=strategy,
                experiment_writer=experiment_writer,
            )
        elif strategy == "k-fold":
            self.trainer.evaluate(
                strategy=strategy,
                num_folds=2, #todo hardcoded
                data=data,
                experiment_writer=experiment_writer
            )
        elif strategy == "group-k-fold":
            self.trainer.evaluate(
                strategy=strategy,
                num_folds=2, #todo hardcoded
                data=data,
                experiment_writer=experiment_writer
            )
        else:
            raise ValueError(f"Invalid cross validation strategy. Available strategies: Hold-out ('hold-out'), K-Fold ('k-fold') and Stratified Group K-Fold ('group-k-fold').")