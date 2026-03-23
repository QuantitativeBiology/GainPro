from utils.data.dataset import Data
from models.GainPro.gain import Gain
from models.GainPro.trainer import Trainer
from utils.model_hypers import GainHypers
from utils.train_hypers import TrainHypers
from utils.writers.experiment_writer import ExperimentWriter

class GainImputationModel:
    def __init__(
        self,
        input_dim: int,
        gain_hypers: GainHypers,
        train_hypers: TrainHypers,
    ) -> "GainImputationModel":
        self.gain = Gain(
            input_dim=input_dim,
            num_hidden_layers_generator=gain_hypers.num_hidden_layers_generator,
            num_hidden_layers_discriminator=gain_hypers.num_hidden_layers_discriminator
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
        strategy: str,
        data: Data,
        experiment_writer: ExperimentWriter,
        idxs_folds: list=None,
        num_folds: int=None, #todo para o caso de não existir um split a priori
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
                idxs_folds=idxs_folds,
                num_folds=num_folds,
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