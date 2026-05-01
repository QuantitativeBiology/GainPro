from utils.data.dataset import Data
from utils.writers.experiment_writer import ExperimentWriter

from evaluation.holdout_strategy import HoldoutStrategy
from evaluation.groupkfold_strategy import GroupKFoldStrategy
from evaluation.evaluation_strategy import EvaluationStrategy

class Evaluator:
    STRATEGIES = {
        "holdout": HoldoutStrategy,
        "groupkfold": GroupKFoldStrategy,
    }
    
    def __init__(
        self,
        strategy: EvaluationStrategy | str,
        experiment_writer: ExperimentWriter,
    ):
        if isinstance(strategy, str):
            strategy_class = self.STRATEGIES.get(strategy.lower())
            if strategy_class is None:
                available = list(self.STRATEGIES.keys())
                raise ValueError(
                    f"Unknown strategy '{strategy}'. "
                    f"Available: {available}"
                )
            self.strategy = strategy_class()
        else:
            self.strategy = strategy
        
        self.experiment_writer = experiment_writer
    
    def evaluate(
        self,
        imputer_factory,
        data: Data,
        **kwargs
    ) -> dict:
        return self.strategy.run(
            imputer_factory=imputer_factory,
            data=data,
            experiment_writer=self.experiment_writer,
            **kwargs
        )