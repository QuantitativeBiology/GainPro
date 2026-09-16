import numpy as np
from abc import ABC, abstractmethod

from GenerativeProteomics.utils.data.dataset import Data
from GenerativeProteomics.utils.writers.experiment_writer import ExperimentWriter

class EvaluationStrategy(ABC):
    @abstractmethod
    def run(
        self,
        imputer_factory,
        data: Data,
        experiment_writer: ExperimentWriter,
        **kwargs
    ) -> None:
        pass