from pathlib import Path

from utils.writers.results_writer import ResultWriter
from utils.writers.metrics_writer import MetricsWriter
from utils.writers.metadata_writer import MetadataWriter
from utils.writers.evaluation_writer import EvaluationWriter

class ExperimentWriter:
    def __init__(
        self,
        run_dir: Path,
    ) -> "ExperimentWriter":
        self.run_dir = run_dir

        self.cfg_dir = self.run_dir / "configs"
        self.data_dir = self.run_dir / "data"
        self.split_dir = self.run_dir / "splits"
        self.train_dir = self.run_dir / "training"
        self.metrics_dir = self.train_dir / "metrics"
        self.preds_dir = self.run_dir / "predictions"
        self.evaluation_dir = self.run_dir / "evaluation"
        self.folds_dir = self.split_dir / "folds"

        for dir in [
            self.cfg_dir,
            self.data_dir,
            self.split_dir,
            self.train_dir,
            self.metrics_dir,
            self.preds_dir,
            self.evaluation_dir,
            self.folds_dir,
        ]:
            dir.mkdir(parents=True, exist_ok=True)
        
        self.result_writer = ResultWriter(preds_dir=self.preds_dir, evaluation_dir=self.evaluation_dir)
        self.evaluation_writer = EvaluationWriter(self.evaluation_dir)
        self.metrics_writer = MetricsWriter(self.metrics_dir)
        self.metadata_writer = MetadataWriter()
