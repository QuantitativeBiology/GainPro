from pathlib import Path

from utils.writers.split_writer import SplitWriter
from utils.writers.results_writer import ResultWriter
from utils.writers.metadata_writer import MetadataWriter

class ExperimentWriter:
    def __init__(
        self,
        run_dir: Path,
    ) -> "ExperimentWriter":
        self.run_dir = run_dir

        self.cfg_dir = self.run_dir / "configs"
        self.data_dir = self.run_dir / "data"
        self.preds_dir = self.run_dir / "predictions"
        self.evaluation_dir = self.run_dir / "evaluation"
        self.split_dir = self.run_dir / "splits"
        self.folds_dir = self.split_dir / "folds"

        for dir in [
            self.cfg_dir,
            self.data_dir,
            self.preds_dir,
            self.evaluation_dir,
            self.folds_dir,
        ]:
            dir.mkdir(parents=True, exist_ok=True)
        
        self.result_writer = ResultWriter()
        self.metadata_writer = MetadataWriter()
