import yaml
from pathlib import Path
from datetime import datetime

class ExperimentManager():
    def __init__(
        cls,
        config: dict,
    ):
        cls.config = config
        cls.root_dir = Path(f"experiments/{cls.config['experiment']['name']}")
        cls.root_dir.mkdir(parents=True, exist_ok=True)
        cls.run_dir = None

    def _timestamp(cls) -> str:
        return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    
    def create_experiment(cls) -> None:
        """Create experiment directory and return its path"""
        cls.run_dir = cls.root_dir / f"{cls._timestamp()}"
        cls.run_dir.mkdir(parents=True)

        # Freeze config
        with open(cls.run_dir / "experiment_config.yaml", "w") as f:
            yaml.safe_dump(cls.config, f)

        # Metadata
        metadata = {
            "created_at": cls._timestamp(),
        }
        with open(cls.run_dir / "experiment_metadata.yaml", "w") as f:
            yaml.safe_dump(metadata, f)

        checkpoints_dir = cls.run_dir / "checkpoints" # stores the trained model
        checkpoints_dir.mkdir(parents=True)

        training_metrics_dir = cls.run_dir / "training"
        training_metrics_dir.mkdir(parents=True)
        
        plots_dir = cls.run_dir / "plots"
        plots_dir.mkdir(parents=True)

        training_dir = plots_dir / "training"
        training_dir.mkdir(parents=True)

        latent_dir = plots_dir / "latent"
        latent_dir.mkdir(parents=True)