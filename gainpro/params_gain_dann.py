import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParamsGainDann:
    """
    Container for **Training Parameters** and **Hyperparameters** of the GAIN-DANN model.

    Args:
        - path_dataset (str): (Training parameter) Path to the dataset used for training.
        - num_epochs (int): (Training parameter) Number of training epochs.
        - num_folds (int): (Training parameter) Number of folds for cross-validation.
        - seed (int): (Training parameter) Random seed for reproducibility.

        - batch_size (int): (Hyperparameter) Number of samples per batch training.
        - learning_rate (float): (Hyperparameter) Learning rate for the optimizer.
        - weight_decay (float): (Hyperparameter) L2 regularization coefficient for the optimizer.

        - num_hidden_layers (int): (Hyperparameter) Number of hidden layers for both the encoder and decoder.
        - hidden_dim (int): (Hyperparameter) Number of unit per hidden layer.
        - dropout_rate (float): (Hyperparameter) Dropout rate.

        - alpha_weight (float): (Hyperparameter) Weight coefficient for the imputation loss (GAIN).
        - weight_decay (float): (Hyperparameter) Weight coefficient for the adversarial domain loss.
        - weight_decay (float): (Hyperparameter) Weight coefficient for the reconstruction loss.

        - miss_rate (float): (?) Rate of artificially induced missing data during training (GAIN). #todo hyperparameter ou parameter??
        - hint_rate (float): (?) Hint rate (GAIN). #todo hyperparameter ou parameter??
    """

    def __init__(self, mode: str,
                 path_dataset: str,
                 path_trained_model: str = None,
                 path_dataset_missing: str = None,
                 num_epochs: int = None, num_folds: int = None, seed: int = None, early_stop_patience: int = None,
                 batch_size: int = None, learning_rate: float = None, weight_decay: float = None,
                 num_hidden_layers: int = None, hidden_dim: int = None, dropout_rate: float = None,
                 alpha_weight: float = None, beta_weight: float = None, gamma_weight: float = None,
                 miss_rate: float = None, hint_rate: float = None):

        if mode not in ["train", "inference"]:
            raise ValueError(f"mode must be either 'train' or 'inference'. got {mode}")
        
        self.mode = mode
        self.path_dataset = path_dataset
        self.path_trained_model = path_trained_model if mode == "inference" else None
        self.path_dataset_missing = path_dataset_missing if mode == "train" else None


        if mode == "train":
            # training parameters
            self.num_epochs = num_epochs
            self.num_folds = num_folds
            self.seed = seed # random seed for reproducibility
            self.early_stop_patience = early_stop_patience

            # hyperparameters
            self.batch_size = batch_size
            self.learning_rate = learning_rate
            self.weight_decay = weight_decay

            self.num_hidden_layers = num_hidden_layers
            self.hidden_dim = hidden_dim
            self.dropout_rate = dropout_rate

            self.alpha_weight = alpha_weight
            self.beta_weight = beta_weight
            self.gamma_weight = gamma_weight

            self.miss_rate = miss_rate
            self.hint_rate = hint_rate

    @staticmethod
    def _read_json(json_path):
        try:
            with open(json_path, "r") as f:
                params = json.load(f)
        except FileNotFoundError:
            logger.error(f"JSON file not found: {json_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON: {e}")
            raise

        mode = params.get("mode", "train")

        if mode == "train":
            required_keys = [
                "mode", "path_dataset", "path_dataset_missing", 
                "num_epochs", "num_folds", "seed", 
                "early_stop_patience", "batch_size", 
                "learning_rate", "weight_decay",
                "num_hidden_layers", "hidden_dim", "dropout_rate",
                "alpha_weight", "beta_weight", "gamma_weight",
                "miss_rate", "hint_rate"
            ]
        else:
            required_keys = ["mode", "path_dataset", "path_trained_model"]

        for key in required_keys:
            if key not in params:
                raise ValueError(f"Missing required parameter: {key}")

        # Type validation
        if not isinstance(params["path_dataset"], str):
            raise TypeError("path_dataset must be a string")
        
        if mode == "inference":
            if not isinstance(params["path_trained_model"], str):
                raise TypeError("path_trained_model must be a string")
        
        if mode == "train":
            if not isinstance(params["path_dataset_missing"], str):
                raise TypeError("path_dataset_missing must be a string")
            if not isinstance(params["num_epochs"], int):
                raise TypeError("num_epochs must be an integer")
            if not isinstance(params["num_folds"], int):
                raise TypeError("num_folds must be an integer")
            if not isinstance(params["seed"], int):
                raise TypeError("seed must be an integer")
            if not isinstance(params["early_stop_patience"], int):
                raise TypeError("early_stop_patience must be an integer")
            
            if not isinstance(params["batch_size"], int):
                raise TypeError("batch_size must be an integer")
            if not isinstance(params["learning_rate"], float):
                raise TypeError("learning_rate must be a float")
            if not isinstance(params["weight_decay"], float):
                raise TypeError("weight_decay must be a float")
            
            if not isinstance(params["num_hidden_layers"], int):
                raise TypeError("num_hidden_layers must be an integer")
            if not isinstance(params["hidden_dim"], int):
                raise TypeError("hidden_dim must be an integer")
            if not isinstance(params["dropout_rate"], float):
                raise TypeError("dropout_rate must be a float")
            
            if not isinstance(params["alpha_weight"], float):
                raise TypeError("alpha_weight must be a float")
            if not isinstance(params["beta_weight"], float):
                raise TypeError("beta_weight must be a float")
            if not isinstance(params["gamma_weight"], float):
                raise TypeError("gamma_weight must be a float")

            if not isinstance(params["miss_rate"], float):
                raise TypeError("miss_rate must be a float")
            if not isinstance(params["hint_rate"], float):
                raise TypeError("hint_rate must be a float")

        return params

    @classmethod
    def read_hyperparameters(cls, params_json):

        params = cls._read_json(params_json)

        mode = params["mode"]

        if mode == "train":
            path_dataset = params["path_dataset"]
            path_dataset_missing = params["path_dataset_missing"]

            num_epochs = params["num_epochs"]
            num_folds = params["num_folds"]
            seed = params["seed"]
            early_stop_patience = params["early_stop_patience"]

            batch_size = params["batch_size"]
            learning_rate = params["learning_rate"]
            weight_decay = params["weight_decay"]

            num_hidden_layers = params["num_hidden_layers"]
            hidden_dim = params["hidden_dim"]
            dropout_rate = params["dropout_rate"]

            alpha_weight = params["alpha_weight"]
            beta_weight = params["beta_weight"]
            gamma_weight = params["gamma_weight"]

            miss_rate = params["miss_rate"]
            hint_rate = params["hint_rate"]
            
            return cls(
                mode,
                path_dataset,
                None,
                path_dataset_missing,
                num_epochs,
                num_folds,
                seed,
                early_stop_patience,
                batch_size,
                learning_rate,
                weight_decay,
                num_hidden_layers,
                hidden_dim,
                dropout_rate,
                alpha_weight,
                beta_weight,
                gamma_weight,
                miss_rate,
                hint_rate
            )
        else:
            path_dataset = params["path_dataset"]
            path_trained_model = params["path_trained_model"]

            return cls(
                mode,
                path_dataset,
                path_trained_model,
                None,
                None, None, None,
                None, None, None,
                None, None, None,
                None, None, None,
                None, None, None,
            )
    
    def __getitem__(self, name):
        return self.__getattribute__(name)

    def __setitem__(self, key, value):
        self.key = value

    def update_hypers(self, params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"{key} is not a valid parameter")

    def to_dict(self) -> dict:
        if self.mode == "inference":
            params = {
                "path_dataset": self.path_dataset,
                "path_trained_model": self.path_trained_model,
            }
        else:
            params = {
                "path_dataset": self.path_dataset,
                "path_dataset_missing": self.path_dataset_missing,
                "num_epochs": self.num_epochs,
                "num_folds": self.num_folds,
                "seed": self.seed,
                "early_stop_patience": self.early_stop_patience,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "num_hidden_layers": self.num_hidden_layers,
                "hidden_dim": self.hidden_dim,
                "dropout_rate": self.dropout_rate,
                "alpha_weight": self.alpha_weight,
                "beta_weight": self.beta_weight,
                "gamma_weight": self.gamma_weight,
                "miss_rate": self.miss_rate,
                "hint_rate": self.hint_rate
            }
        return params
    
    def to_json(self, path: str):
        with open(path, "w") as f:
            j = json.dump(
                    self,
                    f,
                    default=lambda o: o.__dict__, 
                    sort_keys=True,
                    indent=2)
        return j
    
    def __repr__(self):
        s = "=== Params Gain Dann ===\n"
        s += f"Dataset: {self.path_dataset} \n"
        s += f"Number of epochs: {self.num_epochs} \n"
        s += f"Number of folds: {self.num_folds} \n"
        s += f"Seed: {self.seed} \n"
        s += f"Early Stop Patience: {self.early_stop_patience} \n"
        s += f"Batch size: {self.batch_size} \n"
        s += f"Learning rate: {self.learning_rate} \n"
        s += f"Weight decay: {self.weight_decay} \n"
        s += f"Number of layers: {self.num_hidden_layers} \n"
        s += f"Hidden dimension (num neurons): {self.hidden_dim} \n"
        s += f"Dropout rate: {self.dropout_rate} \n"
        s += f"Alpha weight: {self.alpha_weight} \n"
        s += f"Beta weight: {self.beta_weight} \n"
        s += f"Gamma weight: {self.gamma_weight} \n"
        s += f"Miss rate: {self.miss_rate} \n"
        s += f"Hint rate: {self.hint_rate} \n"

        return s