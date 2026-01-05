import json
import optuna
import copy
import os
import matplotlib.pyplot as plt

# from models import plot_folder
from gainpro.train import GainDannTrain
from gainpro.params_gain_dann import ParamsGainDann
from gainpro.data_utils import Data


class OptunaOptimization:
    def __init__(self, data, hypers):
        self.data = data
        self.hypers = hypers
    
    def __call__(self, trial):
        
        hypers = self.sample_params(trial)

        train = GainDannTrain(data, hypers, early_stop_patience=10)

        loss = train.train()

        return loss


    def sample_params(self, trial):
        hypers = copy.deepcopy(self.hypers)

        hypers["hidden_dim"] = trial.suggest_categorical("hidden_dim", [256, 512, 1024])

        # hypers["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True)
        # hypers["weight_decay"] = trial.suggest_float("weight_decay", 1e-5, 0.5, log=True)

        # hypers["num_hidden_layers"] = trial.suggest_int("num_hidden_layers", 1, 3)
        # hypers["dropout_rate"] = trial.suggest_float("dropout_rate", 0.1, 0.6)

        hypers["alpha_weight"] = trial.suggest_float("alpha_weight", 1, 3)
        hypers["beta_weight"] = trial.suggest_float("beta_weight", 2, 5)
        hypers["gamma_weight"] = trial.suggest_float("gamma_weight", 0.01, 1)

        # hypers["hint_rate"] = trial.suggest_float("hint_rate", 0.05, 0.9, log=True)

        #todo talvez scheduler? activation function??

        return hypers

if __name__ == "__main__":

    hypers = ParamsGainDann.read_hyperparameters("../../configs/params_gain_dann.json")

    # Load dataset
    data = Data(dataset_path=hypers.path_dataset, miss_rate=hypers.miss_rate, start_col=8000) #todo tirar isto do start_col na versão oficial e quando corro nos servers
    input_dim = data.n_proteins

    study_name = "Hyperparameters_Optimization"

    project_dir = os.getcwd()
    results_folder = os.path.join(project_dir, "reports/files/")
    os.makedirs(results_folder, exist_ok=True) #todo ter isto talvez como uma variável global como no MOSA

    # optuna.delete_study(study_name=study_name, storage=f"sqlite:///{results_folder}/optuna_{study_name}.db")

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=f"sqlite:///{results_folder}/optuna_{study_name}.db"
    )

    study.optimize(
        OptunaOptimization(data, hypers),
        n_trials=10,
        # n_trials=2,
        show_progress_bar=True,
        n_jobs=1
    )

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(completed_trials))

    print("Best Trial:")
    best_trial = study.best_trial
    print("  Value: ", best_trial.value)
    
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"\t{key} = {value}")

    # Save best hyperparameters to a json file
    best_hypers = ParamsGainDann.read_hyperparameters("../../configs/params_gain_dann.json")
    best_hypers.update_hypers(best_trial.params)

    path = f"{results_folder}/optuna_{study_name}_best_hypers.json"
    best_hypers.to_json(path)
    # with open(f"{results_folder}/optuna_{study_name}_best_hypers.json", "w") as f:
    #     json.dump(best_hypers, f, indent=2)

    # ==== Plot optimization plots ====

    fig = optuna.visualization.plot_optimization_history(study)
    plt.savefig(
        f"{results_folder}/optuna_{study.study_name}_optimization_history.png"
    )

    fig = optuna.visualization.plot_param_importances(study)
    plt.savefig(
        f"{results_folder}/optuna_{study.study_name}_parameters_importance_plot.png"
    )

    fig = optuna.visualization.plot_slice(study)
    plt.savefig(
        f"{results_folder}/optuna_{study.study_name}_slice_plot.png"
    )

    fig = optuna.visualization.plot_edf(study)
    plt.savefig(
        f"{results_folder}/optuna_{study.study_name}_edf_plot.png"
    )

    fig = optuna.visualization.plot_parallel_coordinate(study)
    plt.savefig(
        f"{results_folder}/optuna_{study.study_name}_parallel_coordinate_plot.png"
    )

    # fig = optuna.visualization.plot_contour(
    #     study, params=["learning_rate"]
    # )
    # plt.savefig(
    #     f"{results_folder}/optuna_{study.study_name}_contour_plot.png"
    # )
    
