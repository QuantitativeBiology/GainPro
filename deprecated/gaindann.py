import json
import torch
import pandas as pd
import numpy as np
from datetime import datetime

import seaborn as sns
import matplotlib.pyplot as plt
import umap.umap_ as umap

# model
from gaindannmodel import GAIN_DANN
from hypers import Params
from output import Metrics
from paramsgaindann import ParamsGainDann
from data_utils import Data
from train import GainDannTrain
from dann_utils import inverse_transform_output

# post analysis
# from umap_analysis import umap_analysis
# from pca_analysis import pca_analysis
from correlation import correlation_measured_predicted

import logging
import argparse
import inquirer
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def list_checkpoints():
    checkpoint_root="checkpoints"
    return sorted([d for d in os.listdir(checkpoint_root)
                   if os.path.isdir(os.path.join(checkpoint_root, d))])

def select_checkpoint_interactively():
    checkpoints = list_checkpoints()
    if not checkpoints:
        raise FileNotFoundError("No checkpoints found.")
    
    question = [
        inquirer.List("timestamp",
                      message="Select a checkpoint to load",
                      choices=checkpoints)
    ]
    answer = inquirer.prompt(question)
    return answer["timestamp"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--impute", action="store_true", help="Impute the dataset")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--save", action="store_true", help="Save trained model")
    parser.add_argument("--umap", action="store_true", help="UMAP analysis with the trained model")
    parser.add_argument("--pca", action="store_true", help="PCA analysis with the trained model")
    parser.add_argument("--corr", action="store_true", help="Correlation analysis with the trained model")
    parser.add_argument("--latent", action="store_true") #todo delete depois
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    params_gain_dann = ParamsGainDann.read_hyperparameters("../configs/params_gain_dann.json")
    params = params_gain_dann.to_dict()
    logger.debug(params_gain_dann)

    if params_gain_dann["path_trained_model"] is not None:
        checkpoint_dir = params_gain_dann["path_trained_model"]
        logger.info(f"Loading model from {checkpoint_dir}")
        try:
            json_path = f"{checkpoint_dir}/metadata.json"
            with open(json_path, "r") as f:
                metadata = json.load(f)
        except FileNotFoundError:
            logger.error(f"JSON file not found: {json_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON: {e}")
            raise
    else: #todo não sei se preciso realmente deste else
        timestamp = datetime.now().strftime('%d-%m_%H:%M')
        save_dir = f"../../imgs/train/{timestamp}"


    if args.train:
        # === Read dataset ===
        # ⚠️ todo delete o start_col na versão oficial
        if params_gain_dann.path_dataset_missing != "":
            print(f"Missing pd {params_gain_dann.path_dataset_missing}")
            dataset_missing = pd.read_csv(params_gain_dann.path_dataset_missing, index_col=0) # dataset with induced missingness for benchmarking
            dataset_missing = dataset_missing.iloc[:, 8500:]
            data = Data(dataset_path=params_gain_dann.path_dataset, dataset_missing=dataset_missing, start_col=8500)
        else:
            data = Data(dataset_path=params_gain_dann.path_dataset, miss_rate=params["miss_rate"], start_col=8500)
        protein_names = data.protein_names
        input_dim = data.n_proteins
        gain_params = Params()
        gain_metrics = Metrics(gain_params)
        logger.debug(data)

        # === Train Model ===
        print("Early stop patience: ", params_gain_dann["early_stop_patience"])
        train = GainDannTrain(data, params, early_stop_patience=params_gain_dann["early_stop_patience"], save_model=args.save)
        train.train()
        #todo test if the output of the model returns the missing values imputed and the other values as in the original

    if args.latent:

        logger.info("\nLoading model...\n")
        gain_params = Params()
        gain_metrics = Metrics(gain_params)

        dann_params = {"hidden_dim": metadata["params"]["hidden_dim"], 
                       "dropout_rate": metadata["params"]["dropout_rate"]}

        # load model
        model = GAIN_DANN(metadata["protein_names"], metadata["input_dim"], latent_dim=metadata["latent_dim"], n_class=metadata["n_class"], num_hidden_layers=metadata["params"]["num_hidden_layers"], 
                        dann_params=dann_params, gain_params=gain_params, gain_metrics=gain_metrics)
        model_path = f"{checkpoint_dir}/model.pt"
        if not os.path.isfile(model_path):
            logger.error(f"Model in {model_path} not found.")
            raise FileNotFoundError
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        logger.info("\nModel loaded.\n")

        # read dataset
        df = pd.read_csv(params_gain_dann["path_dataset"], index_col=0)
        print("df", df.shape)
        df = df.iloc[:, 8000:]
        projects = df["Domain"]

        # reduce to only common proteins
        common_proteins = set(model.protein_names) & set(df.columns)
        df = df.loc[:, list(common_proteins)]
        df["Domain"] = projects
        print("df: ", df.shape) # 4820 x 2013
        data = Data(df, miss_rate=0)
        print("data: ", data.dataset_normalized.shape) # 64 x 1568

        # pad with nans
        padded_data = data.dataset_normalized.reindex(columns=model.protein_names) # trick `reindex`
        print("padded: ", padded_data.shape) # 64 x 2013

        # transform to tensor
        model_input = torch.tensor(padded_data.values, dtype=torch.float32)
        print("model input: ", model_input.shape)

        # impute do modelo até o encoder
        logger.info("Encoder...")
        model.encoder.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        x = torch.tensor(padded_data.values, dtype=torch.float32)

        x = x.to(device)
        x_filled = x.clone()
        x_filled[torch.isnan(x_filled)] = 0

        with torch.no_grad():
            x_encoded = model.encoder(x_filled)
            print("X encoded", x_encoded)
        
        seed = 42

        projects = data.domain_labels
        print(projects)

        all_projects = projects.unique()
        palette = sns.color_palette("colorblind", n_colors=len(all_projects))
        project_colors = dict(zip(all_projects, palette))

        reducer = umap.UMAP(random_state=seed)
        # embedding = reducer.fit_transform(hela)
        embedding = reducer.fit_transform(x_encoded)
        print(f"UMAP embedding shape: {embedding.shape}")

        plt.figure(figsize=(11,8))
        # sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=hela_projects["Domain"], palette=project_colors, alpha=0.7, s=70)
        sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=projects, palette=project_colors, alpha=0.7, s=70)
        plt.title("UMAP projection of the latent space", fontsize=12, loc="left")
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.legend(bbox_to_anchor=(1, 1), title="Domain", frameon=False, title_fontsize="large")
        plt.tight_layout()
        plt.savefig("../../imgs/umap-latent-space.png")
        plt.show()






    if args.impute:
        logger.info("\nLoading model...\n")
        gain_params = Params()
        gain_metrics = Metrics(gain_params)

        dann_params = {"hidden_dim": metadata["params"]["hidden_dim"], 
                       "dropout_rate": metadata["params"]["dropout_rate"]}

        # load model
        model = GAIN_DANN(metadata["protein_names"], metadata["input_dim"], latent_dim=metadata["latent_dim"], n_class=metadata["n_class"], num_hidden_layers=metadata["params"]["num_hidden_layers"], 
                        dann_params=dann_params, gain_params=gain_params, gain_metrics=gain_metrics)
        model_path = f"{checkpoint_dir}/model.pt"
        if not os.path.isfile(model_path):
            logger.error(f"Model in {model_path} not found.")
            raise FileNotFoundError
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        logger.info("\nModel loaded.\n")

        # dataset_missing = torch.tensor(data.dataset_missing.values, dtype=torch.float32)
        # dataset_imputed, _ = model(dataset_missing)
        # dataset_hat = inverse_transform_output(dataset_imputed, data.scaler)

        # if args.umap:
        #     # perform umap analysis
        #     umap_analysis(dataset_missing, dataset_hat, data.domain_labels, params["seed"], checkpoint_dir)

        # if args.pca:
        #     # perform pca analysis
        #     n_components = 2
        #     pca_analysis(dataset_hat, n_components, params["seed"], checkpoint_dir)

        # if args.corr:
        #     correlation_measured_predicted(dataset_missing, dataset_hat, data.samples_names, data.sample_to_project, checkpoint_dir)

        # imputation
        #todo verificar a ordem das colunas, isso vai ser importante para o nosso modelo!

        # read dataset
        df = pd.read_csv(params_gain_dann["path_dataset"], index_col=0)

        # reduce to only common proteins
        common_proteins = set(model.protein_names) & set(df.columns)
        df = df.loc[:, list(common_proteins)]
        print("df: ", df.shape) # 64 x 1569

        # print(common_proteins)
        # print(len(common_proteins))

        # create data (that will be scaled)
        # mask alguns elementos
        data = Data(df, miss_rate=0.1, start_col=0)
        print("data: ", data.dataset_normalized.shape) # 64 x 1568

        # use normalized dataset
        # usar dataset with induced missingness
        incomplete_data = data.dataset_missing
        # incomplete_data = data.dataset_normalized

        # pad with nans
        padded_data = incomplete_data.reindex(columns=model.protein_names) # trick `reindex`
        print("padded: ", padded_data.shape) # 64 x 2013

        # transform to tensor
        model_input = torch.tensor(padded_data.values, dtype=torch.float32)
        print("model input: ", model_input.shape)

        # impute to model
        imputed_data, _ = model(model_input)

        # retrieve only original proteins
        imputed_data = pd.DataFrame(imputed_data, index=incomplete_data.index, columns=model.protein_names)
        imputed_data = imputed_data.loc[:, list(common_proteins)]
        print("imputed data: ", imputed_data.shape) # 64 x 1568

        # comparar imputation com o dataset without induced missingness
        ground_truth = torch.tensor(data.dataset_normalized.values, dtype=torch.float32)
        mask = (~torch.isnan(ground_truth)).float()
        ground_truth[torch.isnan(ground_truth)] = 0
        imputed_data_tensor = torch.tensor(imputed_data.values, dtype=torch.float32)
        squared_error = (ground_truth - imputed_data_tensor) ** 2
        rmse = torch.sqrt((squared_error * mask).sum() / mask.sum()) # RMSE error
        print(f"rmse: {rmse}")

        # scale back to real space
        imputed_data_real_values = data.scaler.inverse_transform(imputed_data.values) # in real space
        imputed_data_real = pd.DataFrame(imputed_data_real_values, index=imputed_data.index, columns=imputed_data.columns)

        print("imputed data real df: ", imputed_data_real.shape) # 64 x 1568
        print(imputed_data_real)