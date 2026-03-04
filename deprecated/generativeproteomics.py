from hypers import Params
from model import Network
from dataset import Data
from output import Metrics
from imputation_manager import ImputationManager
import utils

import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import optuna

import time
import cProfile
import pstats
import argparse
import os
import psutil


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="path to missing data")
    parser.add_argument("-o", default="imputed", help="name of output file")
    parser.add_argument("--ref", help="path to a reference (complete) dataset")
    parser.add_argument(
        "--ofolder", default=os.getcwd() + "/results/", help="path to output folder"
    )
    parser.add_argument("--it", type=int, default=2001, help="number of iterations")
    parser.add_argument("--batchsize", type=int, default=128, help="batch size")
    parser.add_argument("--alpha", type=float, default=10, help="alpha")
    parser.add_argument("--miss", type=float, default=0.1, help="missing rate")
    parser.add_argument("--hint", type=float, default=0.9, help="hint rate")
    parser.add_argument(
        "--lrd", type=float, default=0.001, help="learning rate for the discriminator"
    )
    parser.add_argument(
        "--lrg", type=float, default=0.001, help="learning rate for the generator"
    )
    parser.add_argument("--parameters", help="load a parameters.json file")
    parser.add_argument(
        "--override", type=int, default=0, help="override previous files"
    )
    parser.add_argument("--outall", type=int, default=0, help="output all files")
    parser.add_argument("--model", type = str, help = "indicates which model from HuggingFace to use")
    return parser.parse_args()


if __name__ == "__main__":
    start_time = time.time()
    with cProfile.Profile() as profile:

        folder = os.getcwd()

        args = init_arg()

        missing_file = args.i
        output_file = args.o
        ref_file = args.ref
        output_folder = args.ofolder
        num_iterations = args.it
        batch_size = args.batchsize
        alpha = args.alpha
        miss_rate = args.miss
        hint_rate = args.hint
        lr_D = args.lrd
        lr_G = args.lrg
        parameters_file = args.parameters
        override = args.override
        output_all = args.outall
        model = args.model


        if parameters_file is not None:
            params = Params.read_hyperparameters(parameters_file)
            missing_file = params.input
            output_file = params.output
            ref_file = params.ref
            output_folder = params.output_folder
            num_iterations = params.num_iterations
            batch_size = params.batch_size
            alpha = params.alpha
            miss_rate = params.miss_rate
            hint_rate = params.hint_rate
            lr_D = params.lr_D
            lr_G = params.lr_G
            override = params.override
            output_all = params.output_all

        else:
            params = Params(
                missing_file,
                output_file,
                ref_file,
                output_folder,
                None,
                num_iterations,
                batch_size,
                alpha,
                miss_rate,
                hint_rate,
                lr_D,
                lr_G,
                override,
                output_all,
            )

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        if missing_file is None:
            print("Input file not provided")
            exit(1)
        if missing_file.endswith(".csv"):
            df_missing = pd.read_csv(missing_file)
            missing = df_missing.values
            missing_header = df_missing.columns.tolist()
            params.update_hypers(header=missing_header)
        elif missing_file.endswith(".tsv"):
            df_missing = utils.build_protein_matrix(missing_file)
            missing = df_missing.values
            missing_header = df_missing.columns.tolist()
            params.update_hypers(header=missing_header)
        elif missing_file.endswith(".parquet"):
            df_missing = utils.handle_parquet(missing_file)
            missing = df_missing.to_numpy()
            missing_header = df_missing.columns
            params.update_hypers(header=missing_header)
        else:
            print("Invalid file format")
            exit(2)

        if args.model is not None:
            print(args.model)
            imputation_manager = ImputationManager(args.model, df_missing, missing_file)
            imputation_manager.run_model(args.model)
            exit(0)

        else:
        
            dim = missing.shape[1]
            train_size = missing.shape[0]

            h_dim1 = dim
            h_dim2 = dim

            net_G = nn.Sequential(
                nn.Linear(dim * 2, h_dim1),
                nn.ReLU(),
                nn.Linear(h_dim1, h_dim2),
                nn.ReLU(),
                nn.Linear(h_dim2, dim),
                nn.Sigmoid(),
            )

            net_D = nn.Sequential(
                nn.Linear(dim * 2, h_dim1),
                nn.ReLU(),
                nn.Linear(h_dim1, h_dim2),
                nn.ReLU(),
                nn.Linear(h_dim2, dim),
                nn.Sigmoid(),
            )

            metrics = Metrics(params)
            model = Network(hypers=params, net_G=net_G, net_D=net_D, metrics=metrics)

            if ref_file is not None:
                df_ref = pd.read_csv(ref_file)
                ref = df_ref.values
                ref_header = df_ref.columns.tolist()

                if dim != ref.shape[1]:
                    print(
                        "\n\nThe reference and data files provided don't have the same number of features\n"
                    )
                    exit(3.1)
                elif train_size != ref.shape[0]:
                    print(
                        "\n\nThe reference and data files provided don't have the same number of samples\n"
                    )
                    exit(3.2)

                data = Data(missing, miss_rate, hint_rate, ref)
                model.train_ref(data, missing_header)

            else:
                data = Data(missing, miss_rate, hint_rate)
                model.evaluate(data, missing_header)
                model.train(data, missing_header)

            run_time = []
            run_time.append(time.time() - start_time)
            file_path = output_folder + "run_time.csv"

            if override == 1:
                df_run_time = pd.DataFrame(run_time)
                df_run_time.to_csv(file_path, index=False)

            else:
                if os.path.exists(file_path):
                    with open(file_path, "a") as myfile:
                        myfile.write(str(run_time[0]) + "\n")

                else:
                    df_run_time = pd.DataFrame(run_time)
                    df_run_time.to_csv(file_path, index=False)

            print("\n--- %s seconds ---\n\n" % (run_time[0]))
            results = pstats.Stats(profile)
            results.sort_stats(pstats.SortKey.TIME)
            # results.print_stats()
            results.dump_stats("results.prof")
