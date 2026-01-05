from gainpro.hypers import Params
import numpy as np
import pandas as pd
import os


class Metrics:
    def __init__(self, hypers: Params):

        self.hypers = hypers

        self.loss_D = np.zeros(hypers.num_iterations)
        self.loss_D_evaluate = np.zeros(hypers.num_iterations)

        self.loss_G = np.zeros(hypers.num_iterations)
        self.loss_G_evaluate = np.zeros(hypers.num_iterations)

        self.loss_MSE_train = np.zeros(hypers.num_iterations)
        self.loss_MSE_train_evaluate = np.zeros(hypers.num_iterations)

        self.loss_MSE_test = np.zeros(hypers.num_iterations)

        self.cpu = np.zeros(hypers.num_iterations)
        self.cpu_evaluate = np.zeros(hypers.num_iterations)

        self.ram = np.zeros(hypers.num_iterations)
        self.ram_evaluate = np.zeros(hypers.num_iterations)

        self.ram_percentage = np.zeros(hypers.num_iterations)
        self.ram_percentage_evaluate = np.zeros(hypers.num_iterations)

        self.data_imputed = None
        self.ref_data_imputed = None

    def create_output(self, data, name: str):
        hypers = self.hypers  # Accessing metrics.hypers

        if hypers.override == 1:
            df = pd.DataFrame(data)
            df.to_csv(f"{hypers.output_folder}{name}", index=False)

        else:
            if os.path.exists(f"{hypers.output_folder}{name}"):
                df = pd.read_csv(f"{hypers.output_folder}{name}")
                new_df = pd.DataFrame(data)
                df = pd.concat([df, new_df], axis=1)
                df.columns = range(len(df.columns))
                df.to_csv(f"{hypers.output_folder}{name}", index=False)
            else:
                df = pd.DataFrame(data)
                df.to_csv(f"{hypers.output_folder}{name}", index=False)
