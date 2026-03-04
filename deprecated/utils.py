from sklearn.preprocessing import LabelEncoder
from huggingface_hub import hf_hub_download
import torch
import numpy as np
import pandas as pd
import os
import anndata as ad
import polars as pl
import json

def create_csv(data, name: str, header):
    df = pd.DataFrame(data)
    df.to_csv(name + ".csv", index=False, header=header)


def create_dist(size: int, dim: int, name: str):

    X = torch.normal(0.0, 1, (size, dim))
    A = torch.tensor([[1, 2], [-0.1, 0.5]])
    b = torch.tensor([0, 0])
    data = torch.matmul(X, A) + b

    create_csv(data, name)


def create_missing(data, miss_rate: float, name: str, header):

    size = data.shape[0]
    dim = data.shape[1]

    mask = torch.zeros(data.shape)

    for i in range(dim):

        chance = torch.rand(size)
        miss = chance > miss_rate
        mask[:, i] = miss

        missing_data = np.where(mask < 1, np.nan, data)

    name = name + "_{}".format(int(miss_rate * 100))

    create_csv(missing_data, name, header)


def create_output(data, path: str, override: int):

    if override == 1:
        df = pd.DataFrame(data)
        df.to_csv(path, index=False)

    else:
        if os.path.exists(path):
            df = pd.read_csv(path)
            new_df = pd.DataFrame(data)
            df = pd.concat([df, new_df], axis=1)
            df.columns = range(len(df.columns))
            df.to_csv(path, index=False)
        else:
            df = pd.DataFrame(data)
            df.to_csv(path, index=False)


def output(
    data_train_imputed,
    output_folder,
    output_file,
    missing_header,
    loss_D_values,
    loss_G_values,
    loss_MSE_train,
    loss_MSE_test,
    cpu,
    ram,
    ram_percentage,
    override,
):

    create_csv(
        data_train_imputed,
        output_folder + output_file,
        missing_header,
    )
    create_output(
        loss_D_values,
        output_folder + "lossD.csv",
        override,
    )
    create_output(
        loss_G_values,
        output_folder + "lossG.csv",
        override,
    )
    create_output(
        loss_MSE_train,
        output_folder + "lossMSE_train.csv",
        override,
    )

    create_output(
        loss_MSE_test,
        output_folder + "lossMSE_test.csv",
        override,
    )

    create_output(
        cpu,
        output_folder + "cpu.csv",
        override,
    )

    create_output(ram, output_folder + "ram.csv", override)

    create_output(
        ram_percentage,
        output_folder + "ram_percentage.csv",
        override,
    )


def sample_idx(m, n):
    A = np.random.permutation(m)
    idx = A[:n]
    return idx


def build_protein_matrix(tsv_file):

    data = pd.read_csv(
        tsv_file,
        sep="\t",
        lineterminator="\n",
        skiprows=(10),
        header=(0),
        usecols=(0, 1, 4),
    )

    # matrix = data.pivot(index="sample_accession", columns="protein", values="ribaq")
    matrix = data.pivot(index="protein", columns="sample_accession", values="ribaq")

    # print("Number of samples", matrix.shape[0])
    # print("Number of features", matrix.shape[1])
    # print("Missing Rate (%)", matrix.isna().sum().sum() / matrix.size * 100)

    return matrix


def build_protein_matrix_from_anndata(anndata_file: str)  -> pd.DataFrame:
    """
        An anndata file has the .h5ad extension.
    """

    adata = ad.read_h5ad(anndata_file)

    data = pd.DataFrame(
        adata.X,
        index=adata.obs["SampleID"],
        columns=adata.var["ProteinName"]
    )

    data = data.T

    return data

def handle_parquet(parquet_file):
    """ Function to read a parquet file and return a DataFrame with only numeric columns. """

    df = pl.read_parquet(parquet_file)
    df_numeric = df.select(pl.col(pl.Float64, pl.Int64, pl.Int32, pl.Float32))

    return df_numeric

