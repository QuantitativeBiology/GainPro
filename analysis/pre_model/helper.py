import pandas as pd
from pathlib import Path

def load_tsv(
    dataset_path: Path
) -> pd.DataFrame:
    "Tailored to PRIDE datasets."
    data = pd.read_csv(
        dataset_path,
        sep="\t",
        lineterminator="\n",
        skiprows=10,
        header=0,
        usecols=["protein", "sample_accession", "ribaq"],
    )
    X = data.pivot(index="sample_accession", columns="protein", values="ribaq")
    return X

def load_condition_tsv(
    dataset_path: Path,
) -> pd.DataFrame:
    data = pd.read_csv(
        dataset_path,
        sep="\t",
        lineterminator="\n",
        skiprows=10,
        header=0,
        usecols=["sample_accession", "condition"],
    )
    return data

# def load_tsv_with_condition(
#     dataset_path: Path
# ) -> pd.DataFrame:
#     "Tailored to PRIDE datasets."
#     data = pd.read_csv(
#         dataset_path,
#         sep="\t",
#         lineterminator="\n",
#         skiprows=10,
#         header=0,
#         usecols=["protein", "sample_accession", "condition", "ribaq"],
#     )

#     sample_condition = (
#         data[["sample_accession", "condition"]]
#         .drop_duplicates()
#         .set_index("sample_accession")["condition"]
#     )

#     X = data.pivot(index="sample_accession", columns="protein", values="ribaq")
#     sample_condition = sample_condition.loc[X.index]

#     return X, sample_condition

def load_tsv_with_condition(
    dataset_path: Path
) -> pd.DataFrame:
    "Tailored to PRIDE datasets."
    data = pd.read_csv(
        dataset_path,
        sep="\t",
        lineterminator="\n",
        skiprows=10,
        header=0,
        usecols=["protein", "sample_accession", "condition", "ribaq"],
    )

    sample_condition = (
        data[["sample_accession", "condition"]]
        .drop_duplicates()
        .set_index("sample_accession")["condition"]
    )

    X = data.pivot(index="sample_accession", columns="protein", values="ribaq")
    sample_condition = sample_condition.loc[X.index]
    X["condition"] = sample_condition
    return X