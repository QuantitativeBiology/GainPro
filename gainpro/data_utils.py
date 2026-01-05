import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# def generate_missingness(data: pd.DataFrame, miss_rate: float = 0.1) -> pd.DataFrame:
#     """
#     Randomly introduce missingness into the dataset, except for the last column that corresponds
#     to the Domain.

#     Args:
#         - data (pd.DataFrame): Dataset to be induced missingness.
#         - miss_rate (float): Generate more `miss_rate` missing values. Default is 0.1.

#     Returns:
#         pd.DataFrame: With the additional missing rate.
#     """
#     logger.info(f"Injecting missingness with rate={miss_rate}...")
#     size, dim = data.shape

#     mask = np.random.rand(size, dim) > miss_rate
#     data_np = data.to_numpy()

#     # Apply mask
#     missing_data_np = np.where(mask, data_np, np.nan)

#     # Reconstruct DataFrame
#     missing_data = pd.DataFrame(missing_data_np, columns=data.columns, index=data.index)

#     # Logging actual missingness
#     total_missing = missing_data.isna().sum().sum()
#     actual_missing_rate = total_missing / missing_data.size
#     logger.info(f"Actual missingness: {actual_missing_rate:.4%} \n")

#     return missing_data, actual_missing_rate


def generate_additional_missingness(data: pd.DataFrame, miss_rate: float = 0.1) -> tuple[pd.DataFrame, float]:
    """
    Randomly introduce additional missingness into the dataset, except for the last column.

    Args:
        - data (pd.DataFrame): Dataset where we want to add more missingness.
        - add_miss_rate (float): Additional fraction of total entries that should be missing.

    Returns:
        (pd.DataFrame, float): DataFrame with added missing values, and the new missing rate.
    """
    logger.info(f"Adding additional missingness of {miss_rate:.2%}...")

    np.random.seed(42)

    data_missing = data.copy()
    size = data_missing.size 

    # Count existing missing values
    total_missing = data_missing.isna().sum().sum()
    current_rate = total_missing / size

    print("Current missingness", current_rate)

    # Target missingness
    target_rate = min(current_rate + miss_rate, 1.0)
    target_missing = int(target_rate * size)

    # How many more values to mask
    additional_to_mask = target_missing - total_missing
    if additional_to_mask <= 0:
        print("⚠️ No additional missingness needed (already too many NaNs).")
        return data_missing, current_rate

    # Identify all non-missing (eligible) positions, excluding last column since we have the domain column
    eligible_positions = np.argwhere(~data_missing.iloc[:, :-1].isna().values)

    # Randomly sample positions to mask
    to_mask_idx = np.random.choice(len(eligible_positions), size=additional_to_mask, replace=False)
    to_mask = eligible_positions[to_mask_idx]

    # Apply masking
    for i, j in to_mask:
        data_missing.iat[i, j] = np.nan

    new_missing = data_missing.isna().sum().sum()
    new_rate = new_missing / size
    print(f"New missingness: {new_rate:.2%} (target was {target_rate:.2%})\n")

    return data_missing, new_rate


def load_dataset(dataset_path: str, start_col: int = 8000) -> pd.DataFrame:
    """
    Load the dataset from CSV, select relevant columns, and optionally replace zeros with NaNs.

    Args:
        path (str): Path to the dataset CSV.
        #todo afterwards delete start_col

    Returns:
        pd.DataFrame: Dataset.
    """
    logger.info(f"\nLoading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path, index_col=0)  # samples as rows (obs.) and proteins as columns
    
    logger.info(f" Original shape: {df.shape}")
    df = df.iloc[:, start_col:]
    logger.info(f" Sliced shape (from column {start_col}): {df.shape} \n")

    return df


class Data():
    def __init__(self, dataset: pd.DataFrame=None, dataset_path: str=None, 
                 dataset_missing: pd.DataFrame=None, # disclaimer: this dataset missing is helpful to guarantee comparison on the same entries for benchmarking
                 miss_rate: float=0.1, start_col: int=0):
        """
            Data can either receive a the data already on a dataframe through the dataset parameter or can load a dataset from the ´dataset_path´.

            Args:
                - dataset (pd.DataFrame)
        """
        #todo eliminar depois o start_col, apenas para correr localmente
        self.dataset_path = dataset_path

        if dataset_path is not None:
            self.dataset = load_dataset(self.dataset_path, start_col)
        else:
            self.dataset = dataset

        self.shape = self.dataset.shape
        self.n_samples = self.shape[0]

        if "Domain" in self.dataset.columns: # training purposes
            self.n_proteins = self.shape[1] - 1
            self.domain_labels = self._get_domain_labels()
            self.n_projects = self._get_number_projects()
            self.sample_to_project = self._get_sample_to_project() # mapping
            self.drop_domain_labels()
        else: # imputation purposes
            self.n_proteins = self.shape[1]
        
        self.samples_names = self._get_name_samples()
        self.protein_names = list(self.dataset.columns)

        self.scaler = StandardScaler()
        self.dataset_normalized = self.normalize_df(self.dataset)

        self.miss_rate = miss_rate

        if dataset_missing is not None:
            to_mask = dataset_missing.iloc[:, :-1].isna()
            self.dataset_missing = self.dataset_normalized.copy(deep=True)
            self.dataset_missing[to_mask] = np.nan

            missingness = self.dataset_missing.isna().sum().sum()
            rate = missingness / (self.dataset_normalized.size)
            print(f"Missingness rate: {rate:.2%}\n")

            print("Entries to mask: ")
            print(to_mask)

            print("Original matrix:")
            print(self.dataset_normalized)

            print("Masked matrix: ")
            print(self.dataset_missing)
        else:
            if miss_rate != 0:
                self.dataset_missing, self.dataset_missing_miss_rate = generate_additional_missingness(self.dataset_normalized, miss_rate=self.miss_rate)
                self.dataset_missing = self.dataset_missing.astype(np.float32)
            else: # imputation case
                self.dataset_missing = self.dataset_normalized.astype(np.float32)

    def _get_domain_labels(self):# -> Series:
        return self.dataset["Domain"]

    def _get_number_projects(self) -> int:
        return len(np.unique(self.domain_labels))

    def _get_name_samples(self):
        return self.dataset.index.tolist()
    
    def _get_sample_to_project(self):
        return dict(zip(self.dataset.index, self.dataset['Domain']))

    def drop_domain_labels(self) -> None:
        self.dataset = self.dataset.drop(columns="Domain")

    def normalize_df(self, dataset) -> pd.DataFrame:
        """
            Args:
                - dataset (pd.DataFrame): the dataset to normalize

            Returns:
                - pd.DataFrame: the normalized dataset
        """
        x = self.scaler.fit_transform(dataset.values)
        dataset_scaled = pd.DataFrame(x, index=dataset.index, columns=dataset.columns)

        print("Is the mean of the data 0?")
        print(dataset_scaled.mean())

        return dataset_scaled

    def __repr__(self) -> str:
        s = "\n === Dataset information ===\n"
        s+= f"Num samples: {self.n_samples} \n"
        s+= f"Num proteins: {self.n_proteins} \n"
        s+= f"Num of projects: {self.n_projects} \n"
        # s+= f"Original miss rate: {self.dataset_missing_miss_rate - self.miss_rate:.3f} \n"
        # s+= f"Induced miss rate: {self.dataset_missing_miss_rate:.3f} \n"

        return s

