Data Class
========================

.. automodule:: gainpro.dataset
   :members:
   :undoc-members:
   :show-inheritance:

Here you will find the class `Data` and other functions used by it during the process of imputation of missing values.
The `Data` class is responsible for handling datasets with missing values.
It preprocesses the dataset, generates necessary masks, and scales the data for model training.

Attributes
-----------
- `dataset`: Original dataset with missing values to be imputed.
- `mask`: Binary mask indicating observed (`1`) and missing (`0`) values.
- `dataset_scaled`: Scaled version of the dataset, after going throw Min-Max scaling.
- `hint`: Hint matrix used for guiding imputation.
- `ref_dataset`: Reference dataset used to assess the quality of imputation (if provided).
- `ref_mask`: Binary mask for the reference dataset.
- `miss_rate`: Percentage of missing values in the dataset.
- `hint_rate`: Percentage of mask information retained to guide imputation


Methods
---------

- __init__()

   Initializes the `Data` class by processing the dataset and handling missing values.

   **Steps:**

   1. Converts `NaN` values to `0.0` and generates a binary `mask`.
   2. Scales the dataset using MinMax scaling (`dataset_scaled`).
   3. Generates a `hint` matrix to guide imputation.
   4. If a reference dataset is provided, it processes it.
   5. If no reference dataset is provided, `_create_ref()` generates one.


- _create_ref()

   Generates a reference dataset by randomly concealing observed values.

   **Steps:**

   1. Clones the original dataset and mask (`ref_dataset` and `ref_mask`).
   2. Randomly selects observed values (`mask == 1`) to remove based on `miss_rate`.
   3. Sets these values to `0.0` to simulate missing data.
   4. Generates a `ref_hint` matrix to guide imputation.
   5. Scales the reference dataset using `MinMaxScaler`.


- generate_hint(mask, hint_rate):

   Generates a hint matrix to assist in imputation.

   Uses `generate_mask()` to create hints.

- generate_mask(data, miss_rate):

   Randomly masks data based on the specified missing rate.

   Outputs a binary mask (`1 = observed`, `0 = missing`).
