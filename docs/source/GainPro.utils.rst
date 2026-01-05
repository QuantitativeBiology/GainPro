Utils Class
========================

.. automodule:: gainpro.utils
   :members:
   :undoc-members:
   :show-inheritance:

Here you will find the class `Utils` used during the process of data preprocessing and evaluation of the imputation quality.
The `Utils` class is responsible for handling various utility functions such as data normalization, data scaling, data splitting, 
and evaluation metrics calculation.

Functions
----------


- create_csv(data, name: str, header)

    Creates a CSV file from the given dataset.

    **Steps:**

    1. Converts data into a Pandas DataFrame.
    2. Saves the DataFrame as a CSV file with the given name.
    3. Uses the provided header for column naming.

- create_dist(size: int, dim: int, name: str)

    Generates a synthetic dataset based on a normal distribution.

    **Steps:**

    1. Creates a size x dim matrix of normally distributed values.
    2. Applies a linear transformation (A matrix) and shift (b vector).
    3. Saves the transformed dataset as a CSV file.

- create_missing(data, miss_rate: float, name: str, header)

    Generates a dataset with missing values by randomly removing observations.

    **Steps:**

    1. Initializes a zero matrix mask of the same shape as data.
    2. Iterates over each feature, generating a probability for missing values.

    .. code-block:: python

        chance = torch.rand(size)

        miss = chance > miss_rate

    3. Masks values according to miss_rate, replacing them with NaN.

    .. code-block:: python

        mask[:, i] = miss

        missing_data = np.where(mask < 1, np.nan, data)

    4. Saves the new dataset with missing values as a CSV file.

- create_output(data, path: str, override: int)

    Handles the output file generation, either overriding or appending new data.

    **Steps:**

    1. If override is 1, saves data as a new CSV file.
    2. Otherwise, if the file exists, reads it and appends the new data columns.
    3. Concatenates the updated DataFrame and saves it.

- output()

    Stores multiple outputs related to training, including metrics and system resource usage.

    **Steps:**

    1. Calls create_csv to save imputed data.
    2. Calls create_output for:
        - Discriminator loss (lossD.csv).
        - Generator loss (lossG.csv).
        - Training and test loss (lossMSE_train.csv, lossMSE_test.csv).
        - System performance logs (cpu.csv, ram.csv, ram_percentage.csv).

- sample_idx(m, n)

    Generates a random sample of n indices from m elements.

    **Steps:**

    1. Creates a random permutation of integers from 0 to m-1.
    2. Selects the first n elements as the sampled indices.
    3. Returns the selected indices.

- build_protein_matrix(tsv_file)

    Processes a TSV file containing proteomics data and restructures it into a matrix.

    **Steps:**

    1. Reads the TSV file, skipping initial metadata rows.
    2. Extracts relevant columns (protein, sample_accession, ribaq).
    3. Converts data into a pivoted matrix (proteins as index, samples as columns).
    4. Returns the formatted matrix.

- handle_parquet(parquet_file)

    Reads a Parquet file.

    **Steps**

    1. Use polars to read the Parquet file.
    2. Process the data in the file 
    3. Return the resulting DataFrame. 


