Params class
================

.. automodule:: gainpro.hypers
   :members:
   :undoc-members:
   :show-inheritance:

Here you will find the class `Params` used during the process of imputation of missing values.
The `Params` class is responsible for handling hyperparameters used in the model training.
It stores all hyperparameters required for model training, loads hyperparameters from a JSON file updates hyperparameters dynamically

Attributes
-----------
- `input`: The dataset with the missing values to be imputed.
- `output`: The name of the file where the imputed dataset will be saved.
- `ref` : Indicates if a reference dataset is provided, and if it is, the datasetto be used as a reference.
- `output_folder` : The name of the folder where the output file will be saved.
- `num_iterations` : The number of iterations performed to train the model.
- `batch_size` : The number of samples used in each iteration.
- `alpha` : Hyperparameter used in the weighted sum of the loss of the generator.
- `miss_rate` : Percentage of missing values in the dataset.
- `hint_rate` : Percentage of mask information retained to guide imputation.
- `lr_D` : Learning rate for the discriminator.
- `lr_G` : Learning rate for the generator.
- `override` : Indicates if the output file should be overwritten if it already exists (1 to override, 0 otherwise).
- `output_all` : Indicates if the output file should contain all the data or only the imputed values (1 to output all, 0 otherwise).

Methods
---------

- __init__()

    Initializes the `Params` class by processing the hyperparameters used in the model training.

- read_json()

    Reads a JSON file containing the hyperparameters used in the model training.

- read_hyperparameters()

    Reads hyperparameters from a JSON file and returns an instance of Params.

- update_hypers()

    Dynamically updates hyperparameters based on provided keyword arguments.

