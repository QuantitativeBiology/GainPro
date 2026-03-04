Imputation Manager Class
========================

.. automodule:: GainPro.manager
   :members:
   :undoc-members:
   :show-inheritance:

Here you will find the class `Imputation Manager` and other functions used by it 
during the process of managing the selection of an imputation method besides GenerativeProteomics.
This class works as a wrapper, allowing the user to easily add and use different imputation methods.

Attributes
-----------
- `model`: The name of the model to be used for imputation.
- `df`: The dataset containing missing values to be imputed.
- `missing`: The path to the file containing missing values.
- `dict_imputation_methods`: A dictionary to store different imputation methods and their associated functions.

Methods
--------
- __init__(model, df, missing): 

   Initializes the `Imputation_Manager` class by setting the model, dataset, 
   and missing values file.

- add_method(self, model, fn):

   Adds a new imputation method to the manager. 
   The method is associated with a function that performs the imputation.

   **Steps:**

    1. Takes the model name and the function as input.
    2. Checks if the model already exists in the dictionary.
    3. If not, adds the model and function to the dictionary.

- run_model(self, model):

    Checks if the model exists, redirecting to the appropriate imputation function.
    If the model does not exist, it raises a ValueError.

    **Steps:**

    1. Takes the model name as input.   
    2. Checks if the model exists in the dictionary.
    3. If it exists, calls the associated function to perform imputation.
    4. If it does not exist, raises a ValueError indicating the model is not found.

- hugging_face_gain_dann(dataset)

    Downloads and uses a pre-trained model from Hugging Face to perform data imputation.

    **Steps:**

    1. Builds the url of the model repository.
    2. Loads the pre-trained model from the Hugging Face repository.
    3. Saves the model locally.
    4. Calls the model to perform imputation on the dataset located at dataset_path.

- medium_imputation(dataset):

    Performs median imputation on the dataset.

    **Steps:**

    1. Loads the dataset from the provided path.
    2. Uses the median of each column to fill in missing values.
    3. Returns the imputed dataset.


