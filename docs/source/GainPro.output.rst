Metrics Class
========================   

.. automodule:: gainpro.output
   :members:
   :undoc-members:
   :show-inheritance:

Here you will find the class Metrics used during the training and evaluation of the model. 
The Metrics class is responsible for tracking performance metrics during the training and evaluation of the model. 
It stores values such as loss functions, memory usage, and imputed data.

Attributes
------------
    - `hypers`: Instance of Params, storing hyperparameters.
    - `loss_D`: Array storing the Discriminator's loss values over training iterations.
    - `loss_D_evaluate`: Array storing the Discriminator's loss values during evaluation.
    - `loss_G`: Array storing the Generator's loss values over training iterations.
    - `loss_G_evaluate`: Array storing the Generator's loss values during evaluation.
    - `loss_MSE_train`: Array storing the Mean Squared Error (MSE) for training samples.
    - `loss_MSE_train_evaluate`: Array storing MSE values for evaluation samples.
    - `loss_MSE_test`: Array storing MSE values for test samples.
    - `cpu`: Array tracking CPU usage during training.
    - `cpu_evaluate`: Array tracking CPU usage during evaluation.
    - `ram`: Array tracking RAM consumption during training.
    - `ram_evaluate`: Array tracking RAM consumption during evaluation.
    - `ram_percentage`: Array tracking the percentage of RAM used during training.
    - `ram_percentage_evaluate`: Array tracking the percentage of RAM used during evaluation.
    - `data_imputed`: Stores the final imputed dataset after model inference.
    - `ref_data_imputed`: Stores the reference imputed dataset for validation.

Methods
---------

- __init__(self, hypers: Params)

    Initializes the Metrics class by setting up arrays to store performance metrics.

    **Steps:**

    1. Assigns the hypers object (which contains hyperparameters) to self.hypers.
    2. Initializes arrays with zeros for tracking:
        - Generator and Discriminator loss values (loss_D, loss_G).
        - Mean Squared Error (MSE) loss for training and testing (loss_MSE_train, loss_MSE_test).
        - CPU and RAM usage metrics.
    3. Sets data_imputed and ref_data_imputed to None until values are computed.

- create_output(self, data, name: str)

    Stores model output (e.g., imputed data) in CSV files.
    It saves the data to the output_folder, either overwriting existing files or appending new data.

    **Steps:**

    1. Checks whether to override previous files:
        - If hypers.override == 1, it directly saves data as a CSV file.
    2. If override is off (default behavior):
        - If the file already exists:
            - Reads the existing file.
            - Appends new data to it.
            - Ensures column integrity before saving.
        - Otherwise, saves data as a new CSV file.
