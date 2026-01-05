Network Class
========================

.. automodule:: gainpro.model
   :members:
   :undoc-members:
   :show-inheritance:

Here you will find the class `Network` used during the process of imputation of missing values.
The `Network` class is responsible for handling the architecture of the Generative Adversarial Imputation Network (GAIN) 
used in the model training.

It is also responsible for handling the optimization of both the generator and discriminator, generating imputed values for missing data, 
and evaluating the imputation quality using a reference dataset.

Attributes
-----------
- `hypers` : Instance of the `Params` class containing hyperparameters used in the model training.
- `net_G` : Generator network used in the GAIN model.
- `net_D` : Discriminator network used in the GAIN model.
- `metrics` : Object to store performance metrics.
- `optimizer_G` : Optimizer for the generator.
- `optimizer_D` : Optimizer for the discriminator

Methods
---------

- __init__()

    Initializes the `Network` class by processing the hyperparameters used in the model training.

- generate_sample()

    Generates synthetic samples for missing values using the Generator network.

    **Steps:**

    1. Retrieves the data dimensions (dim) and number of samples (size).
    2. Creates a noise matrix Z with small random values.
    3. Fills missing values in the dataset with Z, keeping observed values unchanged.
    4. Concatenates the processed dataset with the mask matrix.
    5. Feeds the input into the Generator network (net_G) to generate an imputed sample.

- impute()

    Performs imputation using the trained Generator model.

    **Steps:**

    1. Calls generate_sample() to obtain imputed values.
    2. Combines original observed values with the generated imputed values.
    3. Rescales the imputed dataset back to its original scale.
    4. Exports the final imputed dataset to a CSV file.

- _evaluate_impute()

    Evaluates the imputation process on a reference dataset.

    **Steps:**

    1. Generates imputed values for the reference dataset.
    2. Merges the observed reference values with the generated samples.
    3. Rescales the reference imputed dataset back to its original scale.
    4. Finds missing values that were intentionally hidden during evaluation.
    5. Creates a structured CSV file to compare original and imputed values.

- _update_G()

    Updates the Generator network using a loss function.

    **Steps:**

    1. Creates a new dataset by replacing missing values with noise.
    2. Prepares input for the Generator by concatenating it with the mask.
    3. Generates a synthetic sample using the Generator.
    4. Constructs an input for the Discriminator using the generated sample.
    5. Computes two loss terms:
        - Adversarial Loss: Encourages the Generator to fool the Discriminator.
        - MSE Loss: Ensures generated values are close to observed values.
    6. Backpropagates the total loss and updates Generator weights.

- _update_D()

    Updates the Discriminator network using a loss function.

    **Steps:**

    1. Creates an input dataset where missing values are replaced with noise.
    2. Generates synthetic samples using the Generator.
    3. Builds Discriminator inputs with real and fake data.
    4. Computes the loss by comparing real vs. fake inputs.
    5. Backpropagates and updates the Discriminator's weights.

- train_ref()

    Trains the Generator and Discriminator using a reference dataset provided by the user.

    **Steps:**

    1. Initializes the training loop with a defined number of iterations.
    2. Randomly samples mini-batches from the dataset.
    3. Computes updates for the Discriminator using _update_D().
    4. Computes updates for the Generator using _update_G().
    5. Evaluates training loss every 100 iterations.
    6. Logs resource usage (CPU, RAM).
    7. Finalizes training and saves the imputed dataset.

- evaluate()

    Evaluates the trained model using a validation dataset.

    **Steps:**

    1. Adapts the batch size if the dataset is too small.
    2. Repeats the training process but on the reference dataset.
    3. Updates Generator and Discriminator with _update_G() and _update_D().
    4. Computes training and testing errors at each iteration.
    5. Saves performance metrics such as loss, RMSE, and resource usage.

- train()

    Trains the model from scratch.

    **Steps:**

    1. Initializes Generator and Discriminator weights.
    2. Configures optimizers for both networks.
    3. Starts the training process with the imputation dataset.
    4. Updates the networks iteratively using _update_G() and _update_D().
    5. Tracks performance metrics and saves the results.
