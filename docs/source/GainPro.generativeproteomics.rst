gainpro Class
========================

.. automodule:: gainpro.generativeproteomics
   :members:
   :undoc-members:
   :show-inheritance:

Here you will find the class `gainpro` used during the training and evaluation of the model.
The `gainpro` class is responsible for setting up, initializing, and running the gainpro imputation process. 
It reads input arguments, configures the model, trains it, and saves the results.

Methods
------------

- Main Script (__main__)

    **Steps:**

    1. Parses Command-Line Arguments using Click to obtain user-defined settings.
    2. Loads Hyperparameters either from command-line arguments or a JSON configuration file.
    3. Reads the Dataset (CSV/TSV format) and preprocesses it.
    4. Initializes the Generator (G) and Discriminator (D) Networks with a specific architecture.
    5. Creates the Network Class with the model and hyperparameters.
    6. Trains or Evaluates the Model:
        - If a reference dataset is provided, it runs training with reference (train_ref).
        - Otherwise, runs evaluation (evaluate) followed by training (train).
    7. Records Execution Time and stores it in run_time.csv.
    8. Performs Profiling with cProfile to measure execution performance.
