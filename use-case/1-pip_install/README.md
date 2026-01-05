# gainpro


In this first part, we will show in a simple and clear way how to install and use `gainpro` 
to perform imputation of missing values of proteomics' datasets.

## Installation


`gainpro` is a Python package for imputation of missing values in the field of proteomics. 
It is currently based on the `Generative Adversarial Imputation Network (GAIN)` architecture.
To use the package, you need to have `Python 3.10` or `Python 3.11` on your system.
To do that, you can create a conda environment, for example.
The package is available on `PyPI` and can be installed using a `pip` command (gainpro 0.2.0).

```bash  
    pip install gainpro 
```

  
By running the pip command, you are also installing all the dependencies required by the package, 
which are the following :

- **torch**
- **torchinfo**
- **numpy**
- **tqdm**
- **pandas**
- **scikit-learn**
- **optuna**
- **argparse**
- **psutil**
- **anndata**

## Input

The package is designed to work with datasets that have missing values, specifically in the context of proteomics.
The input dataset should be in a tabular format, such as a CSV or TSV file.
You can also provide a reference dataset, which is a complete dataset without missing values (helpful for training the model).

It is important to mention that you can define the hyperparameters of the model.

- **input** (path to the input dataset)
- **output** (path to the output dataset)
- **ref** (path to the reference dataset, None if not existent)
- **output_folder** (path to the folder where the output files will be saved)
- **num_iterations** (number of iterations for training the model)
- **batch_size** (size of the batch for training)
- **alpha** (weight for the Generator loss)
- **miss_rate** (missing rate of the dataset, default is 0.1)
- **hint_rate** (rate of hints to be used in the model, default is 0.9)
- **lr_D** (learning rate for the Discriminator, default is 0.001)
- **lr_G** (learning rate for the Generator, default is 0.001)
- **override** (boolean to override the output folder if it already exists, default is False)
- **output_all** (boolean to output all the files, default is False)

## Contents

The package allows you to import and use the following classes, with each one of them having a specific 
role in the imputation process :

1. Data
    - handles datasets with missing values. It preprocesses the dataset, generates necessary masks, and 
    scales the data for model training
    - provides functions like:
        - generate_hint()
        - generate_mask()
        - _create_ref()

2. Metrics
    - tracks performance metrics during the training and evaluation of the model
    - provides functions like:
        - _create_output()

3. Network
    - defines the architecture of the Generator and Discriminator networks
    - provides functions like:
        - generate_sample()
        - impute()
        - evaluate_impute()
        - update_D()
        - update_G()
        - train_ref()
        - evaluate()
        - train()

4. Param
    - contains the hyperparameters of the model
    - provides functions like:
        - read_json()
        - read_hyperparameters()
        - _update_hypers()

5. utils
    - contains utility functions for the model
    - provides functions like:
        - create_csv()
        - create_dist()
        - create_missing()
        - create_output()
        - output()
        - sample_idx()
        - build_protein_matrix()
        - build_protein_matrix_from_anndata()

## Example

In this use-case, you can find a file that showcases how to import and use the functions and classes of gainpro.
This file is called `test.py` and it performs the imputation of missing values on a dataset of proteins from PRIDE.
The dataset in question is called `PXD004452.tsv` and it is also accessible in this directory. 
This dataset has a missing rate of 17.442532054984405%, 8657 samples and 4 features.

To run the file with the PRIDE dataset, you can use the following command:

```bash 
    python test.py 
```


## Expected Output


The imputation model produces several forms of output.
Throughout the imputation process, the model updates on the terminal the progress of the process and 
the loss values of both the Discriminator and Generator.

It produces an `imputed.csv` file, which contains the imputed dataset.
Additionally, it can also produce other csv files with information about the loss values of the Discriminator and Generator, 
as well as the metrics of the imputation process.

In the end, you should have access to the following files:

- imputed.csv
- loss_D.csv
- loss_G.csv
- lossMSE_test.csv
- lossMSE_train.csv
- cpu.csv
- ram.csv 
- ram_percentage.csv


