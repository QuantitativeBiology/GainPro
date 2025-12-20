# Generative Proteomics

[![PyPi Version](https://img.shields.io/pypi/v/GenerativeProteomics?label=PyPi&color=blue&style=flat&logo=pypi)](https://pypi.org/project/GenerativeProteomics/)
[![Colab](https://img.shields.io/badge/Google_Colab-0061F2?style=flat&logo=googlecolab&color=blue&label=Colab&colorB=grey)](https://colab.research.google.com/drive/1ihtmsv_UvEz74YrLHZvATu1y2qH4X9-r?usp=sharing)
[![Documentation](https://img.shields.io/badge/docs-read%20the%20docs-blue)](https://generativeproteomics.readthedocs.io/en/latest/)
[![HuggingFace](https://img.shields.io/badge/Hugging_Face-grey?style=flat&logo=huggingface&color=grey)](https://huggingface.co/QuantitativeBiology)

In this repository you may find a PyTorch implementation of Generative Adversarial Imputation Networks (GAIN) [[1]](#1) for imputing missing iBAQ values in proteomics datasets.

## Table of Contents

- [Repository Strucure](#repository-structure)
- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [GitHub](#github)
- [Demo](#demo)
- [References](#reference)

## Repository Structure

Here are the main components you'll find in this repository:

1. .github/workflows 
    - contains the code for the automatization of the tests in the repository
2. Datasets
    - directory with datasets with missing values from PRIDE that can be used for testing 
3. GenerativeProteomics: 
    - Contains the core package source code 
4. docs/source
    - contains the information used for the documentation of our work (ReadtheDocs)
5. tests:
    - batery of unittests to assess the model's functionality
6. use-case
    - set of clear examples on how to use our model's functionalities 
    - includes examples on how to install the package and use it, how to run the tests, and how to download and use a pre-trained model from HuggingFace


## Installation

### Pip install

We have submitted a package to the Python Package Index (PyPI) for easy installation. You can install the package using the following command:

```bash
pip install GenerativeProteomics
```

This way, you can install the package and its dependencies in one go.

#### Python API 

```python
from GenerativeProteomics import utils, Network, Params, Metrics, Data
import torch
import pandas as pd

# Load your dataset
dataset_path = "your_dataset.tsv"
dataset_df = utils.build_protein_matrix(dataset_path) # use this function if dataset is a tsv
#dataset_df = pd.read_csv(dataset_path)  # if your dataset is a csv
dataset = dataset_df.values
missing_header = dataset_df.columns.tolist()

# Define your parameters
params = Params(
    input=dataset_path,
    output="imputed.csv",
    ref=None,
    output_folder=".",
    num_iterations=2001,
    batch_size=128,
    alpha=10,
    miss_rate=0.1,
    hint_rate=0.9,
    lr_D=0.001,
    lr_G=0.001,
    override=1,
    output_all=1,
)

# Define model architecture
input_dim = dataset.shape[1]
h_dim = input_dim
net_G = torch.nn.Sequential(
    torch.nn.Linear(input_dim * 2, h_dim),
    torch.nn.Sigmoid()
)
net_D = torch.nn.Sequential(
    torch.nn.Linear(input_dim * 2, h_dim),
    torch.nn.Sigmoid()
)

# Set up the model and data
metrics = Metrics(params)
network = Network(hypers=params, net_G=net_G, net_D=net_D, metrics=metrics)
data = Data(dataset=dataset, miss_rate=0.2, hint_rate=0.9, ref=None)

# Run evaluation and training
network.evaluate(data=data, missing_header=missing_header)
network.train(data=data, missing_header=missing_header)
print("Final Matrix:\n", metrics.data_imputed) 
```
For a more detailed explanation on how to use the model and all the functionalities we have to offer, you can open the `use-case` directory.

### GitHub

If you prefer to use the code of the GenerativeProteomics model directly, you can access it in our GitHub repository and follow the next sequence of commands.

1. Clone this repository:  `git clone https://github.com/QuantitativeBiology/GenerativeProteomics/`
2. Create a Python environment: `conda create -n proto python=3.10` if you have conda installed
3. Activate the previously created environment: `conda activate proto`
4. Install the necessary packages: `pip install -r libraries.txt`


#### How to Use GenerativeProteomics

If you just want to impute a general dataset, the most straightforward and simplest way to run GenerativeProteomics is to run: `python generativeproteomics.py -i /path/to/file_to_impute.csv`
Running in this manner will result in two separate training phases.

1) Evaluation run: In this run a percentage of the values (10% by default) are concealed during the training phase and then the dataset is imputed. The RMSE is calculated with those hidden values as targets and at the end of the training phase a `test_imputed.csv` file will be created containing the original hidden values and the resulting imputation, this way you can have an estimation of the imputation accuracy.

2) Imputation run: Then a proper training phase takes place using the entire dataset. An `imputed.csv` file will be created containing the imputed dataset.

However, there are a few arguments which you may want to change. You can do this using a parameters.json file (you may find an example in `GenerativeProteomics/breast/parameters.json`) or you can choose them directly in the command line.

Run with a parameters.json file: `python generativeproteomics.py --parameters /path/to/parameters.json`<br>
Run with command line arguments: `python generativeproteomics.py -i /path/to/file_to_impute.csv -o imputed_name --ofolder ./results/ --it 2001`

#### How to import and use a pre-trained model 

Instead of running our trained model GenerativeProteomics, you can always use other inference forms. To do so, all you need to do is use the --model flag.

Run the following command in order to use an alternative imputation form: `python generativeproteomics.py -i /path/to/file_to_impute.csv --model <name_of_model>`

#### Arguments:

`-i`: Path to file to impute<br> 
`-o`: Name of imputed file<br> 
`--ofolder`: Path to the output folder<br> 
`--it`: Number of iterations to train the model<br> 
`--miss`: The percentage of values to be concealed during the evaluation run (from `0` to `1`)<br>
`--outall`: Set this argument to `1` if you want to output every metric<br> 
`--override`: Set this argument to `1` if you want to delete the previously created files when writing the new output<br> 
`--model`: Contains the name of the imputation form to run. Default value is the GenerativeProteomics model.


If you want to test the efficacy of the code you may give a reference file containing a complete version of the dataset (without missing values): `python generativeproteomics.py -i /path/to/file_to_impute.csv --ref /path/to/complete_dataset.csv`

Running this way will calculate the RMSE of the imputation in relation to the complete dataset.


#### Demo

In this repository you may find a folder named `breast`, inside it you have a breast cancer diagnostic dataset [[2]](#2) which you may use to try out the code.

`breast.csv`: complete dataset<br>
`breastMissing_20.csv`: the same dataset but with 20% of its values taken out


To simply impute `breastMissing_20.csv` run: `python generativeproteomics.py -i ./breast/breastMissing_20.csv` <br>
If you want to compare the imputation with the original dataset run: `python generativeproteomics.py -i ./breast/breastMissing_20.csv --ref ./breast/breast.csv` or `python generativeproteomics.py --parameters ./breast/parameters.json`


If you want to go deep in the analysis of every metric you either set `--outall` to `1` or you run the code in an IPython console, this way you can access every variable you want in the `metrics` object, e.g. `metrics.loss_D`.


## References
<a id="1">[1]</a> 
J. Yoon, J. Jordon & M. van der Schaar (2018). GAIN: Missing Data Imputation using Generative Adversarial Nets <br>
<a id="2">[2]</a> 
https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
