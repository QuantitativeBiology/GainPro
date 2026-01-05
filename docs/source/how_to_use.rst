How to use gainpro
=================================

If your main goal is simply to just impute a general dataset, the most straightforward and simplest way to use gainpro is to run:

.. code-block:: bash

    python generativeproteomics.py -i /path/to/file_to_impute.csv 

By running it in this manner, it will result in two separate training phases.

1. **Evaluation run**: 
    In this run a percentage of the values (10% by default) are concealed during the training phase and then the dataset is imputed. 
    The RMSE (Root Mean Square-Error) is calculated with those hidden values as targets and at the end of the training phase a **test_imputed.csv** file will be created containing 
    the original hidden values and the resulting imputation. 
    This way you can have an estimation of the imputation accuracy.

2. **Imputation run**: 
    Afterwards, a proper training phase takes place using the entire dataset. An **imputed.csv** file will be created containing the imputed dataset.

However, there might be a few arguments which you may want to change. You can do this using a **parameters.json** file 
(you may find an example in ``datasets/breast/parameters.json``) or you can choose them directly in the command line.

Run with a parameters.json file: 

.. code-block:: bash

    python generativeproteomics.py --parameters /path/to/parameters.json

Run with command line arguments: 

.. code-block:: bash

    python generativeproteomics.py -i /path/to/file_to_impute.csv -o imputed_name --ofolder ./results/ --it 2001

Arguments:

- **-i**: Path to file to impute
- **-o**: Name of imputed file
- **--ofolder**: Path to the output folder
- **--it**: Number of iterations to train the model
- **--miss**: The percentage of values to be concealed during the evaluation run (from 0 to 1)
- **--outall**: Set this argument to 1 if you want to output every metric
- **--override**: Set this argument to 1 if you want to delete the previously created files when writing the new output
- **--model**: Choose the model to use (None if gainpro, otherwise provide name of the pre-trained model)

If you want to assess the efficiency of the code you may provide a reference file containing a complete version of the dataset (without missing values):

.. code-block:: bash

    python generativeproteomics.py -i /path/to/file_to_impute.csv --ref /path/to/complete_dataset.csv

Running this way will calculate the RMSE of the imputation in relation to the complete dataset.