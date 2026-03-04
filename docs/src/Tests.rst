Tests
=============

In this section, we offer you information about how to run and use our unit tests.
Unit testing is a crutial part when developing a project, as it tests individual units or components of your code. 
It helps to ensure that the code is working as expected and that the code is not broken when new features are added. 

In order for you to run a unittest, you should use the following command:

.. code-block:: bash

    python -m unittest <name_of_the_test_file>


If you want to have more information and feedback from the test, you should use the -v flag:

.. code-block:: bash

      python -m unittest -v <name_of_the_test_file>



Additinally, if you want to run the entire battery of tests all at once, you can use the following command:

.. code-block:: bash

    python -m unittest discover 


Test Coverage
----------------------

One of the most import aspects of unit testing is the test coverage.
Test coverage is a measure of how much of your code is being tested by your unit tests.
It is important to have a high test coverage, as it ensures that your code is being tested thoroughly and that there are no bugs in your code.

In order to address this, you can use the coverage package, which is a tool for measuring code coverage of your tests.

To install the coverage package, you can use the following command:

.. code-block:: bash

    pip install coverage

To run the tests with coverage, you can use the following command:

.. code-block:: bash

    coverage run -m unittest <name_of_the_test_file>

Just like before, if you intend to run all the unittest at once, you can use the following command:

.. code-block:: bash

    coverage run -m unittest discover

After running the test, if you want to see the coverage report, you can use the following command:

.. code-block:: bash

    coverage report

This report will write in your terminal the coverage of each file and the overall coverage of your project.

If you want an even more detail assessment of the coverage, you can use the following command:

.. code-block:: bash

    coverage html

This command will create a html file in the htmlcov folder, which you can open in your browser and see the coverage of each file,
the lines that were executed and the lines that were not executed.


Test_generate_reference
--------------------------------------

This test aims to check the functionality of the generate_reference function, 
which is the one responsible for generating the synthetic reference for the imputation when one is not provided by the user.

This test receives as input the dataset with the missing values, the hint rate and the missing rate.
It is also used a random seed to ensure reproducibility.

After running the create_ref() function, a reference dataset is created and saved as reference_generated.csv.
Finally, it compares the generated reference with the expected reference and checks if they are equal.

.. code-block:: python

    np.testing.assert_array_equal(df_reference, output.values, "Reference Dataset generated successfully")


Test_hint_generation 
-----------------------------------

This test aims to check the functionality of the hint_generation function, 
which is responsible for generating the hint matrix. The hint matrix is created from the mask matrix, 
and it can be seen as an aditional help for the discriminator to better distinguish the values as observed or missing.


This test receives as input a mask matrix and a hint rate.
It is also used a random seed to ensure reproducibility.

After running the generate_hint() function, a hint matrix is created and saved as hint_matrix.csv.
Finally, it compares the generated hint with the expected hint and checks if they are equal.


Test_hyper
------------------------
This test aims to check the functionality of the class Hypers, which is responsible for storing the hyperparameters of the model.
There are five main aspets tested in this unittest:
 
1. test_read_json: This test checks if the hyperparameters are being read correctly from the json file.
2. test_initialization: This test checks if the model can work with the default hyperparameters.
3. test_read_hyperparameters: This test checks if the hyperparameters are being read correctly from the json file.
4. test_update_hypers: This test checks if the hyperparameters are being updated correctly.
5. test_invalid_json: This test checks if the model can handle an invalid json file.

Test_imputation_with_reference 
--------------------------------

This test aims to check the functionality of the model to perform the imputation of a proteomics dataset, 
testing the entire process and all the classes.

This test receives as input the dataset with the missing values, the reference dataset, the hint rate, the miss rate, the hyperparameters, 
and the generator and discriminator networks.
It is also used a random seed to ensure reproducibility.

After running the train_ref() function, the model is trained and the imputation is performed.
Finally, it compares the imputed dataset with the expected imputed dataset and checks if they are equal.

Test_impute_no_reference 
------------------------------

This test is in almost every aspect similar to the previous one, but it tests the model when the reference dataset is not provided by the user.
The model should be able to generate a synthetic reference and perform the imputation correctly.

This test receives as input the dataset with the missing values, the hint rate, the miss rate, the hyperparameters,
and the generator and discriminator networks.
It is also used a random seed to ensure reproducibility.

As advertised, when the model does not receive a reference dataset, there will be two training phases.
The first one is the evaluation run (evaluate()), where a percentage of the values are concealed during the training phase and then the dataset is imputed.
The second one is the imputation run (train()), where a proper training phase takes place using the entire dataset.

Finally, it compares the imputed dataset with the expected imputed dataset and checks if they are equal.



