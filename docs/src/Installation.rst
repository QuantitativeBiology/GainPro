Installation
================

Using a pip command
-------------------

We have submitted a package to the Python Package Index (PyPI) for easy installation. 
You can install the package using the following command:

.. code-block:: bash

    pip install GenerativeProteomics

This way, you can install the package and its dependencies in one go.

After that, you can import all the functions and classes from the package of the model and use them in your code. 

GitHub
-------------------

If you prefer to use the code of the GenerativeProteomics model directly, you can access it in our GitHub repository.

https://github.com/QuantitativeBiology/GenerativeProteomics

In order to clone the repository, you should use the following command:

.. code-block:: bash

    git clone https://github.com/QuantitativeBiology/GainPro/

After cloning the repository, you should create a Python environment (versions 3.10 and 3.11).
If you have Conda installed, you can use the following command:

.. code-block:: bash

    conda create -n proto python=3.10

Then, you should activate the environment previously created:

.. code-block:: bash

    conda activate proto

Once the environment is all set up, it is important you install the following dependencies:

- **torch**
- **torchinfo**
- **numpy**
- **tqdm**
- **pandas**
- **scikit-learn**
- **optuna**
- **argparse**
- **psutil**

You can install them seperately, or by running the following command:

.. code-block:: bash

    pip install -r requirements.txt

After all of this, you are all set up and ready to use the model.
