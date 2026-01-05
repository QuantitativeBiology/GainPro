# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

#-- Path setup --------------------------------------------------------------

import os
import sys  
sys.path.insert(0, os.path.abspath('..'))

print("Sphynx sys.path", sys.path)


autodoc_mock_imports = ["gainpro"]

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'GainPro'
copyright = '2025, Diogo Ferreira, Emanuel Gonçalves, Jorge Ribeiro, Leandro Sobral, Rita Gama'
author = 'Diogo Ferreira, Emanuel Gonçalves, Jorge Ribeiro, Leandro Sobral, Rita Gama'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [ "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",'sphinx.ext.autosectionlabel']

templates_path = ['_templates']
exclude_patterns = ["tests/*",
                    "test.py",
                    "utils.py"]



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
