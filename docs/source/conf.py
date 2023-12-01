# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'marl-factory-grid'
copyright = '2023, Steffen Illium, Robert Mueller, Joel Friedrich'
author = 'Steffen Illium, Robert Mueller, Joel Friedrich'
release = '2.5.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['myst_parser',
              'sphinx.ext.duration',
              'sphinx.ext.doctest',
              'sphinx.ext.autodoc',
              'sphinx.ext.autosummary',

              ]

templates_path = ['_templates']
exclude_patterns = ['marl_factory_grid.utils.proto']

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
from pathlib import Path
import sys
sys.path.insert(0, (Path(__file__).parents[2]).resolve().as_posix())

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
