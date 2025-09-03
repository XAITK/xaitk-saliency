"""
Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import sys
from pathlib import Path

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))
import xaitk_saliency

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "xaitk-saliency"
copyright = "2023, Kitware, Inc."  # noqa: A001
author = "Kitware, Inc."
release = xaitk_saliency.__version__


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx.ext.imgconverter",
    "sphinx-prompt",
    "sphinx_copybutton",
    "sphinx_click",
    "myst_nb",
]

suppress_warnings = [
    # Suppressing duplicate label warning from autosectionlabel extension.
    # This happens a lot across files that happen to talk about the same
    # topics.
    "autosectionlabel.*",
]

# Autosummary templates reference link:
# https://github.com/JamesALeedham/Sphinx-Autosummary-Recursion/tree/master
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path: list[str] = list()  # ['_static']

# -- MyST-NB settings---------------------------------------------------------
nb_execution_mode = "off"

# -- LaTeX engine ------------------------------------------------------------
latex_engine = "lualatex"
