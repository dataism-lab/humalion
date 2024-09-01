"""Sphinx configuration."""
project = "humalion"
author = "Dataism Lab"
copyright = "2024, Dataism Lab"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"
