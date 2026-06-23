"""Sphinx configuration for SCGO documentation."""

from __future__ import annotations

import sys
from pathlib import Path

# -- Path setup --------------------------------------------------------------
# Allow autodoc to import scgo when the package is installed editable.

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "SCGO"
copyright = "2026, R. Laplaza"
author = "R. Laplaza"

try:
    from importlib.metadata import version as _pkg_version

    release = _pkg_version("scgo")
except Exception:
    release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
]

# Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_ivar = False

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Dataclass fields are documented twice (attributes + __init__ params) under Sphinx 9.
suppress_warnings = ["autodoc.duplicate_object"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_title = "SCGO"
html_static_path = ["_static"]
html_favicon = "_static/scgo_logo.svg"
html_logo = "_static/scgo_logo.svg"

# Furo theme specific settings
html_theme_options = {
    "sidebar_hide_name": True,
    "light_css_variables": {
        "color-brand-primary": "#2c3e50",
        "color-brand-content": "#2c3e50",
    },
}

# -- Options for autodoc -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html

autodoc_mock_imports = [
    "torch",
    "torch_sim",
    "torch_sim_atomistic",
    "mace",
    "mace_torch",
    "mace.calculators",
    "mace.calculators.mace",
    "fairchem",
    "fairchem.core",
    "e3nn",
    "nvalchemi_toolkit_ops",
]

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# -- Options for intersphinx -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# -- Options for todo extension -----------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/todo.html

todo_include_todos = True
