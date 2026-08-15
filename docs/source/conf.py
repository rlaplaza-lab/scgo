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
    from scgo import __version__ as release

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

# Treat every missing cross-reference as a warning so doc regressions surface.
nitpicky = True

# Tolerate references we cannot resolve from any inventory, mostly external
# packages (ase_ga publishes no objects.inv; torch_sim/ase/scipy intersphinx is
# unreachable in this build environment) and private/internal helpers that should
# not be linked.
nitpick_ignore = [
    ("py:class", "ase_ga.data.DataConnection"),
    ("py:class", "ase_ga.startgenerator.StartGenerator"),
    ("py:class", "ase_ga.standard_comparators.SequentialComparator"),
    ("py:class", "ase_ga.offspring_creator.OperationSelector"),
    ("py:class", "ase_ga.offspring_creator.OffspringCreator"),
    ("py:func", "ase.Atoms.get_distance"),
    ("py:class", "scipy.spatial._qhull.ConvexHull"),
    ("py:class", "InFlightAutoBatcher"),
]

nitpick_ignore_regex = [
    (r"py:(class|func|method|mod|data|attr|exc|meth)", r"torch_sim\..*"),
    (r"py:(class|func|method|mod|data|attr|exc|meth)", r"ase\..*"),
    (r"py:(class|func|method|mod|data|attr|exc|meth)", r"ase_ga\..*"),
]

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
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- Options for todo extension -----------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/todo.html

todo_include_todos = False


# -- Custom roles ------------------------------------------------------------

from docutils import nodes

def doi_role(name, rawtext, text, lineno, inliner, options=None, content=None):
    options = options or {}
    text = text.strip()
    if "<" in text:
        label, _, target = text.partition("<")
        label = label.strip()
        target = target.rstrip(">").strip()
    else:
        label, target = text, None
    if target:
        return [nodes.reference(rawtext, label, refuri="https://doi.org/" + target, **options)], []
    return [nodes.Text(label)], []

def setup(app):
    app.add_role("doi", doi_role)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
