# SCGO Documentation

This directory contains the Sphinx documentation source for SCGO (Simple Cluster Global Optimization), built with Sphinx.

## Building the Documentation

### Prerequisites

- Python 3.12+
- SCGO installed (`pip install -e ".[mace]"` from the repository root, or `pip install "scgo[mace]"` from PyPI)
- Documentation dependencies: `pip install -r source/requirements.txt` (from this `docs/` directory)

### Building HTML

```bash
# From repository root
pip install -e ".[mace]"
pip install -r docs/source/requirements.txt
cd docs && make html
```

The built documentation will be available in `docs/build/html/index.html`.

### Building PDF

```bash
cd docs && make latexpdf
```

The PDF will be available in `build/latex/scgo.pdf`.

## Documentation Structure

- `source/` — Sphinx source files (RST format)
  - `api/` — API reference documentation (auto-generated from docstrings)
  - `index.rst` — Main documentation index
  - `installation.rst` — Installation instructions
  - `quickstart.rst` — Quick start guide with working examples
  - `conf.py` — Sphinx configuration
  - `requirements.txt` — Documentation build requirements
  - `Makefile` — Sphinx build automation (invoked via `docs/Makefile`)
- `Makefile` — Delegates to `source/Makefile`

## Online Documentation

This documentation is automatically built and published on [ReadTheDocs](https://scgo.readthedocs.io/). The configuration is in `.readthedocs.yaml` in the project root.

## Writing Documentation

- Use reStructuredText (RST) format
- Follow Google-style docstrings in the Python code
- Use `.. autofunction::` and `.. automodule::` directives for API documentation
- Keep examples concise and practical

## Updating API Documentation

The API documentation is automatically generated from docstrings in the Python code. To update:

1. Add/improve docstrings in the source code
2. Run `make html` from `docs/` to rebuild
3. Commit both code and documentation changes

## Style Guide

- Use sentence case for headings
- Keep line length under 88 characters
- Use code blocks for examples
- Be consistent with existing documentation style
