Installation
=============

SCGO is published on `PyPI <https://pypi.org/project/scgo/>`_ and can also be
installed from source for development.

Prerequisites
-------------

- Python 3.12+
- SQLite with JSON1 extension (``pysqlite3-binary`` if needed for pip installs)
- CUDA (for GPU acceleration with MLIPs)

PyPI (recommended)
------------------

Install the core package with exactly one MLIP extra:

.. code-block:: bash

   pip install "scgo[mace]"

Or with UMA support:

.. code-block:: bash

   pip install "scgo[uma]"

For pip-only installs, ensure ``nvalchemi-toolkit-ops`` is available for the
MACE stack and uninstall ``vesin``/``vesin-torch`` if you encounter
TorchSim-related errors.

Conda (development from source)
---------------------------------

For contributors or editable installs with the full MACE + dev toolchain:

.. code-block:: bash

   git clone https://github.com/rlaplaza-lab/scgo.git
   cd scgo
   conda env create -f environment.yml
   conda activate scgo

The conda environment installs SCGO in editable mode with ``[mace,dev]`` (MACE/TorchSim plus test and lint tooling). Note that ``vesin`` and ``vesin-torch`` conflict with the TorchSim stack used by SCGO and should not be installed.

Editable install from source
----------------------------

.. code-block:: bash

   git clone https://github.com/rlaplaza-lab/scgo.git
   cd scgo
   pip install -e ".[mace]"   # or: pip install -e ".[uma]"

Development Installation
------------------------

For development with tests and linting:

.. code-block:: bash

   pip install -e ".[mace,dev]"  # or: pip install -e ".[uma,dev]"
   pre-commit install

Dependency Notes
----------------

- SCGO requires exactly one of the ``[mace]`` or ``[uma]`` extras for MLIP support
- The MACE and UMA extras use incompatible dependency stacks
- SQLite JSON1 extension is required for database operations (``pysqlite3-binary`` recommended for pip installs)
- Sella is optional for advanced optimization features and requires a C toolchain
- SCGO allows ``scipy>=1.14,<3`` to resolve cleanly with fairchem UMA dependencies

Publishing releases
-------------------

Maintainers publish to PyPI via the **Publish to PyPI** GitHub Actions workflow
(``workflow_dispatch``). Configure trusted publishing on PyPI for the ``pypi`` and
``testpypi`` GitHub environments, then run the workflow with ``confirm`` set to
``publish``.
