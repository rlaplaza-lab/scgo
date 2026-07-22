Installation
=============

SCGO is on `PyPI <https://pypi.org/project/scgo/>`_. Use **exactly one** MLIP
extra per environment (``[mace]``, ``[uma]``, or ``[upet]``); the stacks conflict.

Prerequisites
-------------

- Python 3.12+
- SQLite with JSON1 (``pysqlite3-binary`` if your build lacks it)
- CUDA for GPU MLIPs

PyPI
----

.. code-block:: bash

   pip install "scgo[mace]"    # or [uma] / [upet]
   # UPET only:
   # pip install 'vesin==0.6.0' --force-reinstall --no-deps

For MACE/TorchSim pip installs, ensure ``nvalchemi-toolkit-ops`` is available
and avoid ``vesin`` / ``vesin-torch`` unless you are on the UPET extra.

Unsupported calculator ``device`` values raise ``SCGOValidationError`` (no silent
CPU fallback).

From source
-----------

.. code-block:: bash

   git clone https://github.com/rlaplaza-lab/scgo.git
   cd scgo
   conda env create -f environment.yml   # [mace,dev]; or:
   # pip install -e ".[mace,dev]"         # or [uma,dev] / [upet,dev]
   conda activate scgo
   pre-commit install

``environment.yml`` is MACE-oriented. Use a separate env for UMA or UPET
(``vesin==0.6.0`` for UPET).

Optional: Sella (needs a C toolchain). ``scipy>=1.14,<3`` for fairchem UMA.

Parallel jobs and HPC
---------------------

Run folders are unique (``run_YYYYMMDD_HHMMSS_ffffff``), so parallel jobs under
the same parent usually write different ``*.db`` files. Prefer one output
directory (or scratch) per job when sharing a filesystem.

SQLite defaults to WAL off (fewer ``-wal``/``-shm`` issues on Lustre/GPFS/NFS).
Discovery may write ``.scgo_db_registry.json`` / ``.scgo_db_registry.lock`` at the
``*_searches`` root. If ``flock`` is unreliable, avoid concurrent registry updates.

Set ``SCGO_LOCAL_DEV=1`` or ``configure_logging(..., hpc_mode=False)`` for noisier
local logs.

Direct ``ga_go`` / ``bh_go`` knobs (not accepted in ``run_*`` ``go_params``):
``db_enable_expression_indexes``, ``ga_adaptive_retry_enabled``,
``ga_fast_prefilter_enabled``.

Publishing releases
-------------------

Maintainers: GitHub Actions **Publish to PyPI** (``workflow_dispatch``,
``confirm=publish``). Configure trusted publishing for the ``pypi`` environment
(and ``testpypi`` if used).
