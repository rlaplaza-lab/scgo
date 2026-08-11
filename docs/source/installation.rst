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

For **UPET**, after installing ``scgo[upet]``, manually install the required
``vesin`` version:

.. code-block:: bash

   pip install 'vesin==0.6.0' --force-reinstall --no-deps

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
Database discovery uses an **in-process** registry with a filesystem glob
fallback; it does not write ``.scgo_db_registry.json`` on disk.

Set ``SCGO_LOCAL_DEV=1`` or ``configure_logging(..., hpc_mode=False)`` for noisier
local logs.

**Processor count (parallelism default):** SCGO defaults to a single worker
so it never silently oversubscribes the host alongside internal BLAS / MACE /
TorchSIM thread pools. You control parallelism through ``n_jobs_population_init``
and ``n_jobs_offspring`` in ``go_params`` (or the corresponding ``params`` dict):

- ``1`` (default): sequential — one worker, no parallelism
- ``-1``: use every logical CPU
- ``-2``: use every logical CPU except one (good default on shared nodes)
- ``N`` (positive integer): cap to ``N`` workers

Example (enable parallelism for population initialization):

.. code-block:: python

   from scgo import run_go
   from scgo.param_presets import get_default_params

   params = get_default_params()
   params["optimizer_params"]["ga"]["n_jobs_population_init"] = -2
   params["optimizer_params"]["ga"]["n_jobs_offspring"] = -2

   results = run_go("Pt5", params=params, seed=42, system_type="gas_cluster")

The same ``n_jobs`` semantics apply to ``validation_n_jobs`` for post-GO
Hessian/force validation.

Direct ``ga_go`` / ``bh_go`` knobs (not accepted in ``run_*`` ``go_params``):
``db_enable_expression_indexes``, ``ga_adaptive_retry_enabled``,
``ga_fast_prefilter_enabled``.

Publishing releases
-------------------

Maintainers: GitHub Actions **Publish to PyPI** (``workflow_dispatch``,
``confirm=publish``). Configure trusted publishing for the ``pypi`` environment
(and ``testpypi`` if used).
