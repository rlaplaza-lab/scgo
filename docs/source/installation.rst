Installation
=============

SCGO is published on `PyPI <https://pypi.org/project/scgo/>`_ and can also be
installed from source for development.

Prerequisites
-------------

- Python 3.12+
- SQLite with JSON1 extension (required; use ``pysqlite3-binary`` for pip
  installs if your Python sqlite3 lacks JSON1)
- CUDA (for GPU acceleration with MLIPs)

PyPI (recommended)
------------------

Install the core package with exactly one MLIP extra:

.. code-block:: bash

   pip install "scgo[mace]"

Or with UMA support:

.. code-block:: bash

   pip install "scgo[uma]"

Or with UPET support:

.. code-block:: bash

   pip install "scgo[upet]"
   pip install 'vesin==0.6.0' --force-reinstall --no-deps

For pip-only installs, ensure ``nvalchemi-toolkit-ops`` is available for the
MACE stack and uninstall ``vesin``/``vesin-torch`` if you encounter
TorchSim-related errors (except for UPET, which needs the pinned ``vesin``
above).

Calculator ``device`` options (``cuda``, ``cpu``, and ``mps`` where supported)
are validated: an unsupported explicit device raises
``SCGOValidationError`` instead of silently using CPU.

Conda (development from source)
---------------------------------

For contributors or editable installs with the full MACE + dev toolchain:

.. code-block:: bash

   git clone https://github.com/rlaplaza-lab/scgo.git
   cd scgo
   conda env create -f environment.yml
   conda activate scgo

The conda environment installs SCGO in editable mode with ``[mace,dev]``
(MACE/TorchSim plus test and lint tooling). Note that ``vesin`` and
``vesin-torch`` conflict with the TorchSim stack used by the MACE/UMA extras
and should not be installed in those environments. The UPET extra requires a
pinned ``vesin==0.6.0`` in a separate environment.

Editable install from source
----------------------------

.. code-block:: bash

   git clone https://github.com/rlaplaza-lab/scgo.git
   cd scgo
   pip install -e ".[mace]"   # or: pip install -e ".[uma]" / ".[upet]"

Development Installation
------------------------

For development with tests and linting:

.. code-block:: bash

   pip install -e ".[mace,dev]"  # or: "[uma,dev]" / "[upet,dev]"
   pre-commit install

Dependency Notes
----------------

- SCGO requires exactly one of the ``[mace]``, ``[uma]``, or ``[upet]`` extras for MLIP support
- The MACE, UMA, and UPET extras use incompatible dependency stacks
- SQLite JSON1 extension is required for database operations
  (``pysqlite3-binary`` recommended for pip installs)
- Sella is optional for advanced optimization features and requires a C toolchain
- SCGO allows ``scipy>=1.14,<3`` to resolve cleanly with fairchem UMA
  dependencies

Parallel jobs and output directories
------------------------------------

SCGO generates unique run folders (``run_YYYYMMDD_HHMMSS_ffffff``), so parallel
jobs launched from the same parent output directory usually write to different
``*.db`` files. Log lines like ``Using cached results for: ...`` are normally
in-process cache hits, not a lock by themselves.

SQLite can still serialize writes when two jobs touch the same database file
(for example, reusing the same explicit ``run_id`` or output path), and shared
filesystems may add contention for registry lock files. For large parallel
campaigns, prefer one output directory per job (or job-local scratch, then copy
results back).

For performance-sensitive workflows that call ``ga_go`` or ``bh_go`` directly
(not the high-level ``run_go`` / ``run_go_ts`` runners), optional tuning knobs
are available on those algorithm entry points:

- ``db_enable_expression_indexes=True`` — JSON expression indexes for metadata
  filtering/sorting paths.
- ``ga_adaptive_retry_enabled=True`` (default) with
  ``ga_retry_floor_multiplier``/``ga_retry_ceiling_multiplier`` — bound retry
  budgets without hard-capping exploration.
- ``ga_fast_prefilter_enabled=True`` (default) — low-cost clash rejection before
  full structural validation.

These keys are **not** accepted inside ``go_params`` / ``params`` passed to
``run_*``; use the algorithm functions directly if you need them.

HPC and shared filesystems
--------------------------

When running on Slurm clusters or network filesystems (Lustre, GPFS, NFS):

**SQLite**

SCGO keeps WAL mode off by default (fewer ``-wal``/``-shm`` issues on shared
filesystems). Prefer writing active ``*.db`` files under job-local scratch
(``$SLURM_TMPDIR`` or site-specific scratch) when you can, then copying results
back to project storage.

**Registry**

Discovery may write ``.scgo_db_registry.json`` and ``.scgo_db_registry.lock``
(with ``flock`` on Linux) for fast database listing. When your run lives under a
directory whose name ends in ``_searches``, the index is kept at that parent
only at the searches root (not in nested work-unit folders). If your filesystem does not honor
``flock``, use separate output directories per job or avoid parallel registry
updates.

**Logging**

Batch-friendly defaults suppress noisy third-party loggers. For local debugging,
set ``SCGO_LOCAL_DEV=1`` or call ``configure_logging(..., hpc_mode=False)``.

Publishing releases
-------------------

Maintainers publish to PyPI via the **Publish to PyPI** GitHub Actions
workflow (``workflow_dispatch``). Configure trusted publishing on PyPI for
the ``pypi`` and ``testpypi`` GitHub environments, then run the workflow
with ``confirm`` set to ``publish``.
