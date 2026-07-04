Benchmarks
==========

SCGO ships **long-running MLIP regression scripts** under ``benchmark/`` at the
repository root. These are separate from the fast EMT sanity checks in
``tests/benchmarks/``.

Dependencies
------------

- **MACE (default):** ``pip install -e ".[mace]"`` — TorchSim GA + MACE
- **UMA (optional):** pass ``--backend uma``; use a separate environment from MACE

Output layout
-------------

All scripts write under ``benchmark/results/``.

**Gas-phase Pt sweeps** (``benchmark_Pt.py``):

.. code-block:: text

   benchmark/results/
   └── pt5_mace_mace_matpes_0/     # {formula}_{backend}_{model}
       └── Pt5_searches/
           ├── run_<timestamp>_<microseconds>/
           │   ├── metadata.json
           │   ├── timing.json      # when write_timing_json=True (benchmark default)
           │   └── ga_go.db
           ├── results_summary.json
           └── final_unique_minima/

**Surface Pt-on-graphite** (``benchmark_Pt_surface_graphite.py``):

.. code-block:: text

   benchmark/results/
   └── pt_surface_graphite/        # flat campaign root
       └── Pt5_searches/
           ├── run_<timestamp>_<microseconds>/
           │   ├── metadata.json
           │   ├── timing.json
           │   └── ga_go.db
           ├── results_summary.json
           └── final_unique_minima/

TS runs add sibling ``{Formula}_ts_results/`` trees. See :doc:`/quickstart` for
the full run-oriented layout.

Benchmark GA presets enable ``write_timing_json`` and ``detailed_timing`` so CLI
profiling lines match ``{run_dir}/timing.json`` on disk.

Entry points
------------

.. code-block:: bash

   python -m benchmark.benchmark_Pt --help
   python -m benchmark.benchmark_Pt_surface_graphite --help
   python -m benchmark.benchmark_parallel_neb --help

Shared CLI flags (see ``benchmark.benchmark_common.add_common_benchmark_cli``):

- ``--backend {mace,uma}`` — default from ``SCGO_BENCHMARK_BACKEND`` (``mace``)
- ``--model-name``, ``--seed``, ``--uma-task``, ``--clusters``, ``--niter``, ``--population-size``

Pytest
------

``pytest.ini`` excludes ``benchmark/`` from the default test path. To run MLIP
regression hooks:

.. code-block:: bash

   pytest benchmark/ -m slow

See also ``benchmark/README.md`` in the repository for maintainer notes.
