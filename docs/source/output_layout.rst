Output Layout
=============

This page documents the on-disk directory structure created by SCGO runs.
For a quick reference, see the summary in the :doc:`/quickstart` guide.

------------
Path Keys
------------

The ``path_key`` is a filesystem-safe identifier combining:

- Nanoparticle formula (e.g., ``Pt5``)
- Each adsorbate fragment in order (e.g., ``OH``, ``OH`` → ``OH_OH``)
- ``surface_config.name`` when a surface is used (default ``"slab"``;
  ``make_graphite_surface_config`` sets ``name="graphite"``)

Examples: ``Pt5``, ``Pt5_OH_OH``, ``Pt5_OH_OH_graphite``, ``defected_graphite``,
``OH_n_doped_graphite``.

Chemical composition matching uses ASE-style formulas (e.g., ``H2O2Pt5``).

------------------
Output directories
------------------

``output_dir`` semantics differ by runner:

.. list-table::
   :widths: 22 28 50
   :header-rows: 1

   * - Runner
     - ``output_dir`` is
     - Default when omitted
   * - ``run_go``
     - The ``{path_key}_searches/`` directory itself
     - ``{path_key}_searches/`` in the current working directory
   * - ``run_go_campaign``
     - Campaign parent; each composition → ``{parent}/{path_key}_searches/``
     - Each composition → ``{path_key}_searches/`` in CWD (no shared parent)
   * - ``run_go_ts``
     - Campaign root → ``{root}/{path_key}_searches/`` and ``{root}/{path_key}_ts_results/``
     - ``scgo_runs/{path_key}_{calculator_slug}/`` (see ``output_root`` / ``output_stem`` below)
   * - ``run_go_ts_campaign``
     - Campaign parent; each composition → ``{parent}/{path_key}_campaign/…``
     - ``scgo_runs/go_ts_campaign_{calc}/``
   * - ``run_ts_search``
     - Campaign root (or an existing ``*_searches/`` path — parent is inferred)
     - CWD; minima from ``{path_key}_searches/``, TS to ``{path_key}_ts_results/``
   * - ``run_ts_campaign``
     - Shared campaign root for all compositions
     - CWD per composition

``output_root`` and ``output_stem`` (``run_go_ts`` / ``run_go_ts_campaign`` only):
when ``output_dir`` is omitted, the default root is
``{output_root or ./scgo_runs}/{output_stem or path_key}_{calculator_slug}/``
(for example ``examples/results/pt5_gas_mace/``).

``searches_dir`` (``run_ts_search`` only): explicit path to a GO searches
directory; the campaign root becomes ``searches_dir.parent``.

**Example — ``run_go_ts`` with ``output_root`` / ``output_stem``:**

GO and TS use the same campaign layout: ``{path_key}_searches/`` and
``{path_key}_ts_results/`` are siblings; each holds ``run_*`` directories,
campaign-level summaries, and deduplicated exports. GO databases and TS pair
artifacts live directly under each ``run_*`` (TS uses ``pair_<i>_<j>/`` subdirs).

.. code-block:: text

   results/pt5_gas_mace/
   ├── go_ts_timing.json              # optional (write_timing_json in go/ts params)
   ├── Pt5_searches/
   │   ├── run_20260703_120000_123456/
   │   │   ├── metadata.json
   │   │   ├── timing.json              # optional (write_timing_json=True)
   │   │   └── ga_go.db
   │   ├── results_summary.json
   │   └── final_unique_minima/
   └── Pt5_ts_results/
       ├── run_20260703_130000_654321/
       │   ├── metadata.json
       │   ├── timing.json              # optional (write_timing_json=True)
       │   └── pair_0_1/
       │       └── neb_0_1_metadata.json
       ├── results_summary.json
       ├── ts_network_metadata.json
       └── final_unique_ts/
           └── final_unique_ts_summary.json

**Example — surface adsorbate campaign** (``Pt5`` + 2×``OH`` on graphite):

.. code-block:: text

   results/pt5_2oh_graphite_mace/
   ├── Pt5_OH_OH_graphite_searches/
   │   ├── run_*/
   │   ├── results_summary.json
   │   └── final_unique_minima/
   │       └── Pt5_OH_OH_graphite_minimum_01_<run_id>.xyz
   └── Pt5_OH_OH_graphite_ts_results/
       ├── run_*/
       ├── results_summary.json
       └── final_unique_ts/
           └── Pt5_OH_OH_graphite_ts_01.xyz

**Example — ``run_go_campaign`` with ``output_dir="benchmark/results"``:**

.. code-block:: text

   benchmark/results/
   ├── Pt4_searches/
   ├── Pt5_searches/
   └── Pt6_searches/

**Example — gas-phase MLIP benchmark default** (``benchmark_Pt.py``):

.. code-block:: text

   benchmark/results/
   └── pt5_mace_mace_matpes_0/
       └── Pt5_searches/
           └── run_<timestamp>/

------------------
On-disk layout
------------------

GO and TS write sibling ``{path_key}_searches/`` and ``{path_key}_ts_results/``
trees. Each tree contains datetime-tagged ``run_*`` work directories, campaign
summaries, and deduplicated exports.

Run IDs
~~~~~~~

Each invocation uses ``run_YYYYMMDD_HHMMSS_ffffff`` (microsecond granularity,
UTC-based):

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Runner
     - ``run_id`` behaviour
   * - ``run_go``
     - One new ``run_*`` per call under ``{path_key}_searches/``
   * - ``run_go_campaign``
     - One shared ``run_id`` for all compositions (override with ``run_id=``)
   * - ``run_ts_search`` / ``run_go_ts``
     - TS mints a fresh ``run_*`` under ``{path_key}_ts_results/`` (independent of GO)

Repeat ``run_go`` to add sibling ``run_*`` directories; SCGO merges prior minima
via database discovery and deduplication.
:func:`~scgo.utils.run_tracking.get_run_directories`
lists only datetime-pattern ``run_*`` dirs; custom IDs work at runtime but are
omitted by that helper.

Per-run files
~~~~~~~~~~~~~

Under each ``run_*`` directory:

- ``metadata.json`` — composition, params snapshot, provenance header (``schema_version``,
  ``scgo_version``, ``created_at`` UTC ISO-8601 with ``Z``, ``python_version``, ``timestamp``)
- ``ga_go.db`` / ``bh_go.db`` / ``simple_go.db`` — optimizer database (GO only)
- ``timing.json`` — optional wall-time breakdown (``write_timing_json=True``); includes
  ``run_id``, ``timing_schema_version``, and the same provenance header fields
- ``pair_<i>_<j>/`` — NEB artifacts, ``neb_{i}_{j}_metadata.json``, and optional
  ``timing_{i}_{j}.json`` per pair (TS only)

Campaign-level files:

- ``results_summary.json`` — run statistics and serializable TS pair results
- ``final_unique_minima/`` or ``final_unique_ts/`` — deduplicated structure exports
- ``ts_network_metadata.json`` — minima connectivity graph (TS only)
- ``go_ts_timing.json`` — GO+TS pipeline rollup at the campaign root when timing
  JSON is enabled in ``go_params`` and/or ``ts_params``; includes ``current_go_run_id``,
  ``current_ts_run_id``, and relative paths to per-run ``timing.json`` files when present

See :mod:`scgo.utils.timing_report` for timing JSON layout.

.. _run-ids-and-provenance:

Provenance
~~~~~~~~~~

TS results record endpoint lineage in ``minima_provenance`` (in
``results_summary.json``, NEB metadata, and ``ts_network_metadata.json``):

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Field
     - Meaning
   * - ``schema_version`` / ``scgo_version`` / ``created_at`` / ``python_version``
     - Shared JSON provenance header on summaries, metadata, and timing files
   * - ``run_id``
     - GO run that produced the endpoint minimum
   * - ``source_db`` / ``source_db_relpath``
     - Optimizer database (basename and campaign-relative path)
   * - ``confid`` / ``gaid`` / ``systems_row_id``
     - Row identifiers in the GO database
   * - ``unique_id`` / ``final_id``
     - Dedup and final-export identifiers when present
   * - ``energy``
     - Endpoint energy at pairing time (eV)

See :doc:`/quickstart` for usage examples and :doc:`/api/database` for database
access patterns.

---------------
Reading Results
---------------

To reload minima from completed searches without re-running GO:

.. code-block:: python

   from scgo import load_previous_run_results, SCGODatabaseManager

   minima = load_previous_run_results("Pt5_searches")
   # Or browse databases with context-manager cleanup:
   with SCGODatabaseManager(base_dir="Pt5_searches") as manager:
       refs = manager.load_reference_structures("**/*.db", composition=["Pt"] * 5)

``SCGODatabaseManager`` caching uses both TTL and a lightweight filesystem
fingerprint (database file count/path/mtime), so cached reads are invalidated
when matching database files are added, removed, or updated.

See :doc:`/api/database` for HPC-oriented database access patterns.