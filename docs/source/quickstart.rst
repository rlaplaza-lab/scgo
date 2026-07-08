Quick Start
===========

SCGO helps you find the lowest-energy atomic structures using global optimization. This guide shows how to use all supported workflows.

System Types
------------

You must specify one of four system types:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Type
     - Use when
   * - ``gas_cluster``
     - Optimizing a cluster in vacuum
   * - ``surface_cluster``
     - Optimizing a cluster on a surface
   * - ``gas_cluster_adsorbate``
     - Optimizing a cluster with adsorbates in vacuum
   * - ``surface_cluster_adsorbate``
     - Optimizing a cluster with adsorbates on a surface

For surface types, you need a ``surface_config``. For adsorbate types, you need ``adsorbates``.

-----------
Gas Cluster
-----------

Optimize a simple cluster in vacuum.

**Fast test with EMT:**

.. code-block:: python

   from scgo import run_go
   from scgo.param_presets import get_testing_params

   results = run_go(
       ["Pt"] * 4,
       params=get_testing_params(),
       seed=42,
       system_type="gas_cluster",
   )

   for energy, atoms in results:
       print(f"Energy: {energy:.4f} eV, Formula: {atoms.get_chemical_formula()}")

For multi-element clusters (bimetallics, oxides), atom order follows the
composition list you pass in so GA crossover can pair structures safely.
Initialization favours placing heavier elements first while keeping diversity
across the population. Details: :doc:`/api/initialization`.

**Production run with MACE:**

.. code-block:: python

   from scgo import run_go
   from scgo.param_presets import get_default_params

   params = get_default_params()
   params["calculator_kwargs"]["model_name"] = "mace_mp_small"

   results = run_go(
       "Pt5",
       params=params,
       seed=42,
       system_type="gas_cluster",
   )

--------------
On a Surface
--------------

Use the built-in graphite surface or create your own slab.

**Using graphite preset:**

.. code-block:: python

   from scgo import run_go, make_graphite_surface_config
   from scgo.param_presets import get_default_params

   surface_config = make_graphite_surface_config(slab_layers=3)

   results = run_go(
       "Pt5",
       params=get_default_params(),
       seed=42,
       surface_config=surface_config,
       system_type="surface_cluster",
   )

Use :func:`~scgo.adsorption_energy` to compare adsorption energies on a slab.

**Using a custom slab:**

.. code-block:: python

   from ase.build import fcc111
   from scgo import run_go, make_surface_config
   from scgo.param_presets import get_default_params

   slab = fcc111("Pt", size=(3, 3, 3), vacuum=10.0)
   surface_config = make_surface_config(slab)

   results = run_go(
       "Pt4",
       params=get_default_params(),
       seed=42,
       surface_config=surface_config,
       system_type="surface_cluster",
   )

Defining Custom Surfaces
~~~~~~~~~~~~~~~~~~~~~~~~

Use ``SurfaceSystemConfig`` or the simpler ``make_surface_config()`` helper.

**Slab motion options:**

- ``fix_all_slab_atoms=True``: Entire slab stays frozen (default)
- ``n_relax_top_slab_layers=2``: Allow top 2 layers to relax
- ``n_fix_bottom_slab_layers=1``: Freeze bottom layer only
- Both layer counts = ``None``: Entire slab can relax

Do not use ``n_relax_top_slab_layers`` together with ``n_fix_bottom_slab_layers``.

**Full example:**

.. code-block:: python

   from ase.build import fcc111
   from scgo import run_go
   from scgo.param_presets import get_default_params
   from scgo.surface import SurfaceSystemConfig

   slab = fcc111("Fe", size=(4, 4, 3), vacuum=12.0)

   surface_config = SurfaceSystemConfig(
       slab=slab,
       adsorption_height_min=1.2,
       adsorption_height_max=2.5,
       fix_all_slab_atoms=False,
       n_relax_top_slab_layers=2,
   )

   results = run_go(
       "Pt5",
       params=get_default_params(),
       seed=42,
       surface_config=surface_config,
       system_type="surface_cluster",
   )

---------------
With Adsorbates
---------------

Add adsorbate molecules (OH, CO, etc.) to your cluster.

**Gas phase with adsorbate:**

.. code-block:: python

   from ase import Atoms
   from scgo import run_go
   from scgo.param_presets import get_default_params

   oh = Atoms("OH", positions=[[0, 0, 0], [0, 0, 0.96]])

   results = run_go(
       ["Pt"] * 5,
       params=get_default_params(),
       seed=42,
       system_type="gas_cluster_adsorbate",
       adsorbates=oh,
   )

**Multiple adsorbates:**

.. code-block:: python

   from ase import Atoms
   from scgo import run_go
   from scgo.param_presets import get_default_params

   oh1 = Atoms("OH", positions=[[0, 0, 0], [0, 0, 0.96]])
   oh2 = Atoms("OH", positions=[[0, 0, 0], [0, 0, 0.96]])

   results = run_go(
       "Pt5",
       params=get_default_params(),
       seed=42,
       system_type="gas_cluster_adsorbate",
       adsorbates=[oh1, oh2],
   )

Defining Custom Adsorbates
~~~~~~~~~~~~~~~~~~~~~~~~~~

Any ASE ``Atoms`` object is a valid adsorbate fragment. The GA will:

- Place fragments on cluster surface sites
- Keep fragments rigid (bonds stay intact)
- Optionally reposition fragments during optimization

**Tuning placement:**

.. code-block:: python

   from ase import Atoms
   from scgo import run_go
   from scgo.param_presets import get_default_params
   from scgo.cluster_adsorbate import ClusterAdsorbateConfig

   oh = Atoms("OH", positions=[[0, 0, 0], [0, 0, 0.96]])

   params = get_default_params()
   params["cluster_adsorbate_config"] = ClusterAdsorbateConfig(
       height_min=0.9,
       height_max=2.2,
       max_placement_attempts=200,
   )
   params["freeze_adsorbate_internal_geometry"] = True  # Keep fragment rigid

   results = run_go(
       "Pt5",
       params=params,
       seed=42,
       system_type="gas_cluster_adsorbate",
       adsorbates=oh,
   )

Use :func:`~scgo.is_true_minimum` or :func:`~scgo.perform_local_relaxation` to
validate or re-relax candidates outside a full GO run.

--------------------
Surface + Adsorbates
--------------------

Combine surface and adsorbates.

.. code-block:: python

   from ase import Atoms
   from scgo import run_go, make_graphite_surface_config
   from scgo.param_presets import get_default_params

   surface_config = make_graphite_surface_config(slab_layers=3)
   oh = Atoms("OH", positions=[[0, 0, 0], [0, 0, 0.96]])

   results = run_go(
       "Pt5",
       params=get_default_params(),
       seed=42,
       surface_config=surface_config,
       system_type="surface_cluster_adsorbate",
       adsorbates=oh,
   )

------------------
Transition States
------------------

Find transition states between optimized structures.

**TS from existing minima** (after a prior ``run_go`` or manual GO output):

.. code-block:: python

   from scgo import run_ts_search
   from scgo.param_presets import get_ts_search_params

   ts_params = get_ts_search_params(system_type="gas_cluster", seed=42)
   ts_params["max_pairs"] = 10

   # Campaign root: reads Pt5_searches/, writes Pt5_ts_results/ as sibling
   results = run_ts_search(
       "Pt5",
       ts_params=ts_params,
       seed=42,
       output_dir="results/pt5_gas_mace",
       system_type="gas_cluster",
   )

   # Or pass the searches directory directly (parent becomes campaign root)
   results = run_ts_search(
       "Pt5",
       ts_params=ts_params,
       seed=42,
       searches_dir="results/pt5_gas_mace/Pt5_searches",
       system_type="gas_cluster",
   )

**GO + TS combined:**

.. code-block:: python

   from scgo import run_go_ts
   from scgo.param_presets import get_torchsim_ga_params, get_ts_search_params

   go_params = get_torchsim_ga_params(system_type="gas_cluster", seed=42)
   go_params["optimizer_params"]["ga"].update(niter=10, population_size=50)

   ts_params = get_ts_search_params(system_type="gas_cluster", seed=42)
   ts_params["max_pairs"] = 15

   summary = run_go_ts(
       "Pt5",
       go_params=go_params,
       ts_params=ts_params,
       seed=42,
       system_type="gas_cluster",
   )

**On a surface:**

.. code-block:: python

   from scgo import run_go_ts, make_graphite_surface_config
   from scgo.param_presets import get_torchsim_ga_params, get_ts_search_params

   surface_config = make_graphite_surface_config(slab_layers=3)

   go_params = get_torchsim_ga_params(
       system_type="surface_cluster",
       surface_config=surface_config,
       seed=42,
   )

   ts_params = get_ts_search_params(
       system_type="surface_cluster",
       surface_config=surface_config,
       seed=42,
   )
   ts_params["max_pairs"] = 10

   summary = run_go_ts(
       "Pt5",
       go_params=go_params,
       ts_params=ts_params,
       seed=42,
       surface_config=surface_config,
       system_type="surface_cluster",
   )

------------
Campaigns
------------

Run multiple compositions in one call. Composition builders
(``build_one_element_compositions``, ``build_two_element_compositions``) live in
``scgo.runner_api``, not the top-level ``scgo`` package.

**Global optimization:**

.. code-block:: python

   from scgo import run_go_campaign
   from scgo.param_presets import get_testing_params
   from scgo.runner_api import build_one_element_compositions

   # Pt2, Pt3, Pt4, Pt5, Pt6
   compositions = build_one_element_compositions("Pt", min_atoms=2, max_atoms=6)

   results = run_go_campaign(
       compositions,
       params=get_testing_params(),
       seed=42,
       system_type="gas_cluster",
   )
   # results is dict[formula, list[(energy, Atoms)]]

Failed compositions (e.g. initialization ``SCGOValidationError`` on extreme
stoichiometries) are logged, recorded as empty lists in the returned dict, and
skipped so the rest of the campaign continues. See :doc:`/api/initialization`
for multi-element atom ordering and placement behaviour.

**Binary compositions:**

.. code-block:: python

   from scgo import run_go_campaign
   from scgo.param_presets import get_testing_params
   from scgo.runner_api import build_two_element_compositions

   # All Au/Pt combinations with 2-4 total atoms
   compositions = build_two_element_compositions("Au", "Pt", min_atoms=2, max_atoms=4)

   results = run_go_campaign(
       compositions,
       params=get_testing_params(),
       seed=42,
       system_type="gas_cluster",
   )

**TS from existing minima** (each formula needs a prior ``{formula}_searches/`` tree):

.. code-block:: python

   from scgo import run_ts_campaign
   from scgo.param_presets import get_ts_search_params
   from scgo.runner_api import build_one_element_compositions

   compositions = build_one_element_compositions("Pt", min_atoms=4, max_atoms=6)

   results = run_ts_campaign(
       compositions,
       ts_params=get_ts_search_params(system_type="gas_cluster", seed=42),
       seed=42,
       output_dir="benchmark/results",  # shared campaign root
       system_type="gas_cluster",
   )

**GO + TS for multiple compositions:**

.. code-block:: python

   from scgo import run_go_ts_campaign
   from scgo.param_presets import get_testing_params, get_ts_search_params
   from scgo.runner_api import build_one_element_compositions

   compositions = build_one_element_compositions("Pt", min_atoms=4, max_atoms=5)

   results = run_go_ts_campaign(
       compositions,
       go_params=get_testing_params(),
       ts_params=get_ts_search_params(system_type="gas_cluster", seed=42),
       seed=42,
       output_dir="benchmark/results",
       system_type="gas_cluster",
   )

See :doc:`/api/runner_api` for full signatures.

--------------------
Output directories
--------------------

``output_dir`` means different things depending on the runner. See also
:doc:`/api/runner_api`.

.. list-table::
   :widths: 22 28 50
   :header-rows: 1

   * - Runner
     - ``output_dir`` is
     - Default when omitted
   * - ``run_go``
     - The ``{formula}_searches/`` directory itself
     - ``{formula}_searches/`` in the current working directory
   * - ``run_go_campaign``
     - Campaign parent; each composition → ``{parent}/{formula}_searches/``
     - Each composition → ``{formula}_searches/`` in CWD (no shared parent)
   * - ``run_go_ts``
     - Campaign root → ``{root}/{formula}_searches/`` and ``{root}/{formula}_ts_results/``
     - ``scgo_runs/{formula}_{calculator_slug}/`` (see ``output_root`` / ``output_stem`` below)
   * - ``run_go_ts_campaign``
     - Campaign parent; each composition → ``{parent}/{formula}_campaign/…``
     - ``scgo_runs/go_ts_campaign_{calc}/``
   * - ``run_ts_search``
     - Campaign root (or an existing ``*_searches/`` path — parent is inferred)
     - CWD; minima from ``{formula}_searches/``, TS to ``{formula}_ts_results/``
   * - ``run_ts_campaign``
     - Shared campaign root for all compositions
     - CWD per composition

``output_root`` and ``output_stem`` (``run_go_ts`` / ``run_go_ts_campaign`` only):
when ``output_dir`` is omitted, the default root is
``{output_root or ./scgo_runs}/{output_stem or formula}_{calculator_slug}/``
(for example ``examples/results/pt5_gas_mace/``).

``searches_dir`` (``run_ts_search`` only): explicit path to a GO searches
directory; the campaign root becomes ``searches_dir.parent``.

**Example — ``run_go_ts`` with ``output_root`` / ``output_stem``:**

GO and TS use the same campaign layout: ``{formula}_searches/`` and
``{formula}_ts_results/`` are siblings; each holds ``run_*`` directories,
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
           └── run_<timestamp>_<microseconds>/

------------------
On-disk layout
------------------

.. _output-files:

GO and TS write sibling ``{formula}_searches/`` and ``{formula}_ts_results/``
trees (see *Output directories* for path resolution). Each tree contains
datetime-tagged ``run_*`` work directories, campaign summaries, and deduplicated
exports.

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
     - One new ``run_*`` per call under ``{formula}_searches/``
   * - ``run_go_campaign``
     - One shared ``run_id`` for all compositions (override with ``run_id=``)
   * - ``run_ts_search`` / ``run_go_ts``
     - TS mints a fresh ``run_*`` under ``{formula}_ts_results/`` (independent of GO)

Repeat ``run_go`` to add sibling ``run_*`` directories; SCGO merges prior minima
via database discovery and deduplication. :func:`~scgo.utils.run_tracking.get_run_directories`
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

-----------------------
Reading prior results
-----------------------

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

----------
Parameters
----------

Quick parameter selection:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Preset
     - Use for
   * - ``get_testing_params()``
     - Fast tests (EMT calculator)
   * - ``get_default_params()``
     - Default production (MACE)
   * - ``get_torchsim_ga_params(...)``
     - GPU-accelerated with TorchSim
   * - ``get_ts_search_params(...)``
     - Transition state search

See :doc:`/parameters` for all options and :doc:`/api/param_presets` for details.

----------
Examples
----------

Working examples in the repository (see also ``examples/README.md``):

- ``examples/example_pt5_gas.py``: Pt5 in gas phase (timing JSON + on-disk layout documented in-script)
- ``examples/example_pt5_graphite.py``: Pt5 on graphite
- ``examples/example_pt5_oh_gas.py``: Pt5 + OH in gas phase
- ``examples/example_pt5_2oh_graphite.py``: Pt5 + 2OH on graphite

Each example enables ``write_timing_json`` in both ``go_params`` and ``ts_params``
so per-run ``timing.json`` and campaign ``go_ts_timing.json`` are written. See
*On-disk layout* above.
