API Reference
=============

High-level functions for running global optimization and transition state searches.

--------------
Main Functions
--------------

**Single composition:**

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - ``run_go(...)`` → ``list[tuple[float, Atoms]]``
     - Optimize one composition, return list of (energy, Atoms) tuples
   * - ``run_ts_search(...)`` → ``list[dict[str, Any]]``
     - Find transition states for one composition
   * - ``run_go_ts(...)`` → ``dict[str, Any]``
     - Run GO then TS for one composition

**Multiple compositions (campaigns):**

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - ``run_go_campaign(...)`` → ``dict[str, list[tuple[float, Atoms]]]``
     - Optimize multiple compositions; failed compositions are logged and
       skipped (empty list for that formula). See :doc:`/api/initialization`.
   * - ``run_ts_campaign(...)`` → ``dict[str, list[dict[str, Any]]]``
     - Find TS for multiple compositions
   * - ``run_go_ts_campaign(...)`` → ``dict[str, dict[str, Any]]``
     - Run GO+TS for multiple compositions

All functions accept:

- ``composition``: formula string (``"Pt5"``), symbol list (``["Pt"]*5``), or ASE Atoms
- ``params`` / ``go_params``: GO parameter dictionary (``None`` or partial dict; merged with :func:`~scgo.param_presets.get_default_params` at run time)
- ``ts_params``: TS parameter dictionary (``None`` or partial dict; merged with :func:`~scgo.param_presets.get_ts_search_params` at run time)
- ``seed``: random seed for reproducibility (must agree across ``seed=``, ``go_params['seed']``, and ``ts_params['seed']`` when more than one is set)
- ``system_type``: ``"gas_cluster"``, ``"surface_cluster"``, ``"gas_cluster_adsorbate"``, or ``"surface_cluster_adsorbate"`` (run argument only — not inside preset dicts)
- ``surface_config``: required for surface system types; must agree across run argument and preset dicts when both are set
- ``adsorbates``: ASE Atoms or list of Atoms, required for adsorbate system types
- ``verbosity``: 0=quiet (warnings only), 1=normal (parameter merge and timing summaries), 2=debug (library detail), 3=trace (custom TRACE level for deep diagnostics). Progress bars are shown when ``verbosity >= 1`` (see :func:`~scgo.utils.logging.should_show_progress`).

See :doc:`/parameters` (*Parameter resolution*) for merge rules and logging behaviour.

------------------
Output directories
------------------

``output_dir`` semantics differ by runner. Full table and directory-tree examples:
:doc:`/quickstart` (*Output directories*).

.. list-table::
   :widths: 22 28 50
   :header-rows: 1

   * - Runner
     - ``output_dir`` is
     - Also accepts
   * - ``run_go``
     - ``{formula}_searches/`` directory itself
     - —
   * - ``run_go_campaign``
     - Campaign parent → ``{parent}/{formula}_searches/``
     - —
   * - ``run_go_ts``
     - Campaign root → ``{root}/{formula}_searches/`` + ``{root}/{formula}_ts_results/``
     - ``output_root``, ``output_stem``
   * - ``run_go_ts_campaign``
     - Campaign parent → ``{parent}/{formula}_campaign/…``
     - ``output_root``, ``output_stem``
   * - ``run_ts_search``
     - Campaign root (or existing ``*_searches/`` — parent inferred)
     - ``searches_dir``
   * - ``run_ts_campaign``
     - Shared campaign root for all compositions
     - —

-----------------
Complete Examples
-----------------

.. code-block:: python

   from scgo import run_go
   from scgo.param_presets import get_testing_params

   results = run_go(
       "Pt5",
       params=get_testing_params(),
       seed=42,
       system_type="gas_cluster",
   )

See :doc:`/quickstart` for surface, adsorbate, TS, campaign, and output-layout examples.

-----------------
Utility Functions
-----------------

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - ``build_one_element_compositions(element, min_atoms, max_atoms)``
     - Symbol lists for mono-element size scans (e.g. ``[["Pt", "Pt"], ["Pt", "Pt", "Pt"]]``)
   * - ``build_two_element_compositions(el1, el2, min_atoms, max_atoms)``
     - Symbol lists for bimetallic size scans (all ``el1``/``el2`` splits per atom count)
   * - ``parse_composition_arg(comp_str)``
     - Parse a compact formula (``"Pt5"``) or comma-separated symbols (``"Pt,Pt,Au"``)
   * - ``resolve_workflow_seed(*, seed_kw, go_params, ts_params)``
     - Ensure seed consistency across params
   * - ``log_go_ts_summary(logger, summary, *, wall_time_s=None)``
     - Print summary of a GO+TS run

--------------------
Timing and Profiling
--------------------

Configure timing in ``params`` / ``go_params`` only (``optimizer_params['ga']`` or ``bh``):

- ``write_timing_json=True``: write ``timing.json`` under each run directory (alongside ``metadata.json``)
- ``detailed_timing=True``: add ``per_generation`` rows (requires ``write_timing_json=True``)

For TS, set ``write_timing_json`` in ``ts_params`` when needed.

See :mod:`scgo.utils.timing_report` for the JSON layout and
:mod:`scgo.utils.output_paths` for campaign directory helpers.

----------------
Module Reference
----------------

.. automodule:: scgo.runner_api
   :members:
   :undoc-members:
   :show-inheritance:
