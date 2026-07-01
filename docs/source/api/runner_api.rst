API Reference
============

High-level functions for running global optimization and transition state searches.

----------
Main Functions
----------

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
     - Optimize multiple compositions, return dict[formula, results]
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
- ``verbosity``: 0=quiet, 1=normal (logs parameter merge provenance and resolved settings), 2+=verbose
- ``output_dir`` / ``output_root`` / ``output_stem``: control output locations

See :doc:`/parameters` (*Parameter resolution*) for merge rules and logging behaviour.

----------
Complete Examples
----------

**Gas-phase cluster optimization:**

.. code-block:: python

   from scgo import run_go
   from scgo.param_presets import get_testing_params

   results = run_go(
       "Pt5",
       params=get_testing_params(),
       seed=42,
       system_type="gas_cluster",
   )
   # results: list of (energy, Atoms) tuples

**Surface cluster:**

.. code-block:: python

   from scgo import run_go, make_graphite_surface_config
   from scgo.param_presets import get_testing_params

   surface_config = make_graphite_surface_config(slab_layers=3)

   results = run_go(
       "Pt5",
       params=get_testing_params(),
       seed=42,
       surface_config=surface_config,
       system_type="surface_cluster",
   )

**With adsorbates:**

.. code-block:: python

   from ase import Atoms
   from scgo import run_go
   from scgo.param_presets import get_testing_params

   oh = Atoms("OH", positions=[[0, 0, 0], [0, 0, 0.96]])

   results = run_go(
       "Pt5",
       params=get_testing_params(),
       seed=42,
       system_type="gas_cluster_adsorbate",
       adsorbates=oh,
   )

**GO + TS combined:**

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

**Campaign (multiple compositions):**

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

----------
Utility Functions
----------

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - ``build_one_element_compositions(element, min_atoms, max_atoms)``
     - Create list like ["Pt3", "Pt4", "Pt5"]
   * - ``build_two_element_compositions(el1, el2, min_atoms, max_atoms)``
     - Create all combinations like ["AuPt", "Au2Pt", ...]
   * - ``parse_composition_arg(composition)``
     - Convert string/list/Atoms to symbol list
   * - ``resolve_workflow_seed(run_seed, go_params_seed, ts_params_seed)``
     - Ensure seed consistency across params
   * - ``log_go_ts_summary(summary, verbosity)``
     - Print summary of a GO+TS run

----------
Timing and Profiling
----------

Configure timing in ``params`` / ``go_params`` only (``optimizer_params['ga']`` or ``bh``):

- ``write_timing_json=True``: write ``timing.json`` under each trial directory
- ``detailed_timing=True``: add ``per_generation`` rows (requires ``write_timing_json=True``)

For TS, set ``write_timing_json`` in ``ts_params`` when needed.

See :mod:`scgo.utils.timing_report` for the JSON layout.

----------
Module Reference
----------

.. automodule:: scgo.runner_api
   :members:
   :undoc-members:
   :show-inheritance:
