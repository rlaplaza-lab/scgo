Parameter Presets
==================

This page explains the parameter presets available in SCGO and what each parameter controls.

Overview
--------

SCGO has two types of parameter dictionaries:

- **GO parameters** (``params`` or ``go_params``): for global optimization
- **TS parameters** (``ts_params``): for transition state searches

Both follow the same contract documented in :doc:`/parameters` (section *Parameter resolution*):

- ``None`` → full safe defaults from presets
- Partial dict → deep-merged at run time (GO via :func:`~scgo.utils.run_helpers.initialize_params`, TS via :func:`~scgo.utils.run_helpers.initialize_ts_params`)
- Preset builders return complete starting dicts; edit keys then pass to ``run_*``

At ``verbosity >= 1``, runners log the defaults source and which user keys overrode them.

----------
Preset Functions
----------

Use these preset functions to get started quickly.

**Global Optimization:**

.. list-table:: GO Presets
   :widths: 35 65
   :header-rows: 1

   * - ``get_testing_params()``
     - Fast EMT-based parameters for testing (small populations, few iterations)
   * - ``get_default_params()``
     - Default MACE-based parameters for production
   * - ``get_minimal_ga_params(seed, model_name)``
     - Compact GA parameters that run sequentially (easier to debug)
   * - ``get_torchsim_ga_params(*, system_type, surface_config, seed, model_name)``
     - MACE + TorchSim for GPU acceleration. Requires ``scgo[mace]``.
   * - ``get_default_uma_params()``
     - Default UMA (fairchem) parameters
   * - ``get_uma_ga_benchmark_params(seed, *, model_name, uma_task)``
     - UMA parameters for benchmarking campaigns
   * - ``get_diversity_params(reference_db_glob, max_references, update_interval)``
     - Bias exploration toward diverse structures
   * - ``get_high_energy_params()``
     - Bias exploration toward high-energy structures

**Transition State Search:**

.. list-table:: TS Presets
   :widths: 35 65
   :header-rows: 1

   * - ``get_ts_search_params(calculator, calculator_kwargs, *, system_type, surface_config, seed)``
     - TS-only settings (NEB, calculator, pairing). Requires ``system_type``. For surfaces, also requires ``surface_config``. Default calculator is ``"MACE"``.
   * - ``get_ts_defaults(system_type)``
     - Return NEB knob defaults for a given system type

----------
Preset effects (vs defaults)
----------

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Preset
     - Main differences from :func:`~scgo.param_presets.get_default_params` / :func:`~scgo.param_presets.get_ts_search_params`
   * - ``get_testing_params()``
     - ``calculator="EMT"``; small ``niter`` / ``population_size`` in all optimizer slots
   * - ``get_default_params()``
     - Canonical MACE production defaults (baseline for GO merge)
   * - ``get_minimal_ga_params()``
     - Sequential GA jobs (``n_jobs_* = 1``); optional ``seed`` / ``model_name``
   * - ``get_torchsim_ga_params()``
     - MACE benchmark GA stack + TorchSim relaxer; sets ``surface_config`` for surface types
   * - ``get_default_uma_params()``
     - ``calculator="UMA"`` + FairChem TorchSim relaxer with auto local-step budget
   * - ``get_uma_ga_benchmark_params()``
     - UMA + fixed 200 local steps, autobatcher, ``expected_max_atoms=600`` (benchmark parity with TorchSim GA preset)
   * - ``get_diversity_params()``
     - ``fitness_strategy="diversity"`` + reference DB glob and update interval
   * - ``get_high_energy_params()``
     - ``fitness_strategy="high_energy"``; BH temperature raised to 1000 K
   * - ``get_ts_search_params()``
     - Full flat TS dict for one ``system_type`` (NEB knobs from :func:`~scgo.param_presets.get_ts_defaults`); baseline for TS merge

----------
Parameter reference
----------

See :doc:`/parameters` for the full GO, TS, surface, and adsorbate parameter tables.

----------
Available Models
----------

**MACE models:** ``"mace_matpes_0"``, ``"mace_mp_small"``, ``"mace_mpa_medium"``, ``"mace_off_small"``

**UMA models:** ``"uma-s-1p2"``, ``"uma-s-1p1"``, ``"uma-m-1p1"``

----------
Usage Examples
----------

**Start from a preset:**

.. code-block:: python

   from scgo.param_presets import get_default_params

   params = get_default_params()
   params["calculator_kwargs"]["model_name"] = "mace_mp_small"
   params["optimizer_params"]["ga"]["population_size"] = 100

**Build TS params:**

.. code-block:: python

   from scgo import make_graphite_surface_config
   from scgo.param_presets import get_ts_search_params

   surface_config = make_graphite_surface_config(slab_layers=3)

   ts_params = get_ts_search_params(
       system_type="surface_cluster",
       surface_config=surface_config,
       seed=42,
   )
   ts_params["max_pairs"] = 20
   ts_params["neb_n_images"] = 7

**Combined GO + TS:**

.. code-block:: python

   from scgo import make_graphite_surface_config
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

See :doc:`/quickstart` for complete workflow examples and :doc:`/parameters` for the full parameter list.

----------
Module Reference
----------

.. automodule:: scgo.param_presets
   :members:
   :undoc-members:
   :show-inheritance:
