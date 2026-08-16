Parameter Presets
==================

GO (``params`` / ``go_params``) and TS (``ts_params``) dicts. Merge rules:
:doc:`/parameters`.

Preset Functions
----------------

**Global Optimization:**

.. list-table:: GO Presets
   :widths: 35 65

   * - ``get_testing_params()``
     - Fast EMT-based parameters for testing (small populations, few iterations)
   * - ``get_default_params()``
     - Default MACE-based parameters for production
   * - ``get_minimal_ga_params(seed, model_name)``
     - Compact GA parameters that run sequentially (easier to debug)
   * - ``get_torchsim_ga_params(*, system_type, surface_config, seed, model_name)``
     - MACE + TorchSim for GPU acceleration. Requires ``scgo[mace]``.
   * - ``get_low_effort_torchsim_ga_params(*, system_type, surface_config, seed, model_name)``
     - Reduced-budget (~25%) variant of ``get_torchsim_ga_params`` for demos and CI. Same calculator and relaxer; smaller GA budget; ``n_jobs=1``.
   * - ``get_low_effort_upet_ga_params(*, system_type, surface_config, seed, model_name, version)``
     - Reduced-budget (~25%) UPET GO for demos and CI. TorchSim relaxer is attached after ``model_name`` / ``version``; ``n_jobs=1``.
   * - ``get_low_effort_uma_ga_params(*, system_type, surface_config, seed, model_name, uma_task)``
     - Reduced-budget (~25%) UMA GO for demos and local/Actions CI. FairChem TorchSim relaxer is attached after ``model_name`` / ``uma_task``; ``n_jobs=1``. (UMA is omitted from the Kaggle GPU matrix.)
   * - ``get_default_uma_params()``
     - Default UMA (fairchem) parameters
   * - ``get_uma_ga_benchmark_params(seed, *, model_name, uma_task)``
     - UMA parameters for benchmarking campaigns
   * - ``get_default_upet_params()``
     - Default UPET (metatomic) parameters. Requires ``scgo[upet]``.
   * - ``get_upet_ga_benchmark_params(seed, *, model_name)``
     - UPET + TorchSim benchmark GA parameters
   * - ``get_diversity_params(reference_db_glob, max_references, update_interval)``
     - Bias exploration toward diverse structures
   * - ``get_high_energy_params()``
     - Bias exploration toward high-energy structures

**Transition State Search:**

.. list-table:: TS Presets
   :widths: 35 65

   * - ``get_ts_search_params(calculator, calculator_kwargs, *, system_type, surface_config, seed)``
     - TS-only settings (NEB, calculator, pairing). Requires ``system_type``. For surfaces, also requires ``surface_config``. Default calculator is ``"MACE"``. Empty ``calculator_kwargs`` are filled by :func:`~scgo.param_presets.default_calculator_kwargs`.
   * - ``get_low_effort_ts_search_params(calculator, calculator_kwargs, *, system_type, surface_config, seed)``
     - Reduced-budget (~25%, floored) variant of ``get_ts_search_params`` for demos and CI. Every NEB physics knob is inherited unchanged; only ``neb_steps`` / ``torchsim_max_steps`` shrink. ``max_pairs`` is left uncapped for the caller.
   * - ``low_effort_neb_steps(system_type)``
     - The ``neb_steps`` budget used by :func:`~scgo.param_presets.get_low_effort_ts_search_params` for one system type.
   * - ``get_ts_defaults(system_type)``
     - NEB knob defaults for one system type (used internally by :func:`~scgo.param_presets.get_ts_search_params`; prefer ``get_ts_search_params`` in user code)

.. note::
   Canonical signatures are rendered by the ``automodule`` block below; the
   summary above is a convenience view.

Preset effects (vs defaults)
----------------------------

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
     - MACE benchmark GA stack + TorchSim relaxer; for surface types stamps
       top-level ``surface_config`` only (not into optimizer slots)
   * - ``get_low_effort_torchsim_ga_params()``
     - As ``get_torchsim_ga_params()``, but ~25% of the benchmark GA budget (``niter``, ``population_size``, ``niter_local_relaxation``), sequential (``n_jobs=1``), no early stopping, no timing JSON. Surface types still clamp local relaxation up to 400 steps at run time.
   * - ``get_low_effort_upet_ga_params()``
     - Reduced ~25% GA budget on UPET; TorchSim relaxer attached after ``model_name`` / ``version`` so the PES matches the ASE calculator; ``n_jobs=1``. Surface types still clamp local relaxation up to 400 steps at run time. Stamps top-level ``surface_config`` only.
   * - ``get_low_effort_uma_ga_params()``
     - Reduced ~25% GA budget on UMA; FairChem TorchSim relaxer attached after ``model_name`` / ``uma_task``; ``n_jobs=1``. Surface types still clamp local relaxation up to 400 steps at run time. Stamps top-level ``surface_config`` only.
   * - ``get_default_uma_params()``
     - ``calculator="UMA"`` + FairChem TorchSim relaxer with auto local-step budget
   * - ``get_uma_ga_benchmark_params()``
     - UMA + autobatcher, ``expected_max_atoms=600`` (benchmark parity with TorchSim GA preset); ``relaxer.max_steps`` is ``None`` until GA assigns it from ``niter_local_relaxation``
   * - ``get_diversity_params()``
     - ``fitness_strategy="diversity"`` + reference DB glob and update interval (top-level and BH/GA slots)
   * - ``get_high_energy_params()``
     - ``fitness_strategy="high_energy"``; BH temperature raised to 1000 K
   * - ``get_ts_search_params()``
     - Full flat TS dict for one ``system_type`` (NEB knobs from :func:`~scgo.param_presets.get_ts_defaults`); baseline for TS merge
   * - ``get_low_effort_ts_search_params()``
     - As ``get_ts_search_params()``, but ``neb_steps`` / ``torchsim_max_steps`` reduced to ~25% with a per-type floor (1000 for both bare and adsorbate) so bands still converge to ``neb_fmax``

Parameter reference
-------------------

See :doc:`/parameters` for the full GO, TS, surface, and adsorbate parameter tables.

Available Models
----------------

**MACE models:** ``"mace_matpes_0"``, ``"mace_mp_small"``, ``"mace_mpa_medium"``, ``"mace_off_small"``

**UMA models:** ``"uma-s-1p2"``, ``"uma-s-1p1"``, ``"uma-m-1p1"``

**UPET models:** ``"pet-mad-s"``, ``"pet-mad-xs"``, ``"pet-oam-xl"``, ``"pet-omat-s"``, ``"pet-spice-s"``

Usage Examples
--------------

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

**Low-effort GO + TS (demos, examples, CI):**

Same physics, ~25% of the budget. This is what every script in ``examples/``
and the Kaggle GPU test matrix uses, so the two cannot drift apart.

.. code-block:: python

   from scgo import (
       get_low_effort_torchsim_ga_params,
       get_low_effort_ts_search_params,
       make_graphite_surface_config,
   )

   surface_config = make_graphite_surface_config(slab_layers=3, slab_repeat_xy=3)

   go_params = get_low_effort_torchsim_ga_params(
       system_type="surface_cluster",
       surface_config=surface_config,
       seed=42,
   )

    ts_params = get_low_effort_ts_search_params(
        system_type="surface_cluster",
        surface_config=surface_config,
        seed=42,
    )
    # max_pairs is the dominant TS cost lever and is left to the caller.
    ts_params["max_pairs"] = 6

``get_low_effort_ts_search_params`` already covers MACE, UMA, and UPET uniformly
via its ``calculator`` / ``calculator_kwargs`` arguments — there is no separate
per-calculator TS wrapper.

**Low-effort UPET GO:**

.. code-block:: python

   from scgo import (
       get_low_effort_upet_ga_params,
       get_low_effort_ts_search_params,
       make_graphite_surface_config,
   )

   surface_config = make_graphite_surface_config(slab_layers=3, slab_repeat_xy=3)

   go_params = get_low_effort_upet_ga_params(
       system_type="surface_cluster",
       surface_config=surface_config,
       seed=42,
       model_name="pet-mad-s",
       version="1.5.0",
   )

See :doc:`/quickstart` for complete workflow examples and :doc:`/parameters` for the full parameter list.

Module Reference
----------------

.. automodule:: scgo.param_presets
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: _* TS_DEFAULTS_BY_SYSTEM_TYPE
