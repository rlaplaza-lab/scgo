All Parameters
==================

This page lists all parameters you can use in SCGO. For preset functions and their defaults, see :doc:`/api/param_presets`.

--------------------
Parameter resolution
--------------------

All high-level ``run_*`` functions share the same contract:

1. **Safe defaults** — pass ``params=None``, ``go_params=None``, or ``ts_params=None`` to use full preset defaults.
2. **Partial overrides** — pass a dict with only the keys you want to change; runners merge with defaults before execution.
3. **Presets encouraged** — start from a :doc:`/api/param_presets` builder, inspect/edit, then pass to ``run_*``.

**Merge rules**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Dict
     - Merge behaviour
   * - ``params`` / ``go_params``
     - Deep-merge onto :func:`~scgo.param_presets.get_default_params` via :func:`~scgo.utils.run_helpers.initialize_params`. Nested dicts (e.g. ``optimizer_params["ga"]``, ``calculator_kwargs``) merge recursively; user keys win.
   * - ``ts_params``
     - Deep-merge onto :func:`~scgo.param_presets.get_ts_search_params` via :func:`~scgo.utils.run_helpers.initialize_ts_params`. Not merged with GO defaults. For ``run_go_ts*``, calculator settings align with merged ``go_params`` unless overridden in ``ts_params``.
   * - Forbidden in dicts
     - Top-level ``system_type`` in ``go_params`` / ``ts_params`` (use the run ``system_type=`` argument). Factory-default ``optimizer_params[*]["system_type"]`` values are ignored when they match :func:`~scgo.param_presets.get_default_params`; explicit non-default values must match the run argument.
   * - Run kwargs
     - ``system_type``, ``surface_config``, ``adsorbates``, ``seed``, ``verbosity``, ``output_*`` belong on the ``run_*`` call, not inside preset dicts (except ``surface_config`` may also appear in presets when it must agree with the run argument).

**Logging** (``verbosity >= 1``): SCGO logs the defaults source and a flat list of user overrides, then the resolved GO optimizer settings or TS NEB configuration. See :doc:`/api/utils`.

Verbosity levels (``run_*`` ``verbosity=`` argument):

.. list-table::
   :widths: 15 85
   :header-rows: 1

   * - Level
     - Behaviour
   * - 0
     - Warnings and errors only; no progress bars
   * - 1
     - Normal operation: parameter merge logs, timing summaries, campaign progress, and GA phase summaries (initialization, per-generation crossover/mutation/relaxation)
   * - 2
     - Per-individual GA and initialization detail (offspring outcomes, placement failures, ineligible structures after relaxation); third-party loggers still suppressed in HPC mode
   * - 3
     - TRACE-level diagnostics (deepest SCGO logging)

Configure the root logger with :func:`~scgo.configure_logging`. Set
``SCGO_LOCAL_DEV=1`` for milder third-party log suppression during local
development (see :doc:`/installation`).

**Workflow**

.. code-block:: python

   from scgo import run_go_ts
   from scgo.param_presets import get_default_params, get_ts_search_params

   go_params = get_default_params()
   go_params["optimizer_params"]["ga"]["niter"] = 8

   ts_params = get_ts_search_params(system_type="gas_cluster")
   ts_params["max_pairs"] = 12

   summary = run_go_ts(
       "Pt5",
       go_params=go_params,
       ts_params=ts_params,
       system_type="gas_cluster",
       seed=7,
   )

-------------
GO Parameters
-------------

Passed as ``params`` or ``go_params`` to ``run_go``, ``run_go_campaign``, ``run_go_ts``, etc.

**Algorithm selection**

Runners call :func:`~scgo.runner_api.select_scgo_minima_algorithm` automatically:

- ``gas_cluster`` only, ≤2 mobile atoms → ``simple`` (``simple_go.db``)
- 3 atoms, no adsorbate → Basin Hopping (``bh_go.db``)
- 3 atoms, adsorbate system types → Genetic Algorithm (``ga_go.db``)
- ≥4 atoms → Genetic Algorithm (``ga_go.db``)

**Top-Level:**

.. list-table::
   :widths: 25 10 65
   :header-rows: 1

   * - ``calculator``
     - ``"MACE"``
     - Calculator: ``"MACE"``, ``"UMA"``, ``"UPET"``, or ``"EMT"``
   * - ``calculator_kwargs``
     - ``{}``
     - Calculator options (e.g. ``{"model_name": "mace_mp_small"}``). Unsupported
       ``device`` values raise ``SCGOValidationError``.
   * - ``seed``
     - ``None``
     - Random seed (function argument overrides)
   * - ``fitness_strategy``
     - ``"low_energy"``
     - ``"low_energy"``, ``"high_energy"``, or ``"diversity"``
   * - ``diversity_reference_db``
     - ``None``
     - Glob pattern for reference DBs (for diversity mode)
   * - ``connectivity_factor``
     - ``1.4``
     - Connectivity threshold (covalent radii multiplier) for initialization
       validation and post-operator GA checks; see :doc:`/api/initialization`
   * - ``allow_cluster_fragmentation``
     - ``False``
     - Allow cluster to split (surface only)
   * - ``allow_adsorbate_surface_detachment``
     - ``False``
     - Allow adsorbates without cluster contact
   * - ``enforce_adsorbate_subgraph_integrity``
     - ``True``
     - Keep adsorbate fragments connected
   * - ``freeze_adsorbate_internal_geometry``
     - ``False``
     - Keep adsorbate fragments rigid
   * - ``surface_config``
     - ``None``
     - Required for surface runs (prefer the run-function ``surface_config=``;
       a top-level key in ``go_params`` is also allowed)
   * - ``cluster_adsorbate_config``
     - ``None``
     - Adsorbate placement knobs (in ``go_params`` only)
   * - ``validation_n_jobs``
     - (optional)
     - Parallel workers for post-GO Hessian/force validation
   * - ``validate_with_hessian``
     - ``False``
     - Run vibrational analysis
   * - ``tag_final_minima``
     - ``True``
     - Mark final structures in database
   * - ``fmax_threshold``
     - ``0.05``
     - Force threshold for validation (eV/\ :math:`\AA`)
   * - ``check_hessian``
     - ``True``
     - Check Hessian during validation
   * - ``imag_freq_threshold``
     - ``50.0``
     - Imaginary frequency cutoff (cm\ :sup:`-1`)

**Simple (``optimizer_params["simple"]``)** — used for 1–2 atom gas clusters only:

.. list-table::
   :widths: 25 10 65
   :header-rows: 1

   * - ``optimizer``
     - ``"FIRE"``
     - Local optimizer name
   * - ``fmax``
     - ``0.05``
     - Force convergence (eV/\ :math:`\AA`)
   * - ``niter``
     - ``1``
     - Relaxation steps
   * - ``niter_local_relaxation``
     - ``"auto"``
     - Local relaxation budget

**GA (``optimizer_params["ga"]``):**

.. list-table::
   :widths: 25 10 65
   :header-rows: 1

   * - ``population_size``
     - ``"auto"``
     - Number of structures in population
   * - ``niter``
     - ``"auto"``
     - Number of generations
   * - ``mutation_probability``
     - ``0.4``
     - Probability of mutating each structure
   * - ``offspring_fraction``
     - ``0.5``
     - Fraction of population replaced each generation
   * - ``fmax``
     - ``0.05``
     - Force convergence (eV/\ :math:`\AA`)
   * - ``vacuum``
     - ``10.0``
     - Vacuum around clusters (\ :math:`\AA`)
   * - ``use_adaptive_mutations``
     - ``True``
     - Auto-adjust mutation rate
   * - ``early_stopping_niter``
     - ``10``
     - Stop if no improvement for N generations
   * - ``n_jobs_population_init``
     - ``-2``
     - Parallel jobs for population init (-2 = all but 1)
   * - ``n_jobs_offspring``
     - ``-2``
     - Parallel jobs for offspring
   * - ``write_timing_json``
     - ``False``
     - Write ``{run_dir}/timing.json``; enables ``go_ts_timing.json`` rollup in ``run_go_ts``
   * - ``detailed_timing``
     - ``False``
     - Include per-generation timing
   * - ``stagnation_trigger``
     - ``4``
     - Generations without improvement before adaptive mutation boost
   * - ``stagnation_full_trigger``
     - ``8``
     - Stronger stagnation threshold
   * - ``recovery_window``
     - ``2``
     - Generations to watch after a mutation boost
   * - ``aggressive_burst_multiplier``
     - ``1.8``
     - Mutation-rate multiplier on stagnation
   * - ``max_mutation_probability``
     - ``0.65``
     - Cap on adaptive mutation probability
   * - ``batch_size``
     - ``None``
     - TorchSim batch size (when using a relaxer)
   * - ``relaxer``
     - ``None``
     - Optional TorchSim relaxer instance

**BH (``optimizer_params["bh"]``):**

.. list-table::
   :widths: 25 10 65
   :header-rows: 1

   * - ``temperature``
     - 500K
     - Temperature for accepting moves
   * - ``dr``
     - ``0.2``
     - Maximum step size (\ :math:`\AA`)
   * - ``move_fraction``
     - ``0.3``
     - Fraction of atoms to move
   * - ``deduplicate``
     - ``True``
     - Remove duplicates
   * - ``energy_tolerance``
     - ``1e-5``
     - Energy tolerance for duplicates (eV)
   * - ``move_strategy``
     - ``"random"``
     - Atom move strategy
   * - ``comparator_tol``
     - (default)
     - Structure comparator tolerance
   * - ``comparator_pair_cor_max``
     - (default)
     - Pair correlation cutoff for deduplication
   * - ``comparator_n_top``
     - ``None``
     - Optional ``n_top`` for comparator
   * - ``write_timing_json``
     - ``False``
     - Write ``{run_dir}/timing.json``; enables ``go_ts_timing.json`` rollup in ``run_go_ts``
   * - ``detailed_timing``
     - ``False``
     - Include per-iteration timing breakdown

-------------
TS Parameters
-------------

Passed as ``ts_params`` to ``run_ts_search``, ``run_ts_campaign``, ``run_go_ts``, etc. Sparse dicts are merged with :func:`~scgo.param_presets.get_ts_search_params` defaults at run time.

**Core:**

.. list-table::
   :widths: 25 10 60
   :header-rows: 1

   * - ``calculator``
     - ``"MACE"``
     - Calculator for TS search
   * - ``calculator_kwargs``
     - ``{}``
     - Calculator options
   * - ``max_pairs``
     - ``None``
     - Maximum minima pairs to check (None = all)
   * - ``energy_gap_threshold``
     - ``2.0``
     - Max energy gap to attempt TS (eV)
   * - ``use_torchsim``
     - ``True``
     - Use TorchSim for NEB
   * - ``use_parallel_neb``
     - ``False``
     - Run NEB in parallel
   * - ``dedupe_minima``
     - ``True``
     - Remove duplicate minima before pairing
   * - ``connectivity_factor``
     - ``1.4``
     - Connectivity threshold
   * - ``similarity_tolerance``
     - (default)
     - Minima similarity tolerance for pairing
   * - ``similarity_pair_cor_max``
     - ``0.1``
     - Pair-correlation cap for similarity
   * - ``minima_energy_tolerance``
     - ``1e-5``
     - Energy tolerance when deduplicating minima
   * - ``torchsim_batch_size``
     - ``5``
     - TorchSim NEB batch size
   * - ``write_timing_json``
     - ``False``
     - Write ``{ts_run_dir}/timing.json``; enables ``go_ts_timing.json`` rollup in ``run_go_ts``

**NEB:**

.. list-table::
   :widths: 25 15 50
   :header-rows: 1

   * - ``neb_n_images``
     - ``5``
     - Number of images
   * - ``neb_steps``
     - ``"auto"`` / ``500``
     - Maximum optimization steps
   * - ``neb_fmax``
     - ``0.05`` / ``0.1``
     - Force convergence (eV/\ :math:`\AA`)
   * - ``neb_spring_constant``
     - ``0.1``
     - Spring constant (eV/\ :math:`\AA`\ :sup:`2`)
   * - ``neb_climb``
     - ``False``
     - Use climbing image
   * - ``neb_align_endpoints``
     - ``True``
     - Align endpoints before interpolation
   * - ``neb_interpolation_mic``
     - ``False`` / ``True``
     - Use minimum image convention
   * - ``neb_perturb_sigma``
     - ``0.0``
     - Gaussian perturbation on band (Å)
   * - ``neb_interpolation_method``
     - ``"idpp"``
     - Interpolation method
   * - ``neb_tangent_method``
     - (default)
     - NEB tangent method
   * - ``torchsim_fmax``
     - ``0.05`` / ``0.1``
     - TorchSim force tolerance (not a runner kwarg; mapped internally)
   * - ``torchsim_max_steps``
     - ``"auto"`` / ``500``
     - TorchSim step budget (mapped internally)

**Surface NEB (differences from gas):**

- ``neb_interpolation_mic=True`` (forced)
- ``neb_surface_cell_remap=True``
- ``neb_surface_lattice_rotation=True``
- ``neb_surface_max_lattice_shift=1``

--------------
Surface Config
--------------

.. list-table::
   :widths: 25 10 65
   :header-rows: 1

   * - ``slab``
     - Required
     - ASE Atoms object
   * - ``adsorption_height_min``
     - ``1.2`` (class) / ``2.0`` (``make_surface_config``)
     - Minimum height above slab (\ :math:`\AA`). Alias: ``height_min``.
   * - ``adsorption_height_max``
     - ``3.0`` (class) / ``3.5`` (``make_surface_config``)
     - Maximum height above slab (\ :math:`\AA`). Alias: ``height_max``.
   * - ``surface_normal_axis``
     - ``2``
     - Normal axis (0=x, 1=y, 2=z)
   * - ``fix_all_slab_atoms``
     - ``True``
     - Keep entire slab frozen
   * - ``n_relax_top_slab_layers``
     - ``None``
     - Top layers to relax
   * - ``n_fix_bottom_slab_layers``
     - ``None``
     - Bottom layers to freeze
   * - ``comparator_use_mic``
     - ``False``
     - Use MIC in structure comparator on surfaces
   * - ``cluster_init_vacuum``
     - (optional)
     - Extra vacuum for cluster init on slab
   * - ``init_mode``
     - ``"smart"``
     - Surface cluster init mode: ``smart``, ``seed+growth``, ``random_spherical``,
       or ``template`` (see :doc:`/api/initialization`)
   * - ``max_placement_attempts``
     - (optional)
     - Max cluster placement attempts on slab
   * - ``structure_connectivity_factor``
     - (optional)
     - Connectivity factor for slab validation

.. note::
   Use only one of the layer options, not both. See :doc:`/api/surface`.
   Surface heights: ``adsorption_height_*`` (canonical) or ``height_*`` alias.
   Adsorbate heights: ``height_*`` (canonical) or ``adsorption_height_*`` alias.
   Conflicting values raise ``SCGOValidationError``.

----------------
Adsorbate Config
----------------

.. list-table::
   :widths: 25 10 65
   :header-rows: 1

   * - ``height_min``
     - ``0.9``
     - Minimum placement height (\ :math:`\AA`). Alias: ``adsorption_height_min``.
   * - ``height_max``
     - ``2.2``
     - Maximum placement height (\ :math:`\AA`). Alias: ``adsorption_height_max``.
   * - ``max_placement_attempts``
     - ``80``
     - Maximum placement tries
   * - ``blmin_ratio``
     - ``0.7``
     - Clash threshold

----------
See Also
----------

- :doc:`/quickstart` - How to use these parameters
- :doc:`/api/param_presets` - Preset functions and their defaults
- :doc:`/api/runner_api` - API function documentation
