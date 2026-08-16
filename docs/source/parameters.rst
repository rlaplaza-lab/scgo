All Parameters
==================

This page lists all parameters you can use in SCGO. For preset functions and their defaults, see :doc:`/api/param_presets`.

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
     - Merge behavior
   * - ``params`` / ``go_params``
     - Deep-merge onto :func:`~scgo.param_presets.get_default_params` via :func:`~scgo.utils.run_helpers.initialize_params`. Nested dicts (e.g. ``optimizer_params["ga"]``, ``calculator_kwargs``) merge recursively; user keys win.
   * - ``ts_params``
     - Deep-merge onto :func:`~scgo.param_presets.get_ts_search_params` via :func:`~scgo.utils.run_helpers.initialize_ts_params`. Not merged with GO defaults. For ``run_go_ts*``, calculator settings align with merged ``go_params`` unless overridden in ``ts_params``.
   * - Forbidden in dicts
     - Top-level ``system_type`` in ``go_params`` / ``ts_params`` (use the run
       ``system_type=`` argument). Identity keys
       (``system_type``, ``surface_config``, ``adsorbate_definition``,
       ``adsorbate_fragment_template``, ``cluster_adsorbate_config``) are also
       forbidden inside ``optimizer_params`` slots — those slots hold algorithm
       hyperparameters only.
   * - Run kwargs
     - ``system_type``, ``surface_config``, ``adsorbates``, ``seed``,
       ``verbosity``, ``output_*`` belong on the ``run_*`` call.
       Top-level ``surface_config`` / adsorbate keys in ``go_params`` (or
       ``ts_params`` for ``surface_config``) are allowed when they agree with
       the run argument.

**Logging** (``verbosity >= 1``): SCGO logs the defaults source and a flat list of user overrides, then the resolved GO optimizer settings or TS NEB configuration. See :doc:`/api/utils`.

Verbosity levels (``run_*`` ``verbosity=`` argument):

.. list-table::
   :widths: 15 85
   :header-rows: 1

   * - Level
     - Behavior
   * - 0
     - Warnings and errors only; no progress bars
   * - 1
     - Normal operation: parameter merge logs, timing summaries, campaign progress, GA phase summaries (initialization, per-generation crossover/mutation/relaxation), and a one-line TorchSim autobatcher memory-scaler summary when GPU probing runs
   * - 2
     - Per-individual GA and initialization detail (offspring outcomes, placement failures, ineligible structures after relaxation); per-pair NEB detail; third-party loggers still suppressed in HPC mode
   * - 3
     - TRACE-level diagnostics (deepest SCGO logging)

Configure the root logger with :func:`~scgo.configure_logging`. Set
``SCGO_LOCAL_DEV=1`` for milder third-party log suppression during local
development (see :doc:`/installation`). Torch-sim's multi-line
``Model Memory Estimation`` stdout prints are suppressed; the scaler summary
above replaces them at default verbosity.

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
     - Connectivity threshold for initialization, post-operator GA checks,
       per-minimum algorithm gates, the ``run_trials`` final structural gate, and
       TS. Accepts a global float or a dict of per-element and/or per-pair
       multipliers (see :doc:`/validation_and_constraints`). Bonded means
       distance ≤ threshold:

       - float ``f``: ``(r_i + r_j) * f``
       - element dict ``{"Pt": 1.8, "C": 1.4}``: ``r_i*f_i + r_j*f_j``
         (missing symbols use ``1.4``)
       - pair entry ``"Pt-C"`` or ``("Pt", "C")``: ``(r_i + r_j) * f_ij``
         (order-independent; pair overrides element-derived thresholds)

       Example for Pt on graphite: ``{"Pt": 1.4, "C": 1.4, "Pt-C": 1.8}``.
       Effective value resolves via
       :func:`~scgo.system_types.resolve_connectivity_factor` with precedence
       ``connectivity_factor`` → ``ClusterAdsorbateConfig.structure_connectivity_factor``
       → ``SurfaceSystemConfig.structure_connectivity_factor`` → ``1.4``. Set
       config-level fallbacks on ``cluster_adsorbate_config`` / ``surface_config``
       (not as a separate top-level key).
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
   * - ``n_jobs``
     - ``1``
     - Single CPU parallelism knob. ``1`` = sequential; ``-1`` = all CPUs;
       ``-2`` = all but one CPU; or a positive worker count. Inherited by GA
       population init, GA offspring, and post-GO validation unless those
       stages are set explicitly.
   * - ``validation_n_jobs``
     - (optional)
     - Parallel workers for post-GO Hessian/force validation. ``None`` (default)
       inherits the top-level ``n_jobs``; an explicit value overrides it.
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

The subsections below list **algorithm hyperparameters** only
(``optimizer_params["simple"|"bh"|"ga"]``). Do not put ``system_type``,
``surface_config``, or adsorbate identity keys in these slots — see
*Parameter resolution* above.

**Simple** (``optimizer_params["simple"]``) — used for 1–2 atom gas clusters only:

.. list-table::
   :widths: 25 10 65

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

**GA** (``optimizer_params["ga"]``):

Parallelism is opt-in and driven by one top-level knob, ``params["n_jobs"]``
(default ``1``, sequential). Set it to ``-2`` (all but one CPU) or ``-1``
(every CPU) to parallelize *every* CPU stage at once — GA population
initialization, GA offspring construction, and post-GO Hessian/force validation.
SCGO keeps the default sequential so it never oversubscribes the host alongside
the internal BLAS / MACE / TorchSIM thread pools. The per-stage keys
(``n_jobs_population_init``, ``n_jobs_offspring``, ``validation_n_jobs``) remain
available as overrides: ``None`` inherits ``n_jobs``, and an explicit value wins
for that stage only. The production/torchsim/UMA/UPET benchmark presets already
default to ``-2``. So in practice:

.. code-block:: python

   params = get_default_params()
   params["n_jobs"] = -2  # one switch parallelizes population init, offspring, and validation

.. list-table::
   :widths: 25 10 65

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
     - ``None`` (inherits ``n_jobs``)
     - Workers for population initialization. ``None`` inherits the top-level ``params["n_jobs"]``; pass ``-1`` (all CPUs), ``-2`` (all but one), or a positive worker count to enable parallelism.
   * - ``n_jobs_offspring``
     - ``None`` (inherits ``n_jobs``)
     - Workers for offspring construction. Same semantics as ``n_jobs_population_init``; ``None`` inherits the top-level ``n_jobs``.
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

**BH** (``optimizer_params["bh"]``):

.. list-table::
   :widths: 25 10 65

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

TS Parameters
-------------

Passed as ``ts_params`` to ``run_ts_search``, ``run_ts_campaign``, ``run_go_ts``, etc. Sparse dicts are merged with :func:`~scgo.param_presets.get_ts_search_params` defaults at run time.

**Core:**

.. list-table::
   :widths: 25 10 60

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
     - ``2.0`` / ``0.75`` (adsorbate)
     - Max energy gap to attempt TS (eV)
   * - ``use_torchsim``
     - ``True``
     - Use TorchSim for NEB
   * - ``dedupe_minima``
     - ``True``
     - Remove duplicate minima before pairing
   * - ``connectivity_factor``
     - ``1.4``
     - Same connectivity spec as GO (float or per-element/pair dict); resolved
       with the same precedence for TS structural gates.
   * - ``similarity_tolerance``
     - (default)
     - Minima similarity tolerance for pairing
   * - ``similarity_pair_cor_max``
     - ``0.1``
     - Pair-correlation cap for similarity
   * - ``minima_energy_tolerance``
     - ``1e-5``
     - Energy tolerance when deduplicating minima
   * - ``write_timing_json``
     - ``False``
     - Write ``{ts_run_dir}/timing.json``; enables ``go_ts_timing.json`` rollup in ``run_go_ts``

**NEB:**

.. list-table::
   :widths: 25 15 50

   * - ``neb_n_images``
     - ``5`` / ``7`` (adsorbate)
     - Number of images
   * - ``neb_steps``
     - ``"auto"`` / ``2000`` (bare surface) / ``4000`` (adsorbate)
     - Maximum optimization steps
   * - ``neb_fmax``
     - ``0.20``
     - Force convergence (eV/\ :math:`\AA`); shared across all system types
   * - ``neb_spring_constant``
     - ``0.1`` / ``0.5`` (adsorbate)
     - Spring constant (eV/\ :math:`\AA`\ :sup:`2`)
   * - ``neb_climb``
     - ``False`` / ``True`` (adsorbate)
     - Use climbing image
   * - ``use_parallel_neb``
     - ``True``
     - Batch multiple NEB bands in one TorchSim force eval (all system types)
   * - ``parallel_neb_max_bands``
     - ``None`` / ``4`` (surface)
     - Explicit cap on concurrent bands in the parallel NEB runner. Surface
       defaults to ``4`` bands per force batch for OOM safety on large slab
       cells. When ``None``, bands are chunked by
       ``parallel_neb_max_batch_atoms`` instead
   * - ``parallel_neb_max_batch_atoms``
     - ``6000`` / ``4000`` (surface)
     - Atom budget (sum of ``n_images * n_atoms``) per fused parallel NEB force
       batch, used only when ``parallel_neb_max_bands`` is ``None``. Also sizes
       the TorchSim relaxer's ``expected_max_atoms`` / ``max_atoms_to_try``. A
       chunk that hits CUDA OOM is retried once at half the budget
   * - ``max_endpoint_mismatch``
     - ``None`` / ``1.25`` (gas adsorbate) / ``1.25`` (surface) / ``1.5`` (surface adsorbate)
     - Å geometric gate on comparator ``max_diff``; when set, also enables the pre-NEB endpoint-displacement check. Surface presets newly gain this (was unset).
   * - ``neb_prescreen_clash_distance``
     - ``1.0`` (bare gas) / ``0.7`` (surface + adsorbate)
     - Interior NEB image min mobile pairwise distance (Å) below which the initial path is rejected.
   * - ``min_saddle_prominence``
     - ``0.10`` (bare gas) / ``0.40`` (surface + adsorbate)
     - Minimum interior-max prominence (eV) above both endpoints for a band to pass the pre-NEB energy profile gate.
   * - ``neb_max_spurious_barrier``
     - ``8.0`` (all types)
     - Maximum allowed IDPP barrier (eV) before a band is rejected as discontinuous.
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
     - ``0.20``
     - TorchSim force tolerance (mapped internally). Keep equal to ``neb_fmax`` unless you intentionally diverge them
   * - ``torchsim_max_steps``
     - ``"auto"`` / ``2000`` (bare surface) / ``4000`` (adsorbate)
     - TorchSim step budget (mapped internally)

**NEB pre-screen gates:**

Before any NEB optimization, ``validate_initial_neb_path`` runs for **every**
system type (bare gas, adsorbate, and surface; TorchSim and serial ASE paths).
``validate_initial_neb_energy_profile`` runs only when ``max_endpoint_mismatch``
is set (bare ``gas_cluster`` leaves it ``None`` and skips the energy-profile
screen):

- Interior-image clash check (min mobile pairwise distance vs
  ``neb_prescreen_clash_distance``) always runs; the aligned endpoint-displacement
  gate additionally runs when ``max_endpoint_mismatch`` is set.
- Energy-profile check (barrier cap ``neb_max_spurious_barrier``; endpoint-energy
  drift ``> 0.5`` eV and interior-max prominence below ``min_saddle_prominence``)
  runs only when ``max_endpoint_mismatch`` is set and canonical endpoint energies
  are available. Bands with fewer than three images skip the prominence/drift
  check.

Per-system-type defaults for the three pre-screen knobs are listed under
:doc:`/validation_and_constraints` (bare gas is looser:
``neb_prescreen_clash_distance=1.0`` / ``min_saddle_prominence=0.10``;
surface and adsorbate are tighter: ``0.7`` / ``0.40``).

**Adsorbate NEB specifics** (beyond the gates above):

- Fragment-wise adsorbate matching and core-anchored alignment
- Pair selection prefers activated hops (moderate mismatch / core RMS),
  oversamples candidates (``10× max_pairs``), and re-ranks by IDPP profile so
  the NEB budget favors robust interior maxima when any exist
- Climbing NEB: two-stage only when the IDPP path has a robust interior maximum
  (barrier ``≥ 1.0`` eV); endpoint-max and soft interior IDPP climb from step 0
- Finalize also rejects barriers ``> 8`` eV

**Surface NEB (differences from gas):**

- ``neb_interpolation_mic=True`` (forced)
- ``neb_surface_cell_remap=True``
- ``neb_surface_lattice_rotation=True`` for bare ``surface_cluster`` /
  ``surface``; ``False`` for ``surface_cluster_adsorbate`` /
  ``surface_adsorbate`` (registry-safe)
- ``neb_surface_max_lattice_shift=1``
- ``parallel_neb_max_bands=4`` (parallel NEB path stays on; bands are
  chunked four-at-a-time for OOM safety on large slab cells)
- ``parallel_neb_max_batch_atoms=4000`` (atom budget used when the band cap is
  cleared to ``None``; kept at/below the previous 4-band path so the TorchSim
  memory-scaler disk cache bucket is reused)

Surface Config
--------------

.. list-table::
   :widths: 25 10 65

   * - ``slab``
     - Required
     - ASE Atoms object
   * - ``name``
     - ``"slab"``
     - Path-key surface segment (filesystem-safe). Graphite preset uses
       ``"graphite"`` (e.g. ``Pt5_OH_OH_graphite_searches``).
   * - ``adsorption_height_min``
     - ``1.2`` (class) / ``2.0`` (``make_surface_config``)
     - Minimum height above slab (\ :math:`\AA`).
   * - ``adsorption_height_max``
     - ``3.0`` (class) / ``3.5`` (``make_surface_config``)
     - Maximum height above slab (\ :math:`\AA`).
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
   * - ``defect_bias_probability``
     - ``0.0`` (class) / ``0.5`` if ``monovacancy`` else ``0.0`` (preset)
     - Fraction (0.0–1.0) of placements biased onto a recorded slab vacancy;
       ignored when the slab has no vacancy (see :doc:`/surface_slab_guide`).
   * - ``comparator_use_mic``
     - ``True``
     - Use MIC in structure comparator on surfaces
   * - ``cluster_init_vacuum``
     - ``8.0``
     - Extra vacuum for cluster init on slab
   * - ``init_mode``
     - ``"smart"``
     - Surface cluster init mode: ``smart``, ``seed+growth``, ``random_spherical``,
       or ``template`` (see :doc:`/api/initialization`)
   * - ``max_placement_attempts``
     - ``200`` (class) / ``500`` (``make_surface_config``); presets use ``1000``
     - Max cluster placement attempts on slab
   * - ``structure_connectivity_factor``
     - ``1.4``
     - Fallback connectivity spec (float or dict; same forms as top-level
       ``connectivity_factor``) when the GO/TS param is omitted. Read by
       :func:`~scgo.system_types.resolve_connectivity_factor` after any explicit
       ``connectivity_factor`` and the ``ClusterAdsorbateConfig`` value, before the
       module default. Used for slab-contact / supported-deposit checks, not only
       placement.

.. note::
   Use only one of the layer options, not both. See :doc:`/api/surface`.

.. note::
   The graphite preset functions override ``adsorption_height_min`` /
   ``adsorption_height_max``: ``make_graphite_surface_config`` and
   ``make_defected_graphite_surface_config`` use **0.5 / 1.0 Å**, while
   ``make_graphene_surface_config`` and ``make_n_doped_graphite_surface_config``
   use **0.5 / 1.5 Å**. The values above are the class and
   ``make_surface_config`` defaults, which apply only when you build a config
   directly rather than through a preset. See :doc:`/surface_slab_guide`.

Adsorbate Config
----------------

.. list-table::
   :widths: 25 10 65

   * - ``height_min``
     - ``0.9``
     - Minimum placement height (\ :math:`\AA`).
   * - ``height_max``
     - ``2.2``
     - Maximum placement height (\ :math:`\AA`).
   * - ``max_placement_attempts``
     - ``80``
     - Maximum placement tries
   * - ``blmin_ratio``
     - ``0.7``
     - Clash threshold
   * - ``structure_connectivity_factor``
     - ``1.4``
     - Fallback connectivity spec (float or dict; same forms as top-level
       ``connectivity_factor``) when the GO/TS param is omitted.

See Also
----------

- :doc:`/quickstart` - How to use these parameters
- :doc:`/api/param_presets` - Preset functions and their defaults
- :doc:`/api/runner_api` - API function documentation
- :doc:`/validation_and_constraints` - How validation and constraints interact
