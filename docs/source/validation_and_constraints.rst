Validation & Constraints
========================

SCGO couples *validation* (what is a legal structure) with *constraints* (what
is allowed to move) and a small set of outcome toggles. This page explains how
the layers fit together.

Why validation & constraints are coupled
----------------------------------------

"Valid" depends on what moved. A relaxed structure is only legal relative to the
constraints that defined the search space:

- Slab ``FixAtoms`` and the ``fix_core`` core-freeze decide which atoms could
  have shifted.
- The outcome toggles (``allow_cluster_fragmentation``,
  ``allow_adsorbate_surface_detachment``,
  ``enforce_adsorbate_subgraph_integrity``) decide which splits and detachments
  are permitted.

Constraints describe the *search space*; validation rejects structures that fall
out of it. They are two views of the same legality rule, so they must be read
together.

At a glance
-----------

A quick mental model for the page:

- **Constraints** define what atoms may move: slab fixation, core freezing, and
  adsorbate rigidity.
- **Validation** rejects structures that violate the allowed geometry: clashes,
  connectivity, penetration, and fragmentation rules. All gas systems
  (``gas_cluster`` and ``gas_cluster_adsorbate``) are connectivity-checked: the
  whole region must form a single connected component, so a fragmented bare
  ``gas_cluster`` or a gas adsorbate detached from its core is rejected. Surface
  systems (``surface_cluster`` and the ``*_adsorbate`` variants) enforce
  connectivity + slab-contact per mobile subgroup via the shared
  :func:`~scgo.system_types.validate_connectivity_policy` gate. The bare
  ``surface`` type is the one exemption from the *per-minimum deposit/connectivity*
  gate: it still runs per-minimum slab-prefix validation through
  :func:`~scgo.surface.partition.validate_slab_search_config` but skips the
  supported-deposit gate (``needs_supported_deposit_validation=False``).
- **Outcome toggles** decide how permissive the surface search can be when a
  structure is partially detached or fragmented.
- **Numerical floors** are deliberately looser during placement than during final
  validation, so the search explores broadly but legal candidates still get
  rejected at the stricter check.
- **Consistency:** ``basinhopping_go`` and ``geneticalgorithm_go_torchsim``
  validate every candidate through the shared
  :func:`~scgo.system_types.validate_minimum_structure` helper (which wraps
  :func:`~scgo.system_types.validate_structure_for_system_type`); ``simple_go``
  does the same for its single relaxed structure when a ``system_type`` is
  supplied (it returns an empty list if the structure is invalid, and skips the
  gate when ``system_type`` is ``None``). The validation *rule set* is therefore
  identical regardless of algorithm; the connectivity factor resolves through the
  same :func:`~scgo.system_types.resolve_connectivity_factor` precedence at every
  gate.

Validation layers
-----------------

All validation is reached through
:func:`~scgo.system_types.validate_structure_for_system_type`, which branches on
:class:`~scgo.system_types.SystemPolicy`:

- Surface policies (``uses_surface=True``) call
  :func:`~scgo.surface.validation.validate_surface_config_slab_prefix` and then
  :func:`~scgo.surface.validation.validate_supported_cluster_deposit`.
- Gas / adsorbate systems (``gas_cluster`` and ``gas_cluster_adsorbate``, no
  surface) run a clash sibling (``validate_cluster_structure(check_clashes=True,
  check_connectivity=False)``) followed by the unified
  :func:`~scgo.system_types.validate_connectivity_policy` gate, which requires the
  whole region to be a single connected component.
- :func:`~scgo.surface.validation.validate_supported_cluster_deposit` internally
  calls :func:`~scgo.initialization.validate_cluster_structure`
  for the clash check, applies the penetration check, calls
  :func:`~scgo.cluster_adsorbate.validation.validate_adsorbate_fragment_integrity`,
  and then enforces connectivity + slab-contact through the shared
  :func:`~scgo.system_types.validate_connectivity_policy` (surface mode).

.. code-block:: text

   validate_structure_for_system_type (system_types.validation)
   │  resolves connectivity_factor once:
   │    explicit params['connectivity_factor']
   │      → ClusterAdsorbateConfig.structure_connectivity_factor
   │      → SurfaceSystemConfig.structure_connectivity_factor
   │      → default 1.4
   │    (float or per-element/pair dict; see Tunables)
   │
   ├── surface policy ──▶ validate_surface_config_slab_prefix
   │                     └─▶ validate_supported_cluster_deposit (surface/validation.py)
   │                            ├─ validate_cluster_structure  (clash)
   │                            ├─ penetration check
   │                            ├─ validate_adsorbate_fragment_integrity
   │                            └─ validate_connectivity_policy (surface)
   │                                  (connected subgroups, each touches slab)
   │
   ├── gas_cluster_adsorbate ──▶ validate_cluster_structure (clash)
   │                          └─ validate_connectivity_policy (gas; whole region connected)
   │
   └── bare gas_cluster ──────▶ validate_cluster_structure (clash)
                              └─ validate_connectivity_policy (gas; whole region connected)

   # bare `surface`: no per-minimum deposit/connectivity gate (slab-prefix
   # validation still runs per minimum).

Outcome toggles
---------------

The source of truth for the toggle defaults is :doc:`/parameters` and
:class:`~scgo.system_types.GLOptimizerParams`.

.. list-table::
   :widths: 38 10 30 32
   :header-rows: 1

   * - Toggle
     - Default
     - Gas (``gas_cluster_adsorbate``)
     - Surface (``surface_cluster_adsorbate``)
   * - ``allow_cluster_fragmentation``
     - ``False``
     - n/a (no slab)
     - Multiple core/mixed subgroups OK, each must touch slab
   * - ``allow_adsorbate_surface_detachment``
     - ``False``
     - n/a
     - Adsorbate-only subgroups on slab allowed
   * - ``enforce_adsorbate_subgraph_integrity``
     - ``True``
     - Fragments kept connected
     - Fragments kept connected
   * - ``freeze_adsorbate_internal_geometry``
     - ``False``
     - Rigid fragment (``FixBondLengths``)
     - Rigid fragment (``FixBondLengths``)
   * - ``validate_combined_structure`` (``ClusterAdsorbateConfig``)
     - ``True``
     - Pre/post-relax combined check
     - —

.. note::

   The first two toggles (``allow_cluster_fragmentation``,
   ``allow_adsorbate_surface_detachment``) are **surface-only**. Gas paths ignore
   them because there is no slab. Surface defaults require a single connected
   mobile component touching the slab.

Global-optimization validation flow
-----------------------------------

Every candidate minimum produced by the global optimizers is validated through
the shared helper
:func:`~scgo.system_types.validate_minimum_structure` (a thin wrapper over
:func:`~scgo.system_types.validate_structure_for_system_type`):

- ``basinhopping_go`` validates each relaxed trial through the shared helper.
  The *initial* relaxed seed is validated only softly: a failure is logged as a
  warning and the run proceeds with the seed as the starting structure (later
  moves and the final gate still reject disconnected minima).
- ``geneticalgorithm_go_torchsim`` validates each child pre-relax, and routes the
  GA storage gate (:func:`~scgo.algorithms.ga_common.validate_structure_for_ga_storage`)
  through the same helper.
- ``simple_go`` previously validated **nothing**; it now validates its single
  relaxed structure (when a ``system_type`` is given) and returns an empty list if
  it is invalid.

Because the validation rule set is identical, a fragmented or disconnected
structure is rejected the same way no matter which algorithm produced it. As a
final backstop, :func:`~scgo.minima_search.core.run_trials` applies the same
structural gate to the dedup'd unique candidates **after** global optimization:
any dedup'd candidate that fails
:func:`~scgo.system_types.validate_minimum_structure` is dropped before the
physical Hessian/vibration gate. Surface candidates are validated against the
*prepared* slab search config when ``slab_is_search_target`` is set, so the final
gate is exact rather than relying on stored ``n_slab_atoms`` tags. The final gate
honors the same connectivity-factor precedence as the algorithm and TS gates
(explicit ``connectivity_factor`` → ``ClusterAdsorbateConfig`` →
``SurfaceSystemConfig`` → ``1.4``), including float or per-element/pair dict
specs.

Minima pair selection
---------------------

Before NEB, endpoints are chosen by
:func:`~scgo.ts_search.transition_state_io.select_structure_pairs`. Hard gates
(``energy_gap_threshold``, ``max_endpoint_mismatch``, ``pair_core_rms_max``) and
soft ranking (``pair_score_*``) are documented under **Pair selection** in
:doc:`/parameters`. Defaults from
:func:`~scgo.pair_selection_defaults.pair_selection_param_defaults`.

``max_pairs`` is the NEB budget. Adsorbate searches may oversample the select
pool (then IDPP-re-rank) via
:func:`~scgo.ts_search.transition_state_io.resolve_ts_pair_select_cap`; bare
types — including surface presets that set ``max_endpoint_mismatch`` — do not.
See **Budget and oversampling** in :doc:`/parameters`.

NEB pre-screen gates
--------------------

Before any NEB optimization,
:func:`~scgo.ts_search.transition_state.validate_initial_neb_path` runs for
**every** system type (TorchSim and serial ASE paths).
:func:`~scgo.ts_search.transition_state.validate_initial_neb_energy_profile`
runs only when ``max_endpoint_mismatch`` is set (bare ``gas_cluster`` leaves it
``None``):

- ``validate_initial_neb_path`` always runs the interior-image clash check
  (min mobile pairwise distance vs ``neb_prescreen_clash_distance``); the
  aligned endpoint-displacement gate is additionally enabled when
  ``max_endpoint_mismatch`` is set.
- ``validate_initial_neb_energy_profile`` runs only when
  ``max_endpoint_mismatch`` is set (barrier cap ``neb_max_spurious_barrier``;
  drift + ``min_saddle_prominence`` checks when canonical endpoint energies are
  available).

Per-system-type defaults:

.. list-table::
   :widths: 30 22 22 22 22
   :header-rows: 1

   * - Preset
     - ``neb_prescreen_clash_distance``
     - ``min_saddle_prominence``
     - ``neb_max_spurious_barrier``
     - ``max_endpoint_mismatch``
   * - ``_GAS_TS_NEB_DEFAULTS`` (bare gas)
     - 1.0
     - 0.10
     - 8.0
     - ``None`` (clash always; energy-profile skipped)
   * - ``_SURFACE_TS_NEB_DEFAULTS`` (surface cluster)
     - 0.7
     - 0.40
     - 8.0
     - 1.25 (newly enabled — was unset)
   * - ``_GAS_ADSORBATE_TS_NEB_DEFAULTS``
     - 0.7 (override)
     - 0.40 (override)
     - 8.0
     - 1.25
   * - ``_SURFACE_ADSORBATE_TS_NEB_DEFAULTS``
     - 0.7 (override)
     - 0.40 (override)
     - 8.0
     - 1.5

.. note::

   The gas-adsorbate preset explicitly re-sets the three pre-screen knobs so the
   looser bare-gas values do not silently leak in through dictionary inheritance.
   The surface-adsorbate preset spreads ``_SURFACE_TS_NEB_DEFAULTS`` (which already
   carries ``0.7`` / ``0.40`` / ``8.0``), so its explicit re-set is a no-op.

Constraint model
----------------

Slab ``FixAtoms`` modes
~~~~~~~~~~~~~~~~~~~~~~~~~

Configured via :class:`~scgo.surface.config.SurfaceSystemConfig`:

- ``fix_all_slab_atoms=True`` (default),
- ``n_relax_top_slab_layers=N``, or
- ``n_fix_bottom_slab_layers=L-N``.

Applied by :func:`~scgo.surface.constraints.attach_slab_constraints`, which
computes layer indices via ``_layer_indices_by_clustering`` and calls
``_replace_slab_fixatoms``.

``_replace_slab_fixatoms`` strips any existing ``FixAtoms`` and re-appends,
**preserving non-``FixAtoms`` constraints** (for example, adsorbate
``FixBondLengths``).

Core freeze
~~~~~~~~~~~~

``fix_core`` in
:func:`~scgo.cluster_adsorbate.relax.relax_metal_cluster_with_adsorbate`:
when ``True``, it freezes core indices ``0..n_core-1`` via ``FixAtoms``
(gas-phase only; periodic cores keep their cell).

Adsorbate rigidity
~~~~~~~~~~~~~~~~~~~

``freeze_adsorbate_internal_geometry`` triggers
:func:`~scgo.cluster_adsorbate.constraints.attach_adsorbate_internal_geometry_constraints`,
which appends one multi-pair ``FixBondLengths`` per fragment. These
non-``FixAtoms`` constraints *survive* slab re-freezing because of
``_replace_slab_fixatoms``.

Tunables & floors
-----------------

The easiest way to read the thresholds is by what they control:

Placement and steric room
~~~~~~~~~~~~~~~~~~~~~~~~~

- ``height_min`` / ``height_max``: default ``0.9`` / ``2.2`` Å; placement
  height range in :mod:`scgo.cluster_adsorbate.config`.
- ``blmin_ratio``: default ``0.7``; clash threshold with a floor of ``0.55`` in
  ``_BLMIN_RATIO_FLOOR`` in :mod:`scgo.cluster_adsorbate.placement`.
- ``structure_min_distance_factor``: default ``0.4``; floor ``0.3`` in
  ``_MIN_DISTANCE_FACTOR_FLOOR`` in :mod:`scgo.cluster_adsorbate.placement`.

Connectivity and legal topology
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``connectivity_factor`` / ``structure_connectivity_factor``: default ``1.4``;
  defined in ``CONNECTIVITY_FACTOR`` in :mod:`scgo.initialization.initialization_config`.
- This is the main structural legality check for connectivity during
  initialization, after GA operators, at per-minimum algorithm gates, at the
  ``run_trials`` final structural gate, and in TS. All paths resolve the same
  precedence (explicit ``connectivity_factor`` →
  ``ClusterAdsorbateConfig.structure_connectivity_factor`` →
  ``SurfaceSystemConfig.structure_connectivity_factor`` → ``1.4``).
- The value may be a global float or a dict:

  - float ``f``: bonded if ``d <= (r_i + r_j) * f``
  - element dict: bonded if ``d <= r_i*f_i + r_j*f_j`` (missing symbols use ``1.4``)
  - pair keys ``"Pt-C"`` or ``("Pt", "C")``: bonded if ``d <= (r_i + r_j) * f_ij``;
    pair entries override element-derived thresholds

  Example (Pt on graphite): ``{"Pt": 1.4, "C": 1.4, "Pt-C": 1.8}``.

Surface contact and penetration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Penetration tolerance: ``0.1`` Å in
  ``_BINDING_PENETRATION_TOLERANCE_A`` in :mod:`scgo.surface.validation`.
- ``_H_CONTACT_THRESHOLD_A``: ``1.15`` Å, non-tunable, in
  :mod:`scgo.cluster_adsorbate.validation`. It separates weak H-bond-like
  contacts from newly formed covalent bonds between fragments.

Why the numbers are intentionally split
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The "loosen-placement / tighten-validate" design in
:mod:`scgo.initialization.initialization_config` is intentional: placement uses
a steric floor such as ``blmin_ratio=0.7``, while validation runs at
``connectivity_factor=1.4``. That gives the search enough room to explore, but
still rejects borderline disconnected or topologically illegal candidates before
they survive to the final candidate set.

MIC semantics
-------------

Gas always uses ``use_mic=False``. Surface derives it from
``SurfaceSystemConfig.comparator_use_mic``, resolved by
:func:`~scgo.system_types.resolve_structure_mic`; it returns ``False`` for
non-surface systems and raises if the system is surface-based but
``surface_config`` is ``None``.
