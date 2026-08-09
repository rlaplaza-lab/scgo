System Types Guide
==================

SCGO defines six system types plus the policies, adsorbate definitions, and
validation rules that govern each one. Use this guide to choose a system type
and set the right surface, adsorbate, and validation options. The full module
reference (policies, ``AdsorbateDefinition``, ``validate_structure_for_system_type``)
lives at :doc:`/api/system_types`.

Available system types
----------------------

Pass ``system_type`` as a run argument (not inside a preset dict). SCGO supports
six system types:

1. **gas_cluster**: gas-phase cluster (no slab, no adsorbates).
2. **surface_cluster**: cluster supported on a slab. Set ``surface_config``.
3. **gas_cluster_adsorbate**: gas-phase cluster with adsorbates.
4. **surface_cluster_adsorbate**: supported cluster with adsorbates. Set
   ``surface_config``.
5. **surface**: bare slab as the GA/BH search target. Top layers are mobile; set
   ``surface_config`` with ``fix_all_slab_atoms=False`` and a top/bottom layer
   policy.
6. **surface_adsorbate**: top slab layers plus adsorbate fragments as the search
   target (no cluster core). Set ``surface_config`` and ``adsorbates``.

Preset dicts returned by :func:`~scgo.param_presets.get_default_params` and
:func:`~scgo.param_presets.get_ts_search_params` are typed as
:class:`~scgo.system_types.GLOptimizerParams` and
:class:`~scgo.system_types.TSParams` respectively (still plain dicts at runtime).
For the policy objects themselves, see
:class:`~scgo.system_types.SystemPolicy` and
:class:`~scgo.system_types.AdsorbateDefinition` in the module reference.

NEB policy flags
----------------

Each system type sets NEB defaults consumed by
:func:`~scgo.param_presets.get_ts_search_params`. The relevant flags on
:class:`~scgo.system_types.SystemPolicy` are:

- ``neb_force_mic`` — surface types use minimum-image path interpolation.
- ``neb_disable_alignment`` — when ``False`` (default),
  ``neb_align_endpoints`` stays on in presets.
- ``neb_surface_cell_remap`` — enabled for surface types (lattice-image
  selection / MIC snap before NEB interpolation).
- ``neb_surface_lattice_rotation`` — enabled for bare ``surface_cluster`` and
  ``surface``; disabled for ``surface_cluster_adsorbate`` and
  ``surface_adsorbate`` (free in-plane Kabsch breaks adsorbate–slab registry).
- The remap search span is controlled at runtime by
  ``neb_surface_max_lattice_shift`` in TS presets (default ``1`` cell in each
  in-plane direction).

Surface mobile connectivity
----------------------------

:func:`~scgo.system_types.validate_structure_for_system_type` delegates slab
checks to :func:`~scgo.surface.validation.validate_supported_cluster_deposit`.
Two runtime flags (default ``False`` in GO and TS presets) control mobile-region
fragmentation:

- ``allow_cluster_fragmentation`` — allow multiple disconnected core/mixed
  mobile subgroups.
- ``allow_adsorbate_surface_detachment`` — allow adsorbate-only mobile subgroups
  on the slab without cluster contact (with exactly one core/mixed subgroup when
  fragmentation is off).

For ``*_adsorbate`` types, ``n_core_mobile`` is inferred from
``adsorbate_definition['core_symbols']``. The former ``allow_dissociative_adsorption``
parameter is removed; set both flags above to ``True`` for the old permissive
behavior.

Adsorbate subgraph integrity
----------------------------

When ``enforce_adsorbate_subgraph_integrity=True`` (default in GO/TS presets),
SCGO rejects disconnected adsorbate subgraphs.

- With runner-style ``adsorbates=Atoms | list[Atoms]`` input, each input fragment
  must be connected and SCGO stores fragment boundaries for per-fragment checks.
- For manual ``adsorbate_definition`` input, ``adsorbate_fragment_lengths`` is
  optional:

  - when provided, integrity is enforced per fragment;
  - when omitted, integrity is enforced on the full adsorbate block as one
    connected subgraph.

This design supports non-linear molecules without requiring a rigid
``fragment_bond_axis`` contract.

Adsorbate placement tuning
--------------------------

For ``*_adsorbate`` GO runs, the placement and validation knobs live in
``go_params`` only:

- ``connectivity_factor`` — primary threshold for structure validation (and the
  fallback for hierarchical placement when no config is set).
- ``cluster_adsorbate_config`` — optional
  :class:`~scgo.cluster_adsorbate.config.ClusterAdsorbateConfig` (fragment height
  range, ``max_placement_attempts``, ``blmin_ratio``, clash/connectivity checks).
  Placement samples convex-hull vertex/edge/facet sites, ranks candidates by
  steric deficit, and relaxes placement thresholds on retry. Prefer
  ``connectivity_factor`` alone unless you need placement-specific overrides.
- ``freeze_adsorbate_internal_geometry`` — Kabsch-restore fragments after
  mutations (strict template mode). Default ``False`` still preserves
  intra-fragment bonds via tag-rigid GA operators.

GA operators for adsorbate types
--------------------------------

When :class:`~scgo.system_types.SystemPolicy` ``has_adsorbate`` is true, the GA
partitions the mobile region with ASE tags (core = ``0``, each fragment =
``1..N``):

- **Crossover** splices the core only; adsorbate fragments inherit from parent 0.
- **Mutations** use tag-rigid displacements for rattle, anisotropic rattle, and
  overlap relief so intra-fragment geometry is unchanged.
- **Rotational / mirror / flattening / breathing / in-plane slide** use
  core-only or adsorbate-scoped variants (``*_core``, ``*_ads``).
- **``fragment_reposition``** re-places one adsorbate on fresh hull sites using
  the same placement engine as initialization.

Operator clash checks use :func:`~scgo.initialization.atomic_radii.build_blmin`
(``BLMIN_RATIO_DEFAULT = 0.7``). Post-operator validation uses
``connectivity_factor`` (typically 1.4) via
:func:`~scgo.system_types.validate_structure_for_system_type`.

Full module reference (policies, ``AdsorbateDefinition``,
``validate_structure_for_system_type``): :doc:`/api/system_types`.
