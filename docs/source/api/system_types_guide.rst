System Types Guide
==================

SCGO builds every workflow from three pieces: a **cluster** (also called the
**core** when molecules are present), an **adsorbate**, and a **slab**. Use this
guide to choose a system type and set the right surface, adsorbate, and
validation options. The full module reference lives at
:doc:`/api/system_types`.

Available system types
----------------------

Pass ``system_type`` as a run argument (not inside a preset dict, and not inside
``optimizer_params`` slots). SCGO supports six system types:

1. **gas_cluster**: cluster in vacuum (no slab, no adsorbates).
2. **surface_cluster**: cluster on a slab. Set ``surface_config``.
3. **gas_cluster_adsorbate**: cluster plus adsorbates in vacuum.
4. **surface_cluster_adsorbate**: cluster plus adsorbates on a slab. Set
   ``surface_config``.
5. **surface**: top layers of a bare slab as the search target. Set
   ``surface_config`` with ``fix_all_slab_atoms=False`` and a top/bottom layer
   policy.
6. **surface_adsorbate**: top slab layers plus adsorbate fragments (no cluster
   core). Set ``surface_config`` and ``adsorbates``.

Preset dicts returned by :func:`~scgo.param_presets.get_default_params` and
:func:`~scgo.param_presets.get_ts_search_params` are plain ``dict[str, Any]``
values (see :class:`~scgo.system_types.GLOptimizerParams` for the key layout).
For the policy objects themselves, see
:class:`~scgo.system_types.SystemPolicy` and
:class:`~scgo.system_types.AdsorbateDefinition` in the module reference.

Surface mobile connectivity
----------------------------

:func:`~scgo.system_types.validate_structure_for_system_type` checks supported
cluster deposits through
:func:`~scgo.surface.validation.validate_supported_cluster_deposit`. Two runtime
flags (default ``False`` in GO and TS presets) control mobile-region
fragmentation:

- ``allow_cluster_fragmentation``: allow multiple disconnected core/mixed
  mobile subgroups.
- ``allow_adsorbate_surface_detachment``: allow adsorbate-only mobile subgroups
  on the slab without cluster contact (with exactly one core/mixed subgroup when
  fragmentation is off).

For ``*_adsorbate`` types, ``n_core_mobile`` is inferred from
``adsorbate_definition['core_symbols']``.

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

Adsorbate placement tuning
--------------------------

For ``*_adsorbate`` GO runs, the placement and validation knobs live in
``go_params`` only:

- ``connectivity_factor``: primary connectivity spec for all structural gates
  (initialization, post-operator checks, per-minimum algorithm gates, the
  ``run_trials`` final gate, and TS). Accepts a float or a per-element/pair
  dict; see :doc:`/validation_and_constraints`.
- ``cluster_adsorbate_config``: optional
  :class:`~scgo.cluster_adsorbate.config.ClusterAdsorbateConfig` (fragment height
  range, ``max_placement_attempts``, ``blmin_ratio``, clash/connectivity checks).
  Prefer ``connectivity_factor`` alone unless you need placement-specific
  overrides.
- ``freeze_adsorbate_internal_geometry``: keep fragments rigid after mutations
  (strict template mode). Default ``False`` still preserves intra-fragment bonds
  via tag-rigid GA operators.

Defaults you rarely change
--------------------------

NEB path defaults come from each system type's
:class:`~scgo.system_types.SystemPolicy` and are consumed by
:func:`~scgo.param_presets.get_ts_search_params`:

- Surface types use minimum-image path interpolation
  (``neb_force_mic``).
- Endpoint alignment stays on by default
  (``neb_disable_alignment=False``).
- Surface types enable lattice-image remapping before NEB
  (``neb_surface_cell_remap``).
- Bare ``surface_cluster`` and ``surface`` enable in-plane lattice rotation;
  adsorbate surface types disable it so fragment-slab registry stays intact.
- Remap search span is set by ``neb_surface_max_lattice_shift`` in TS presets
  (default ``1`` cell in each in-plane direction).

When a system has adsorbates, the GA partitions the mobile region with ASE tags
(core = ``0``, each fragment = ``1..N``):

- Crossover splices the core only; adsorbate fragments inherit from parent 0.
- Mutations keep intra-fragment geometry unchanged.
- Rotational, mirror, flattening, breathing, and in-plane slide use core-only
  or adsorbate-scoped variants. Untagged gas-phase clusters omit ``mirror``.
- ``fragment_reposition`` re-places one adsorbate on fresh surface sites.

Operator clash checks use
:func:`~scgo.initialization.atomic_radii.build_blmin`
(``BLMIN_RATIO_DEFAULT = 0.7``). Post-operator validation uses the shared
``connectivity_factor`` spec via
:func:`~scgo.system_types.validate_structure_for_system_type`.

Full module reference: :doc:`/api/system_types`.
