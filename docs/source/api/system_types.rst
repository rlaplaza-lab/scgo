System Types
============

System type definitions and validation.

.. automodule:: scgo.system_types
   :members:
   :undoc-members:
   :show-inheritance:

Available System Types
----------------------

SCGO supports four explicit system types (``SystemType`` is a ``Literal`` alias):

1. **gas_cluster**: Gas-phase cluster (no slab, no adsorbates)
2. **surface_cluster**: Cluster supported on a slab (``surface_config`` required)
3. **gas_cluster_adsorbate**: Gas-phase cluster with adsorbates
4. **surface_cluster_adsorbate**: Supported cluster with adsorbates
   (``surface_config`` required)

See :class:`~scgo.system_types.SystemPolicy` and
:class:`~scgo.system_types.AdsorbateDefinition` in the module reference above.

Preset dicts returned by :func:`~scgo.param_presets.get_default_params` and
:func:`~scgo.param_presets.get_ts_search_params` are typed as
:class:`~scgo.system_types.GLOptimizerParams` and
:class:`~scgo.system_types.TSParams` respectively (still plain dicts at runtime).

NEB policy flags (via :class:`~scgo.system_types.SystemPolicy`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each system type sets defaults consumed by
:func:`~scgo.param_presets.get_ts_search_params`:

- ``neb_force_mic`` — surface types use minimum-image path interpolation.
- ``neb_disable_alignment`` — when ``False`` (default),
  ``neb_align_endpoints`` stays on in presets.
- ``neb_surface_cell_remap`` / ``neb_surface_lattice_rotation`` — enabled for
  ``surface_cluster`` and ``surface_cluster_adsorbate``; use lattice-compatible
  in-plane shifts and global rotation before NEB interpolation (not
  independent mobile-only rotations).
- The remap search span is controlled at runtime by
  ``neb_surface_max_lattice_shift`` in TS presets (default ``1`` cell in each
  in-plane direction).

Surface mobile connectivity
~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~scgo.system_types.validate_structure_for_system_type` delegates slab checks to
:func:`~scgo.surface.validation.validate_supported_cluster_deposit`. Two runtime flags
(default ``False`` in GO and TS presets):

- ``allow_cluster_fragmentation`` — multiple disconnected core/mixed mobile subgroups.
- ``allow_adsorbate_surface_detachment`` — adsorbate-only mobile subgroups on the slab
  without cluster contact (with exactly one core/mixed subgroup when
  fragmentation is off).

For ``*_adsorbate`` types, ``n_core_mobile`` is inferred from
``adsorbate_definition['core_symbols']``. The former ``allow_dissociative_adsorption``
parameter is removed; set both new flags to ``True`` for the old permissive behavior.

Adsorbate subgraph integrity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~

For ``*_adsorbate`` GO runs, optional placement and validation knobs live in
``go_params`` only:

- ``connectivity_factor`` — primary threshold for structure validation (and
  fallback for hierarchical placement when no config is set).
- ``cluster_adsorbate_config`` — optional
  :class:`~scgo.cluster_adsorbate.config.ClusterAdsorbateConfig` (fragment
  height range, ``max_placement_attempts``, ``blmin_ratio``, clash/connectivity
  checks). Fragment placement samples convex-hull vertex/edge/facet sites,
  ranks candidates by steric deficit, and relaxes placement thresholds on retry.
  Prefer ``connectivity_factor`` alone unless you need placement-specific overrides.
- ``freeze_adsorbate_internal_geometry`` — Kabsch-restore fragments after
  mutations (strict template mode). Default ``False`` still preserves
  intra-fragment bonds via tag-rigid GA operators.

GA operators for adsorbate types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When ``SystemPolicy.has_adsorbate`` is true, the GA partitions the mobile region
with ASE tags (core = ``0``, each fragment = ``1..N``):

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
