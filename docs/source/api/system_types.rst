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
4. **surface_cluster_adsorbate**: Supported cluster with adsorbates (``surface_config`` required)

See :class:`~scgo.system_types.SystemPolicy` and :class:`~scgo.system_types.AdsorbateDefinition` in the module reference above.

NEB policy flags (via :class:`~scgo.system_types.SystemPolicy`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each system type sets defaults consumed by :func:`~scgo.param_presets.get_ts_search_params`:

- ``neb_force_mic`` — surface types use minimum-image path interpolation.
- ``neb_disable_alignment`` — when ``False`` (default), ``neb_align_endpoints`` stays on in presets.
- ``neb_surface_cell_remap`` / ``neb_surface_lattice_rotation`` — enabled for ``surface_cluster`` and ``surface_cluster_adsorbate``; use lattice-compatible in-plane shifts and global rotation before NEB interpolation (not independent mobile-only rotations). The remap search span is controlled at runtime by ``neb_surface_max_lattice_shift`` in TS presets (default ``1`` cell in each in-plane direction).

Surface mobile connectivity
~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~scgo.system_types.validate_structure_for_system_type` delegates slab checks to
:func:`~scgo.surface.validation.validate_supported_cluster_deposit`. Two runtime flags
(default ``False`` in GO and TS presets):

- ``allow_cluster_fragmentation`` — multiple disconnected core/mixed mobile subgroups.
- ``allow_adsorbate_surface_detachment`` — adsorbate-only mobile subgroups on the slab
  without cluster contact (with exactly one core/mixed subgroup when fragmentation is off).

For ``*_adsorbate`` types, ``n_core_mobile`` is inferred from
``adsorbate_definition['core_symbols']``. The former ``allow_dissociative_adsorption``
parameter is removed; set both new flags to ``True`` for the old permissive behavior.
