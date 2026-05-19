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
