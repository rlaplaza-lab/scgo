Surface workflows
=================

Slab configuration, deposition, and validation for supported-cluster runs.

Deposition and adsorbate initialization
---------------------------------------

:func:`~scgo.surface.deposition.create_deposited_cluster` builds initial
structures for ``surface_cluster`` and ``surface_cluster_adsorbate``:

- **Plain cluster** — gas-phase seed via :mod:`scgo.initialization` (``init_mode``
  on :class:`~scgo.surface.config.SurfaceSystemConfig`), then rotate/translate
  above the slab with covalent-radius connectivity height heuristics.
- **Cluster + adsorbate** — hierarchical core + fragment placement (hull sites,
  ranked steric candidates), then deposit with surface-biased rotation.
- **Adsorbate-only mobile region** (empty ``core_symbols``) — fragments placed
  directly on slab top-layer hull sites via
  :func:`~scgo.cluster_adsorbate.placement.place_fragment_on_cluster`.

.. autoclass:: scgo.surface.config.SurfaceSystemConfig
   :members:
   :show-inheritance:

.. autofunction:: scgo.surface.make_surface_config
.. autofunction:: scgo.surface.make_graphite_surface_config
.. autofunction:: scgo.surface.describe_surface_config
.. autofunction:: scgo.surface.create_deposited_cluster
.. autofunction:: scgo.surface.create_deposited_cluster_batch
.. autofunction:: scgo.surface.adsorption_energy

Composition builders for campaigns live in :mod:`scgo.runner_api`
(``build_one_element_compositions``, ``build_two_element_compositions``);
high-level runners are in :doc:`/api/runner_api`.
