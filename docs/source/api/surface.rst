Surface workflows
=================

Slab configuration, deposition, and validation for supported-cluster runs.

.. autoclass:: scgo.surface.config.SurfaceSystemConfig
   :members:
   :show-inheritance:

.. autofunction:: scgo.surface.make_surface_config
.. autofunction:: scgo.surface.make_graphite_surface_config
.. autofunction:: scgo.surface.describe_surface_config

Composition builders for campaigns live in :mod:`scgo.runner_api` (``build_one_element_compositions``,
``build_two_element_compositions``); high-level runners are in :doc:`/api/runner_api`.
