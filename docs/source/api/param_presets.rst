Parameter Presets
=================

Predefined parameter sets for common use cases.

Transition-state presets (:func:`~scgo.param_presets.get_ts_search_params`) set
``neb_align_endpoints=True`` for every system type. Surface types also default
``neb_interpolation_mic``, ``neb_surface_cell_remap``, and
``neb_surface_lattice_rotation`` to ``True`` so NEB bands start from
lattice-compatible aligned endpoints. See :doc:`/quickstart` (NEB endpoint alignment).

.. automodule:: scgo.param_presets
   :members:
   :undoc-members:
   :show-inheritance: