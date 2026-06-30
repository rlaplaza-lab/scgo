Parameter Presets
=================

Predefined parameter sets for common use cases.

Transition-state presets (:func:`~scgo.param_presets.get_ts_search_params`) set
``neb_align_endpoints=True`` for every system type. Surface types also default
``neb_interpolation_mic``, ``neb_surface_cell_remap``, and
``neb_surface_lattice_rotation`` to ``True``, and ``neb_surface_max_lattice_shift``
to ``1``, so NEB bands start from lattice-compatible aligned endpoints. See
:doc:`/parameters` (Transition State Search).

GO presets from :func:`~scgo.param_presets.get_default_params` and TS presets from
:func:`~scgo.param_presets.get_ts_search_params` also include
``allow_cluster_fragmentation`` and ``allow_adsorbate_surface_detachment`` (default
``False``), plus ``enforce_adsorbate_subgraph_integrity`` (default ``True``) to
reject dissociative adsorbate subgraphs (per fragment when fragment lengths are
available; otherwise on the full adsorbate block). ``freeze_adsorbate_internal_geometry``
defaults to ``False``; intra-fragment bonds are still preserved by tag-rigid GA
operators unless you opt into strict template restore. See :doc:`/api/system_types`
(surface mobile connectivity and GA operator partitioning).

.. automodule:: scgo.param_presets
   :members:
   :undoc-members:
   :show-inheritance: