Cluster adsorbate placement
===========================

Hierarchical core + rigid fragment initialization and GA repositioning.

Configuration
-------------

.. autoclass:: scgo.cluster_adsorbate.config.ClusterAdsorbateConfig
   :members:
   :show-inheritance:

Placement and hierarchical builds
---------------------------------

.. autofunction:: scgo.cluster_adsorbate.placement.place_fragment_on_cluster
.. autofunction:: scgo.cluster_adsorbate.placement.radii_derived_height_bounds
.. autofunction:: scgo.cluster_adsorbate.hierarchical.build_hierarchical_core_fragment_cluster
.. autofunction:: scgo.cluster_adsorbate.hierarchical.build_adsorbate_only_cluster

Pre-GO feasibility checks
-------------------------

.. autofunction:: scgo.cluster_adsorbate.feasibility.validate_adsorbate_placement_feasibility
.. autofunction:: scgo.cluster_adsorbate.feasibility.count_adsorption_site_candidates
.. autofunction:: scgo.cluster_adsorbate.feasibility.estimate_fragment_footprint_radius

GA repositioning and rigid geometry
-------------------------------------

.. autoclass:: scgo.cluster_adsorbate.reposition.FragmentRepositionMutation
   :members:

.. autofunction:: scgo.cluster_adsorbate.rigid.enforce_frozen_adsorbate_geometry
.. autofunction:: scgo.cluster_adsorbate.rigid.restore_rigid_adsorbate_fragments

Radii and steric scoring
------------------------

Operator clash checks and placement ranking use covalent-radius ``blmin`` tables
(``BLMIN_RATIO_DEFAULT = 0.7``). Structure validation uses ``connectivity_factor``
(typically 1.4) via :func:`~scgo.system_types.validate_structure_for_system_type`.

.. autofunction:: scgo.initialization.atomic_radii.build_blmin
.. autofunction:: scgo.initialization.atomic_radii.build_blmin_from_zs
.. autofunction:: scgo.initialization.steric_scoring.steric_deficit
.. autofunction:: scgo.initialization.steric_scoring.steric_deficit_two_sets
