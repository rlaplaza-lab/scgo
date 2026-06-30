Parameter Reference
===================

This guide lists the settings available for SCGO workflows.

Global Optimization
-------------------

Set these via `optimizer_params['ga']` or `optimizer_params['bh']`.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Setting
     - Description
   * - ``niter``
     - Number of optimization iterations. Use ``"auto"`` for automatic scaling.
   * - ``population_size``
     - Number of structures in the GA population. Use ``"auto"`` for automatic scaling.
   * - ``fmax``
     - Convergence threshold for structure relaxation (eV/Å).
   * - ``mutation_probability``
     - Chance to perform a mutation in GA (0.0 to 1.0).

Transition State (TS) Search
----------------------------

Set these via `ts_params`.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Setting
     - Description
   * - ``max_pairs``
     - Max minima pairs to evaluate for transition states.
   * - ``neb_n_images``
     - Number of images in the NEB band.
   * - ``neb_steps``
     - Max optimization steps for NEB.
   * - ``neb_align_endpoints``
     - Align endpoints before NEB interpolation. (Default: ``True``)

System & Calculator
-------------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Setting
     - Description
   * - ``cluster_adsorbate_config``
     - Optional :class:`~scgo.cluster_adsorbate.config.ClusterAdsorbateConfig` in ``go_params`` for hierarchical adsorbate fragment placement. Controls height range, ``max_placement_attempts``, ``blmin_ratio`` (covalent-radius clash table via :func:`~scgo.initialization.atomic_radii.build_blmin`), and structure checks during placement. Placement ranks candidate hull sites by steric deficit and relaxes thresholds progressively on retry. Set on ``go_params`` only—not as a ``run_*`` keyword. For typical runs, ``connectivity_factor`` alone is enough.
   * - ``connectivity_factor``
     - Global structure connectivity threshold ``(r_i + r_j) * factor`` for validation and (when ``cluster_adsorbate_config`` is omitted) hierarchical adsorbate placement. Overrides ``ClusterAdsorbateConfig.structure_connectivity_factor`` when both are set. GA operator sterics use ``BLMIN_RATIO_DEFAULT`` (0.7) separately from this validation threshold (default 1.4).
   * - ``allow_cluster_fragmentation``
     - Allow the cluster to break into pieces during optimization.
   * - ``enforce_adsorbate_subgraph_integrity``
     - For ``*_adsorbate`` system types, require connected adsorbate subgraphs (default: ``True``). Uses per-fragment checks when ``adsorbate_fragment_lengths`` is provided; otherwise validates the full adsorbate block.
   * - ``freeze_adsorbate_internal_geometry``
     - For ``*_adsorbate`` GO runs, Kabsch-restore each fragment to its template after mutations and omit adsorbate-internal distortions (default: ``False``). With the default, intra-fragment bonds are still preserved by tag-rigid operators; enable this for strict template fidelity.
   * - ``calculator``
     - The model to use: ``"MACE"``, ``"UMA"``, or ``"EMT"``.
   * - ``calculator_kwargs``
     - Arguments for the calculator (e.g., ``model_name``).
