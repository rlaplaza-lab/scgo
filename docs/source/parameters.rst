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
   * - ``allow_cluster_fragmentation``
     - Allow the cluster to break into pieces during optimization.
   * - ``calculator``
     - The model to use: ``"MACE"``, ``"UMA"``, or ``"EMT"``.
   * - ``calculator_kwargs``
     - Arguments for the calculator (e.g., ``model_name``).
