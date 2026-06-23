Quick Start
===========

SCGO provides several example scripts in the ``examples/`` directory that demonstrate different use cases. This guide explains how these examples work and how to adapt them for your own research.

Example Structure
-----------------

All examples follow a similar pattern:

1. Import necessary modules
2. Define system parameters (composition, seed, system type, etc.)
3. Build parameter dictionaries using presets
4. Customize parameters as needed
5. Call the appropriate ``run_*`` function

Basic Gas-Phase Cluster (example_pt5_gas.py)
---------------------------------------------

This example demonstrates global optimization and transition state search for a simple gas-phase Pt5 cluster.

**Key Features:**
- ``system_type="gas_cluster"`` — no slab, no adsorbates
- Uses ``run_go_ts`` for combined GO + TS pipeline
- TS preset keeps ``neb_align_endpoints=True`` (3D Kabsch alignment before NEB interpolation)
- Customizes GA parameters (niter, population_size)
- Limits TS search with max_pairs

**Code:**

.. literalinclude:: ../../examples/example_pt5_gas.py
   :language: python
   :linenos:

**How to Run:**

.. code-block:: bash

   cd examples
   python example_pt5_gas.py

**Output:**
- Results under ``examples/results/pt5_gas_mace/`` (calculator slug appended to ``output_stem``)
- Contains ``Pt5_searches/`` with GO minima and ``ts_results_Pt5/`` for TS results

Surface-Supported Cluster (example_pt5_graphite.py)
---------------------------------------------------

This example shows how to optimize a Pt5 cluster supported on a graphite surface.

**Key Features:**
- ``system_type="surface_cluster"`` — supported cluster with slab (no separate adsorbate fragment)
- Uses ``make_graphite_surface_config(slab_layers=...)`` for surface setup
- Passes the same ``surface_config`` into presets and ``run_go_ts``
- Includes surface-specific GA parameters (niter, population_size)

**Code:**

.. literalinclude:: ../../examples/example_pt5_graphite.py
   :language: python
   :linenos:

**Key Differences from Gas-Phase:**
- Requires ``surface_config`` in ``get_torchsim_ga_params`` / ``get_ts_search_params`` and on ``run_go_ts``
- Periodic slab in all relaxed structures
- TS presets enable MIC path interpolation and surface PBC endpoint alignment by default (see below)
- Output layout matches gas-phase but under a surface-specific output directory

Gas-Phase Cluster with Adsorbate (example_pt5_oh_gas.py)
---------------------------------------------------------

This example demonstrates a gas-phase Pt5 cluster with OH adsorbate.

**Key Features:**
- ``system_type="gas_cluster_adsorbate"`` — gas-phase with adsorbates
- Uses ASE Atoms object to define adsorbate fragment
- Passes core-only ``composition`` plus ``adsorbates`` to ``run_go_ts``
- Shows hierarchical adsorbate placement

**Adsorbate Setup:**

.. code-block:: python

   ADSORBATES = [Atoms(symbols=["O", "H"], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.96]])]

Surface-Supported Cluster with Multiple Adsorbates (example_pt5_2oh_graphite.py)
---------------------------------------------------------------------------------

This advanced example shows a surface-supported Pt5 cluster with two OH adsorbates.

**Key Features:**
- ``system_type="surface_cluster_adsorbate"`` — surface + adsorbate policies
- Combines ``make_graphite_surface_config`` with multiple ``adsorbates`` fragments
- Demonstrates advanced TS search parameter customization

**Advanced TS Parameters:**

.. code-block:: python

   ts_params["energy_gap_threshold"] = 1.0
   ts_params["neb_n_images"] = 7
   ts_params["neb_steps"] = 800

NEB endpoint alignment
----------------------

All examples use ``get_ts_search_params(...)``, which leaves **endpoint alignment on by default**
(``neb_align_endpoints=True``). SCGO aligns product to reactant **before** building the initial NEB
band; ASE ``NEB.interpolate`` then fills only the interior images.

**Gas-phase** (``gas_cluster``, ``gas_cluster_adsorbate``):

- Atom reordering (fingerprint matching; blockwise when ``adsorbates`` are passed).
- 3D Kabsch rigid fit on the mobile region.

**Surface / slab** (``surface_cluster``, ``surface_cluster_adsorbate``):

- Same reordering, plus slab-aware matching when ``adsorbates`` define core/adsorbate blocks.
- ``neb_interpolation_mic=True`` (enforced for surface system types).
- ``neb_surface_cell_remap`` and ``neb_surface_lattice_rotation`` (default ``True``): MIC-aware
  fingerprint matching, collective in-plane lattice-image selection for mobile atoms, per-atom
  MIC snapping, integer in-plane lattice translations (search span
  ``neb_surface_max_lattice_shift``, default ``1``), and **global** in-plane rotation evaluated
  jointly with each shift candidate. Fixed slab atoms remain anchored to the reactant frame.

Do not disable alignment unless you deliberately want raw GO minima as NEB endpoints:

.. code-block:: python

   # Rare: skip alignment (not recommended for production surface NEB)
   ts_params["neb_align_endpoints"] = False

Optional surface-only toggles (defaults are usually correct):

.. code-block:: python

   ts_params["neb_surface_cell_remap"] = True       # in-plane lattice-image search
   ts_params["neb_surface_lattice_rotation"] = True  # global in-plane Kabsch + MIC snap
   ts_params["neb_surface_max_lattice_shift"] = 2   # if minima differ by >1 cell in-plane

Surface mobile connectivity relaxation
--------------------------------------

Surface GO/TS presets include ``allow_cluster_fragmentation`` and
``allow_adsorbate_surface_detachment`` (both default ``False``). They control
:func:`~scgo.surface.validation.validate_supported_cluster_deposit` during GA/BH
and TS geometry gates:

- **Strict (default):** one connected mobile cluster bound to the slab.
- **Fragmentation only** (``allow_cluster_fragmentation=True``): multiple
  core/mixed subgroups; adsorbate-only fragments on the slab are still rejected.
- **Detachment** (``allow_adsorbate_surface_detachment=True``): one core/mixed
  subgroup plus optional adsorbate-only subgroups on the slab.
- **Both True:** any mobile split allowed if every subgroup touches the slab.

The removed ``allow_dissociative_adsorption`` flag maps to **both** new flags set
to ``True``. Pass the new keys in ``go_params`` / ``ts_params`` or per-optimizer
``optimizer_params`` entries.

For surface runs, TS pair selection and similarity use ``n_slab`` from
``surface_config`` so slab-only coordinate shifts do not count as distinct minima.

Multi-composition campaigns
---------------------------

For size scans over one or two elements, build composition lists and pass them to
``run_go_campaign`` (re-exported from ``scgo``) or
``_run_go_campaign_compositions``:

.. code-block:: python

   from scgo import run_go_campaign
   from scgo.param_presets import get_testing_params
   from scgo.runner_api import (
       build_one_element_compositions,
       build_two_element_compositions,
   )

   params = get_testing_params()
   compositions = build_one_element_compositions("Pt", min_atoms=2, max_atoms=6)
   # compositions = build_two_element_compositions("Au", "Pt", 2, 4)
   results = run_go_campaign(
       compositions,
       params=params,
       seed=42,
       system_type="gas_cluster",
   )

Custom ASE slabs (non-graphite)
-------------------------------

The graphite examples use ``make_graphite_surface_config``. For any other ASE slab,
normalize PBC and build a ``SurfaceSystemConfig`` with ``make_surface_config``:

.. code-block:: python

   from ase.build import fcc111
   from scgo.surface import make_surface_config

   slab = fcc111("Pt", size=(3, 3, 3), vacuum=10.0)
   surface_config = make_surface_config(slab)

Pass the same ``surface_config`` into ``get_torchsim_ga_params``,
``get_ts_search_params``, and ``run_go_ts`` / ``run_go``.

Creating Your Own Examples
--------------------------

To create your own SCGO workflow:

1. **Choose the right system type:**
   - ``gas_cluster`` — simple gas-phase clusters
   - ``surface_cluster`` — surface-supported clusters
   - ``gas_cluster_adsorbate`` — gas-phase with adsorbates
   - ``surface_cluster_adsorbate`` — surface-supported with adsorbates

2. **Start with presets:**

   .. code-block:: python

      from scgo.param_presets import get_torchsim_ga_params, get_ts_search_params

      go_params = get_torchsim_ga_params(system_type="gas_cluster", seed=42)
      ts_params = get_ts_search_params(system_type="gas_cluster", seed=42)

   For surface system types, pass ``surface_config`` to both preset builders and to
   ``run_go_ts`` (values must match).

3. **Customize parameters:**

   .. code-block:: python

      go_params["optimizer_params"]["ga"].update(niter=10, population_size=50)
      ts_params["max_pairs"] = 15

4. **Run the workflow:**

   .. code-block:: python

      from scgo import run_go_ts

      results = run_go_ts(
          ["Pt"] * 5,
          go_params=go_params,
          ts_params=ts_params,
          seed=42,
          system_type="gas_cluster",
      )

   For adsorbate modes, pass ``adsorbates=...`` (one ``Atoms`` or a list of fragments)
   with core-only ``composition``.

Parameter Customization Guide
-----------------------------

**Global Optimization Parameters:**
- ``niter`` — Number of GA iterations
- ``population_size`` — GA population size
- ``batch_size`` — Batch size for TorchSim GA relaxations
- ``calculator`` — ``"MACE"``, ``"UMA"``, or ``"EMT"`` (testing)

**Transition State Search Parameters:**
- ``max_pairs`` — Maximum number of minima pairs to search
- ``energy_gap_threshold`` — Energy threshold for pair selection
- ``neb_n_images`` — Number of NEB images
- ``neb_steps`` — Maximum NEB optimization steps
- ``neb_align_endpoints`` — Align endpoints before interpolation (default ``True``)
- ``neb_interpolation_mic`` — MIC during path interpolation (default ``True`` on surfaces)
- ``neb_surface_cell_remap`` — In-plane lattice-boundary remapping for slab endpoints (surface default ``True``)
- ``neb_surface_lattice_rotation`` — Lattice-compatible global in-plane rotation (surface default ``True``)
- ``neb_surface_max_lattice_shift`` — Maximum integer in-plane cell index searched during remap (default ``1``; increase when adsorbates hop multiple cells)
- ``neb_interpolation_method`` — ``"idpp"`` (default) or ``"linear"``
- ``neb_perturb_sigma`` — Optional Gaussian noise on interior images only (Å)

**Output Control:**
- ``output_root`` — Base output directory
- ``output_stem`` — Output directory name stem (calculator slug appended for ``run_go_ts``)
- ``seed`` — Random seed for reproducibility

Best Practices
--------------

1. **Start small:** Begin with small test systems (e.g., Pt4) before scaling up
2. **Use presets:** Always start with parameter presets, then customize
3. **Set seeds:** Use fixed seeds (``seed=42``) for reproducible results
4. **Monitor resources:** Surface systems and TS searches can be resource-intensive
5. **Keep alignment on:** Leave ``neb_align_endpoints=True`` unless debugging; surface runs rely on PBC-aware alignment for reasonable initial NEB paths
6. **Check outputs:** Examine the output directories to understand results structure; ``reactant_*.xyz`` / ``product_*.xyz`` reflect aligned endpoints used for the band

Understanding Output Structure
------------------------------

For ``run_go_ts`` with default layout (see also the project README):

.. code-block:: text

   {output_root}/{output_stem}_{mace|uma}/
   └── {formula}_searches/
       ├── run_<timestamp>/trial_<N>/
       │   └── ga_go.db
       ├── results_summary.json
       ├── final_unique_minima/
       └── ts_results_{formula}/
           ├── ts_<pair_id>.xyz
           ├── neb_<pair_id>_metadata.json
           └── ts_search_summary_{formula}.json

For ``run_go`` alone, the tree is the same ``{formula}_searches/`` folder without
``ts_results_*``.

Next Steps
----------

- Explore the :doc:`API Reference </api/runner_api>` for detailed function documentation
