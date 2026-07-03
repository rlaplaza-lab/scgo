Quick Start
===========

SCGO helps you find the lowest-energy atomic structures using global optimization. This guide shows how to use all supported workflows.

System Types
------------

You must specify one of four system types:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Type
     - Use when
   * - ``gas_cluster``
     - Optimizing a cluster in vacuum
   * - ``surface_cluster``
     - Optimizing a cluster on a surface
   * - ``gas_cluster_adsorbate``
     - Optimizing a cluster with adsorbates in vacuum
   * - ``surface_cluster_adsorbate``
     - Optimizing a cluster with adsorbates on a surface

For surface types, you need a ``surface_config``. For adsorbate types, you need ``adsorbates``.

-----------
Gas Cluster
-----------

Optimize a simple cluster in vacuum.

**Fast test with EMT:**

.. code-block:: python

   from scgo import run_go
   from scgo.param_presets import get_testing_params

   results = run_go(
       ["Pt"] * 4,
       params=get_testing_params(),
       seed=42,
       system_type="gas_cluster",
   )

   for energy, atoms in results:
       print(f"Energy: {energy:.4f} eV, Formula: {atoms.get_chemical_formula()}")

For multi-element clusters (bimetallics, oxides), atom order follows the
composition list you pass in so GA crossover can pair structures safely.
Initialization favours placing heavier elements first while keeping diversity
across the population. Details: :doc:`/api/initialization`.

**Production run with MACE:**

.. code-block:: python

   from scgo import run_go
   from scgo.param_presets import get_default_params

   params = get_default_params()
   params["calculator_kwargs"]["model_name"] = "mace_mp_small"

   results = run_go(
       "Pt5",
       params=params,
       seed=42,
       system_type="gas_cluster",
   )

--------------
On a Surface
--------------

Use the built-in graphite surface or create your own slab.

**Using graphite preset:**

.. code-block:: python

   from scgo import run_go, make_graphite_surface_config
   from scgo.param_presets import get_default_params

   surface_config = make_graphite_surface_config(slab_layers=3)

   results = run_go(
       "Pt5",
       params=get_default_params(),
       seed=42,
       surface_config=surface_config,
       system_type="surface_cluster",
   )

**Using a custom slab:**

.. code-block:: python

   from ase.build import fcc111
   from scgo import run_go, make_surface_config
   from scgo.param_presets import get_default_params

   slab = fcc111("Pt", size=(3, 3, 3), vacuum=10.0)
   surface_config = make_surface_config(slab)

   results = run_go(
       "Pt4",
       params=get_default_params(),
       seed=42,
       surface_config=surface_config,
       system_type="surface_cluster",
   )

Defining Custom Surfaces
~~~~~~~~~~~~~~~~~~~~~~~~

Use ``SurfaceSystemConfig`` or the simpler ``make_surface_config()`` helper.

**Slab motion options:**

- ``fix_all_slab_atoms=True``: Entire slab stays frozen (default)
- ``n_relax_top_slab_layers=2``: Allow top 2 layers to relax
- ``n_fix_bottom_slab_layers=1``: Freeze bottom layer only
- Both layer counts = ``None``: Entire slab can relax

Do not use ``n_relax_top_slab_layers`` together with ``n_fix_bottom_slab_layers``.

**Full example:**

.. code-block:: python

   from ase.build import fcc111
   from scgo import run_go
   from scgo.param_presets import get_default_params
   from scgo.surface import SurfaceSystemConfig

   slab = fcc111("Fe", size=(4, 4, 3), vacuum=12.0)

   surface_config = SurfaceSystemConfig(
       slab=slab,
       adsorption_height_min=1.2,
       adsorption_height_max=2.5,
       fix_all_slab_atoms=False,
       n_relax_top_slab_layers=2,
   )

   results = run_go(
       "Pt5",
       params=get_default_params(),
       seed=42,
       surface_config=surface_config,
       system_type="surface_cluster",
   )

---------------
With Adsorbates
---------------

Add adsorbate molecules (OH, CO, etc.) to your cluster.

**Gas phase with adsorbate:**

.. code-block:: python

   from ase import Atoms
   from scgo import run_go
   from scgo.param_presets import get_default_params

   oh = Atoms("OH", positions=[[0, 0, 0], [0, 0, 0.96]])

   results = run_go(
       ["Pt"] * 5,
       params=get_default_params(),
       seed=42,
       system_type="gas_cluster_adsorbate",
       adsorbates=oh,
   )

**Multiple adsorbates:**

.. code-block:: python

   from ase import Atoms
   from scgo import run_go
   from scgo.param_presets import get_default_params

   oh1 = Atoms("OH", positions=[[0, 0, 0], [0, 0, 0.96]])
   oh2 = Atoms("OH", positions=[[0, 0, 0], [0, 0, 0.96]])

   results = run_go(
       "Pt5",
       params=get_default_params(),
       seed=42,
       system_type="gas_cluster_adsorbate",
       adsorbates=[oh1, oh2],
   )

Defining Custom Adsorbates
~~~~~~~~~~~~~~~~~~~~~~~~~~

Any ASE ``Atoms`` object is a valid adsorbate fragment. The GA will:

- Place fragments on cluster surface sites
- Keep fragments rigid (bonds stay intact)
- Optionally reposition fragments during optimization

**Tuning placement:**

.. code-block:: python

   from ase import Atoms
   from scgo import run_go
   from scgo.param_presets import get_default_params
   from scgo.cluster_adsorbate import ClusterAdsorbateConfig

   oh = Atoms("OH", positions=[[0, 0, 0], [0, 0, 0.96]])

   params = get_default_params()
   params["cluster_adsorbate_config"] = ClusterAdsorbateConfig(
       height_min=0.9,
       height_max=2.2,
       max_placement_attempts=200,
   )
   params["freeze_adsorbate_internal_geometry"] = True  # Keep fragment rigid

   results = run_go(
       "Pt5",
       params=params,
       seed=42,
       system_type="gas_cluster_adsorbate",
       adsorbates=oh,
   )

--------------------
Surface + Adsorbates
--------------------

Combine surface and adsorbates.

.. code-block:: python

   from ase import Atoms
   from scgo import run_go, make_graphite_surface_config
   from scgo.param_presets import get_default_params

   surface_config = make_graphite_surface_config(slab_layers=3)
   oh = Atoms("OH", positions=[[0, 0, 0], [0, 0, 0.96]])

   results = run_go(
       "Pt5",
       params=get_default_params(),
       seed=42,
       surface_config=surface_config,
       system_type="surface_cluster_adsorbate",
       adsorbates=oh,
   )

------------------
Transition States
------------------

Find transition states between optimized structures.

**GO + TS combined:**

.. code-block:: python

   from scgo import run_go_ts
   from scgo.param_presets import get_torchsim_ga_params, get_ts_search_params

   go_params = get_torchsim_ga_params(system_type="gas_cluster", seed=42)
   go_params["optimizer_params"]["ga"].update(niter=10, population_size=50)

   ts_params = get_ts_search_params(system_type="gas_cluster", seed=42)
   ts_params["max_pairs"] = 15

   summary = run_go_ts(
       "Pt5",
       go_params=go_params,
       ts_params=ts_params,
       seed=42,
       system_type="gas_cluster",
   )

**On a surface:**

.. code-block:: python

   from scgo import run_go_ts, make_graphite_surface_config
   from scgo.param_presets import get_torchsim_ga_params, get_ts_search_params

   surface_config = make_graphite_surface_config(slab_layers=3)

   go_params = get_torchsim_ga_params(
       system_type="surface_cluster",
       surface_config=surface_config,
       seed=42,
   )

   ts_params = get_ts_search_params(
       system_type="surface_cluster",
       surface_config=surface_config,
       seed=42,
   )
   ts_params["max_pairs"] = 10

   summary = run_go_ts(
       "Pt5",
       go_params=go_params,
       ts_params=ts_params,
       seed=42,
       surface_config=surface_config,
       system_type="surface_cluster",
   )

------------
Campaigns
------------

Run multiple compositions in one call.

.. code-block:: python

   from scgo import run_go_campaign
   from scgo.param_presets import get_testing_params
   from scgo.runner_api import build_one_element_compositions

   # Pt2, Pt3, Pt4, Pt5, Pt6
   compositions = build_one_element_compositions("Pt", min_atoms=2, max_atoms=6)

   results = run_go_campaign(
       compositions,
       params=get_testing_params(),
       seed=42,
       system_type="gas_cluster",
   )
   # results is dict[formula, list[(energy, Atoms)]]

Failed compositions (e.g. initialization ``ValueError`` on extreme
stoichiometries) are logged, recorded as empty lists in the returned dict, and
skipped so the rest of the campaign continues. See :doc:`/api/initialization`
for multi-element atom ordering and placement behaviour.

**Binary compositions:**

.. code-block:: python

   from scgo import run_go_campaign
   from scgo.param_presets import get_testing_params
   from scgo.runner_api import build_two_element_compositions

   # All Au/Pt combinations with 2-4 total atoms
   compositions = build_two_element_compositions("Au", "Pt", min_atoms=2, max_atoms=4)

   results = run_go_campaign(
       compositions,
       params=get_testing_params(),
       seed=42,
       system_type="gas_cluster",
   )

------------------
Output Files
------------------

All runs create a ``{formula}_searches/`` directory containing:

- ``run_<date>/trial_<N>/``: Individual optimization runs
  - ``ga_go.db`` or ``bh_go.db``: Database of candidate structures
- ``results_summary.json``: Summary of all minima found
- ``final_unique_minima/``: XYZ files of the best structures

Transition state runs add a ``ts_results_{formula}/`` folder with TS geometries.

----------
Parameters
----------

Quick parameter selection:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Preset
     - Use for
   * - ``get_testing_params()``
     - Fast tests (EMT calculator)
   * - ``get_default_params()``
     - Default production (MACE)
   * - ``get_torchsim_ga_params(...)``
     - GPU-accelerated with TorchSim
   * - ``get_ts_search_params(...)``
     - Transition state search

See :doc:`/parameters` for all options and :doc:`/api/param_presets` for details.

----------
Examples
----------

Working examples in the repository:

- ``examples/example_pt5_gas.py``: Pt5 in gas phase
- ``examples/example_pt5_graphite.py``: Pt5 on graphite
- ``examples/example_pt5_oh_gas.py``: Pt5 + OH in gas phase
- ``examples/example_pt5_2oh_graphite.py``: Pt5 + 2OH on graphite
