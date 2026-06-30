Quick Start
===========

SCGO streamlines global optimization for atomic clusters using Basin Hopping (BH) and Genetic Algorithms (GA).

Setup for Arbitrary Systems
---------------------------

To optimize your own system:

1. **Initialize Parameters**: Start with a preset based on your target setup.

   .. code-block:: python

      from scgo.param_presets import get_torchsim_ga_params

      # For gas-phase clusters
      go_params = get_torchsim_ga_params(system_type="gas_cluster", seed=42)

      # For surface-supported clusters
      from scgo.surface import make_surface_config
      # Create your slab (e.g., via ASE) and initialize
      surface_config = make_surface_config(slab_atoms)
      go_params = get_torchsim_ga_params(
          system_type="surface_cluster",
          surface_config=surface_config,
          seed=42,
      )

2. **Run Optimization**: Build TS params and execute the run function with your composition.

   .. code-block:: python

      from scgo import run_go_ts
      from scgo.param_presets import get_ts_search_params

      ts_params = get_ts_search_params(system_type="gas_cluster", seed=42)

      results = run_go_ts(
          ["Pt"] * 5,
          go_params=go_params,
          ts_params=ts_params,
          seed=42,
          system_type="gas_cluster",
      )

Adsorbate workflows (gas or surface)
------------------------------------

For ``gas_cluster_adsorbate`` or ``surface_cluster_adsorbate``, pass core-only
``composition`` and ``adsorbates`` (one ASE ``Atoms`` fragment or a list).
SCGO builds hierarchical initial structures and runs a tag-aware GA that
preserves intra-fragment bonds by default.

.. code-block:: python

   from ase import Atoms
   from scgo import run_go_ts
   from scgo.param_presets import get_torchsim_ga_params, get_ts_search_params

   oh = Atoms("OH", positions=[[0, 0, 0], [0, 0, 0.96]])
   go_params = get_torchsim_ga_params(system_type="gas_cluster_adsorbate", seed=42)
   ts_params = get_ts_search_params(system_type="gas_cluster_adsorbate", seed=42)

   run_go_ts(
       ["Pt"] * 5,
       go_params=go_params,
       ts_params=ts_params,
       seed=42,
       system_type="gas_cluster_adsorbate",
       adsorbates=oh,
   )

See :doc:`/api/system_types` for GA operator partitioning and
``freeze_adsorbate_internal_geometry``. Full examples:
``examples/example_pt5_oh_gas.py`` and ``examples/example_pt5_2oh_graphite.py``.

For detailed customization, refer to the :doc:`Parameter Reference </parameters>`.
