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

2. **Run Optimization**: Execute the run function with your composition.

   .. code-block:: python

      from scgo import run_go_ts

      # Run combined GA and TS search
      results = run_go_ts(
          ["Pt"] * 5,
          go_params=go_params,
          seed=42,
          system_type="gas_cluster",
      )

For detailed customization, refer to the :doc:`Parameter Reference </parameters>`.
