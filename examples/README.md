# SCGO examples

`run_go_ts` smoke scripts for all four system types (MACE + TorchSim; small
`niter` / `population_size`). Each script builds params from
`get_torchsim_ga_params` / `get_ts_search_params` for its `system_type` and only
overrides campaign size (`niter`, `population_size`, `max_pairs`) plus a few
example-specific knobs (`connectivity_factor=1.8` and
`freeze_adsorbate_internal_geometry=True` on adsorbate runs). Full NEB defaults:
`docs/source/parameters.rst`.

| Script | System type | TS preset highlights |
|--------|-------------|----------------------|
| `example_pt5_gas.py` | `gas_cluster` | no climb, `neb_fmax=0.05`, 5 images, serial |
| `example_pt5_oh_gas.py` | `gas_cluster_adsorbate` | climb, `neb_fmax=0.20`, 7 images, parallel NEB |
| `example_pt5_graphite.py` | `surface_cluster` | no climb, `neb_fmax=0.1`, MIC + lattice rotation |
| `example_pt5_2oh_graphite.py` | `surface_cluster_adsorbate` | climb, `neb_fmax=0.25`, serial, no lattice rotation |

```bash
pip install -e ".[mace]"
python examples/example_pt5_gas.py
python examples/example_pt5_oh_gas.py
python examples/example_pt5_graphite.py
python examples/example_pt5_2oh_graphite.py
```

Each run creates a new datetime `run_*` under `examples/results/{stem}_mace/`
(`*_searches/` and `*_ts_results/`; timing JSON enabled). Reusing the same
`output_stem` can seed GO from prior DBs in that tree — use a fresh stem (or
delete the old tree) for a clean end-to-end check. See the docs *On-disk layout*
section.
