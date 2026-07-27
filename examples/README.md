# SCGO examples

`run_go_ts` smoke scripts for all four system types (MACE + TorchSim; small
`niter` / `population_size`). Each script builds params from
`get_torchsim_ga_params` / `get_ts_search_params` for its `system_type` and only
overrides campaign size (`niter`, `population_size`, `max_pairs`) plus a few
example-specific knobs. Adsorbate examples set
`connectivity_factor=1.8` and `freeze_adsorbate_internal_geometry=True`; the bare
surface example also uses `connectivity_factor=1.8` for slab validation.
Adsorbate examples use fewer `max_pairs` because IDPP screening is heavier.
Full NEB defaults: `docs/source/parameters.rst`.

| Script | System type | `max_pairs` | TS preset highlights |
|--------|-------------|-------------|----------------------|
| `example_pt5_gas.py` | `gas_cluster` | 15 | no climb, shared `neb_fmax=0.20`, 5 images, parallel NEB |
| `example_pt5_oh_gas.py` | `gas_cluster_adsorbate` | 12 | climb, shared `neb_fmax=0.20`, 7 images, parallel NEB, `max_endpoint_mismatch=1.25` Å |
| `example_pt5_graphite.py` | `surface_cluster` | 10 | no climb, shared `neb_fmax=0.20`, MIC + lattice rotation, parallel NEB |
| `example_pt5_2oh_graphite.py` | `surface_cluster_adsorbate` | 6 | climb, shared `neb_fmax=0.20`, parallel NEB, no lattice rotation, `max_endpoint_mismatch=1.5` Å |

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
delete the old tree) for a clean end-to-end check. Override the stem without
editing the script via `SCGO_EXAMPLE_OUTPUT_STEM=my_fresh_stem`. See the docs
*On-disk layout* section.
