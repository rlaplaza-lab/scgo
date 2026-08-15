# SCGO Examples

`run_go_ts` smoke scripts for the supported system types (MACE + TorchSim). Each
script builds params from `get_low_effort_torchsim_ga_params` /
`get_low_effort_ts_search_params` for its `system_type` and only overrides
`max_pairs` plus a few example-specific knobs. Those presets are the
reduced-budget (~25% of production) variants of `get_torchsim_ga_params` /
`get_ts_search_params`: the calculator, TorchSim relaxer and every NEB physics
knob are inherited unchanged, and only the GA (`niter`, `population_size`,
`niter_local_relaxation`) and NEB step budgets shrink — floored so bands still
converge. Surface system types clamp `niter_local_relaxation` back up to 400 at
run time, so slab searches keep production-strength local relaxation.

See ``docs/source/parameters.rst`` for merge / identity rules.

Adsorbate examples set `connectivity_factor=1.8` and
`freeze_adsorbate_internal_geometry=True`; the bare surface example also uses
`connectivity_factor=1.8` for slab validation. Adsorbate examples use fewer
`max_pairs` because IDPP screening is heavier and their bands run two-stage
climb over 7 images. Full NEB defaults: `docs/source/parameters.rst`.

The Kaggle GPU matrix in `tests/integration/test_gpu_examples_integration.py`
builds its params from the same two presets, so the CI cannot drift from these
scripts. To run a full-strength campaign instead, swap in
`get_torchsim_ga_params` / `get_ts_search_params`.

| Script | System type | `max_pairs` | `neb_steps` | TS preset highlights |
|--------|-------------|-------------|-------------|----------------------|
| `example_pt5_gas.py` | `gas_cluster` | 6 | 1000 | no climb, shared `neb_fmax=0.20`, 5 images, parallel NEB |
| `example_pt5_oh_gas.py` | `gas_cluster_adsorbate` | 6 | 1000 | climb, shared `neb_fmax=0.20`, 7 images, parallel NEB, `max_endpoint_mismatch=1.25` Å |
| `example_pt5_graphite.py` | `surface_cluster` | 6 | 1000 | no climb, shared `neb_fmax=0.20`, MIC + lattice rotation, parallel NEB |
| `example_pt5_2oh_graphite.py` | `surface_cluster_adsorbate` | 4 | 1000 | climb, shared `neb_fmax=0.20`, parallel NEB, no lattice rotation, `max_endpoint_mismatch=1.5` Å |
| `example_defected_graphite.py` | `surface` | 4 | 1000 | top-layer slab search on vacancy-defected graphite |
| `example_n_doped_graphite.py` | `surface_adsorbate` | 4 | 1000 | top-layer + OH on N-doped graphite |

All four graphite scripts use `slab_layers=3`, `slab_repeat_xy=3` (7.38 Å cell,
53–54 atoms).

```bash
pip install -e ".[mace]"
python examples/example_pt5_gas.py
python examples/example_pt5_oh_gas.py
python examples/example_pt5_graphite.py
python examples/example_pt5_2oh_graphite.py
python examples/example_defected_graphite.py
python examples/example_n_doped_graphite.py
```

Each run creates a new datetime `run_*` under `examples/results/{stem}_mace/`
(`{path_key}_searches/` and `{path_key}_ts_results/`; timing JSON enabled).
Path keys are component-aware, e.g. `Pt5`, `Pt5_OH`, `Pt5_graphite`,
`Pt5_OH_OH_graphite`, `defected_graphite`, `OH_n_doped_graphite`. Reusing the same
`output_stem` can seed GO from prior DBs in that tree — use a fresh stem (or
delete the old tree) for a clean end-to-end check. Override the stem without
editing the script via `SCGO_EXAMPLE_OUTPUT_STEM=my_fresh_stem`. See the docs
*On-disk layout* section.
