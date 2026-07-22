# SCGO examples

`run_go_ts` smoke scripts for all four system types (MACE + TorchSim; small
`niter` / `population_size` / `max_pairs` — scale up for production).

| Script | System type |
|--------|-------------|
| `example_pt5_gas.py` | `gas_cluster` |
| `example_pt5_oh_gas.py` | `gas_cluster_adsorbate` |
| `example_pt5_graphite.py` | `surface_cluster` |
| `example_pt5_2oh_graphite.py` | `surface_cluster_adsorbate` |

```bash
pip install -e ".[mace]"
python examples/example_pt5_gas.py
```

Outputs under `examples/results/{stem}_mace/` with sibling `*_searches/` and
`*_ts_results/` (timing JSON enabled). See the docs *On-disk layout* section.
