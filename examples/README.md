# SCGO examples

Runnable scripts demonstrating `run_go_ts` for the four supported system types.
Each script uses TorchSim GA presets with small iteration counts suitable for
smoke testing; scale `niter`, `population_size`, and `max_pairs` for production.

| Script | System type | Description |
|--------|-------------|-------------|
| `example_pt5_gas.py` | `gas_cluster` | Pt5 gas-phase cluster |
| `example_pt5_oh_gas.py` | `gas_cluster_adsorbate` | Pt5 + OH in gas phase |
| `example_pt5_graphite.py` | `surface_cluster` | Pt5 on graphite slab |
| `example_pt5_2oh_graphite.py` | `surface_cluster_adsorbate` | Pt5 + 2×OH on graphite |

## Requirements

Install with the MACE extra (examples use MACE + TorchSim):

```bash
pip install -e ".[mace]"
```

## Running

From the repository root:

```bash
python examples/example_pt5_gas.py
```

Output is written under `examples/results/{output_stem}_mace/` with sibling
`Pt5_searches/` and `Pt5_ts_results/` trees. See the docs *On-disk layout*
section for the full directory schema.

All examples enable `write_timing_json` in both `go_params` and `ts_params` so
per-run `timing.json` and campaign `go_ts_timing.json` files are produced.
