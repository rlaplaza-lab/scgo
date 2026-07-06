# SCGO test suite

## Marker policy

| Marker | Meaning | CI |
|--------|---------|-----|
| `slow` | Real optimizers, NEB, or heavy placement loops | `slow and not benchmark` |
| `integration` | Full workflow (GO campaigns, output trees) | Excluded from fast job (`not integration`) |
| `benchmark` | Long MLIP regression (Cu₄ MACE E2E) | Excluded from CI |
| `requires_cuda` | Needs GPU | Skipped on CPU runners |
| `requires_multicore` | Needs ≥2 CPUs | Skipped on single-core |

Fast CI (every PR): `pytest tests/ -m "not slow and not integration"`

Slow CI (every PR): `pytest tests/ -m "slow and not benchmark"` with `SCGO_BATCH_TEST_SAMPLES=15`

## Physics helpers

Shared assertions live in [`test_utils.py`](test_utils.py):

- `assert_ts_result_valid` — interior TS image, barrier band, endpoint ordering
- `assert_nn_distances_in_band` — covalent-radii-scaled NN distances
- `assert_adsorption_height_in_bounds` — slab adsorption height window
- `assert_pt_o_distance_reasonable` — Pt–O bond sanity

Constants in [`constants.py`](constants.py) (`EMT_PT2_BOND_ANG`, `EMT_H2_BARRIER_EV`, etc.).

## Local runs

```bash
# Fast subset (matches CI)
pytest tests/ -m "not slow and not integration"

# Slow subset
SCGO_BATCH_TEST_SAMPLES=15 pytest tests/ -m "slow and not benchmark"

# New physics reference tests only
pytest tests/physics/test_reference_emt.py -v
```

Install dev extras: `pip install -e ".[mace,dev]"` or `pip install -e ".[uma,dev]"`.
