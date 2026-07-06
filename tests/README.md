# SCGO test suite

## Marker policy

| Marker | Meaning | CI |
|--------|---------|-----|
| `slow` | Real optimizers, NEB, or heavy placement loops | `slow and not benchmark` |
| `integration` | Full workflow (GO campaigns, output trees) | Excluded from fast job (`not integration`) |
| `benchmark` | Long MLIP regression (Cu₄ MACE E2E) | Excluded from CI |
| `requires_cuda` | Needs GPU | Skipped on CPU runners |
| `requires_mace` | Needs MACE extra | Excluded from UMA CI jobs |
| `requires_multicore` | Needs ≥2 CPUs | Skipped on single-core |

Fast CI (every PR): `pytest tests/ -m "not slow and not integration"`

Slow CI (every PR): `pytest tests/ -m "slow and not benchmark"` with `SCGO_BATCH_TEST_SAMPLES=15`

## Physics helpers

Shared assertions live in [`test_utils.py`](test_utils.py):

- `assert_ts_result_valid` — interior TS image, barrier band, endpoint ordering
- `assert_nn_distances_in_band` — covalent-radii-scaled NN distances
- `assert_deposition_height_in_bounds` — **placement-stage** height window from `SurfaceSystemConfig` (initial deposition only; not valid after GA/NEB)
- `assert_supported_cluster_binding` — **post-relaxation** slab contact, no burial, connectivity, fragment integrity
- `assert_pt_o_distance_reasonable` — Pt–O bond sanity

Constants in [`constants.py`](constants.py) (`EMT_PT2_BOND_ANG`, `PT4_EMT_BARRIER_EV`, etc.).

### Surface height checks (placement vs relaxation)

`adsorption_height_min/max` constrain the **deposition sampler** in
`create_deposited_cluster`: how far the cluster bottom is placed above the
slab top. After GA or NEB, atoms may move outside that window while still
being chemisorbed. Tests must use:

- `assert_deposition_height_in_bounds` — mock-relaxer / fresh placement smoke tests
- `assert_supported_cluster_binding` — real EMT relaxation and end-to-end GO

Hierarchical core+fragment deposits use fragment placement on the cluster
hull; validate with `assert_supported_cluster_binding`, not bare-slab height
windows.

### Optional MLIP extras on CI

`requires_mace` marks tests that import the MACE stack at runtime. UMA CI
jobs install only `uma` extras and exclude these tests by marker — not because
the physics is optional, but because the calculators are mutually exclusive
install targets on disk-limited runners.

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

## Kaggle GPU CI (manual)

GPU tests are **not** run on GitHub-hosted CPU runners. Trigger manually:

1. GitHub → Actions → **Kaggle GPU tests** → **Run workflow**
2. Leave defaults (`ref=main`, `marker=requires_cuda and not benchmark`) unless testing a branch
3. Requires repo secret `KAGGLE_API_TOKEN` (single-line API token from Kaggle Settings → API Tokens, or legacy `kaggle.json` pasted as one secret — the workflow normalizes both)

Example-mimic GPU integration coverage: `tests/integration/test_gpu_examples_integration.py` (all four `system_type` values from `examples/`).

### Local equivalents

```bash
# All GPU tests (skipped without CUDA)
pytest tests/ -m "requires_cuda and not benchmark" -v

# Example-mimic GPU integration only
pytest tests/integration/test_gpu_examples_integration.py -v
```
