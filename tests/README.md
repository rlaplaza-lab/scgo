# SCGO test suite

## Marker policy

| Marker | Meaning | CI |
|--------|---------|-----|
| `slow` | Real optimizers, NEB, or heavy placement loops | `slow and not benchmark` |
| `integration` | Full workflow (GO campaigns, output trees) | Excluded from fast job (`not integration`) |
| `benchmark` | Long MLIP regression (Cu₄ MACE E2E) | Excluded from CI |
| `requires_cuda` | Needs GPU | Skipped on CPU runners |
| `requires_mace` | Needs MACE extra | Excluded from UMA CI jobs; Kaggle MACE suite |
| `requires_upet` | Needs UPET extra | Kaggle UPET suite (`scgo[upet]`) |
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
install targets on disk-limited runners. `requires_upet` is the analogous
marker for the UPET / metatomic-torchsim stack.

## Local runs

```bash
# Fast subset (matches CI)
pytest tests/ -m "not slow and not integration"

# Slow subset
SCGO_BATCH_TEST_SAMPLES=15 pytest tests/ -m "slow and not benchmark"

# New physics reference tests only
pytest tests/physics/test_reference_emt.py -v
```

Install dev extras: `pip install -e ".[mace,dev]"`, `pip install -e ".[uma,dev]"`,
or `pip install -e ".[upet,dev]"` (exactly one MLIP extra per environment).

## Kaggle GPU CI (manual)

GPU tests are **not** run on GitHub-hosted CPU runners. Trigger manually:

1. GitHub → Actions → **Kaggle GPU tests** → **Run workflow**
2. Leave defaults (`ref=main`, empty `marker`) unless testing a branch — the workflow
   runs **two kernels** in parallel:
   - **MACE**: `requires_cuda and requires_mace and not benchmark`
   - **UPET**: `requires_cuda and requires_upet and not benchmark`
3. **UMA is not run on Kaggle** (HuggingFace auth for fairchem / UMA weights is
   typically unavailable there).
4. Requires repo secret `KAGGLE_API_TOKEN` (single-line API token from Kaggle Settings → API Tokens, or legacy `kaggle.json` pasted as one secret — the workflow normalizes both)

The workflow uploads a source tarball to the private Kaggle dataset `rlaplaza/scgocisrc` so the GPU kernel can run without relying on GitHub network access from Kaggle. Kaggle may mount that dataset as either `scgo-src.tar.gz` or an extracted tree under `/kaggle/input/scgocisrc/`. **Pip installs (MACE/UPET/TorchSim) still require internet on the Kaggle kernel** — enable it in your Kaggle account settings and complete phone verification if GPU sessions cannot reach PyPI. The kernel requests a **Tesla T4** GPU (`machine_shape: NvidiaTeslaT4`). Kaggle may assign a P100 otherwise; its sm_60 architecture is incompatible with the cu124 PyTorch wheels used here.

Example-mimic GPU integration coverage (MACE): `tests/integration/test_gpu_examples_integration.py` (all four `system_type` values from `examples/`).

UPET GPU smoke coverage: `tests/integration/test_gpu_upet_smoke.py`.

### Local equivalents

```bash
# MACE GPU suite (skipped without CUDA / scgo[mace])
pytest tests/ -m "requires_cuda and requires_mace and not benchmark" -v

# UPET GPU suite (skipped without CUDA / scgo[upet])
pytest tests/ -m "requires_cuda and requires_upet and not benchmark" -v

# Example-mimic GPU integration only (MACE)
pytest tests/integration/test_gpu_examples_integration.py -v
```
