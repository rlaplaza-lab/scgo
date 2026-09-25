# SCGO test suite

## Marker policy

| Marker | Meaning | CI |
|--------|---------|-----|
| `slow` | Real optimizers, NEB, or heavy placement loops | Selected by slow job (`slow and not benchmark`) |
| `integration` | Full workflow (GO campaigns, output trees) | Must also be `slow`; excluded from fast job (`not integration`), selected by slow job |
| `benchmark` | Long MLIP regression (Cu₄ MACE E2E) | Excluded from CI |
| `requires_cuda` | Needs GPU | Deselected on CPU runners (`not requires_cuda`); Kaggle / Modal GPU CI |
| `requires_mace` | Needs MACE extra | Excluded from UMA/UPET CI jobs; Kaggle MACE suite |
| `requires_upet` | Needs UPET extra | Excluded from MACE/UMA CI jobs; Kaggle UPET suite |
| `requires_uma` | Needs UMA extra | Excluded from MACE/UPET CI jobs |
| `gpu_smoke` | Short GPU smoke (relaxer + tiny GO) | Kaggle `mode=smoke` (default / weekly cron) |
| `requires_multicore` | Needs ≥2 CPUs | Skipped on single-core |
| `reproducibility` | Asserts deterministic RNG/seed behavior | Runs within its suite's fast/slow split |
| `requires_cache_isolation` | Needs clean global init caches | Runs within its suite's fast/slow split |
| `xdist_group` | Groups tests for pytest-xdist | Internal grouping only |

### GitHub Actions (CPU)

Source of truth: [`.github/workflows/ci.yml`](../.github/workflows/ci.yml).

**Every PR** (unique coverage, less wall-clock):

| Suite | Python | Fast | Slow |
|-------|--------|------|------|
| mace | 3.13 | yes | yes |
| uma | 3.13 | yes | no |
| upet | 3.13 | yes | no |

Fast markers always include `not requires_cuda` (deselected, not collected-then-skipped).
Example MACE fast expression:
`pytest tests/ -m "not slow and not integration and not requires_cuda and not requires_upet and not requires_uma"`.

**Push to `main`, nightly cron, and `workflow_dispatch`:** full matrix — all three suites × Python 3.12 and 3.13, each with fast **and** slow.

### Local runs

```bash
# Fast subset (matches the MACE suite's CI marker; see ci.yml for uma/upet)
pytest tests/ -m "not slow and not integration and not requires_cuda and not requires_upet and not requires_uma"

# Slow subset (matches the MACE suite's CI slow marker)
SCGO_BATCH_TEST_SAMPLES=15 pytest tests/ -m "slow and not benchmark and not requires_cuda and not requires_upet and not requires_uma"

# CPU EMT 6-type e2e only
pytest tests/integration/test_run_go_e2e.py -v

# Physics reference tests
pytest tests/physics/test_reference_emt.py -v
```

Install dev extras: `pip install -e ".[mace,dev]"`, `pip install -e ".[uma,dev]"`,
or `pip install -e ".[upet,dev]"` (exactly one MLIP extra per environment).

## Primary integration contract

**CPU EMT (all six `system_type`s):** [`tests/integration/test_run_go_e2e.py`](integration/test_run_go_e2e.py)

Parametrized public `run_go` / `run_go_ts` matrix plus tagging and H₂ negative-control cases. Shared strict bars live in [`helpers.py`](helpers.py) (`assert_e2e_minima_list`, `assert_e2e_go_ts_summary`, `assert_e2e_run_artifacts`):

- Finite energies and expected atom counts (slab + mobile)
- Database written and SCGO-stamped; `run_*/metadata.json` present
- Final tagging / XYZ when requested
- Supported surface deposits: `assert_supported_cluster_binding` (+ fragment lengths)
- TS: candidate counts when required; `assert_ts_result_valid` / finite barriers on success; explicit zero-TS negative control

**GPU MACE (all six `system_type`s):** [`tests/integration/test_gpu_examples_integration.py`](integration/test_gpu_examples_integration.py) — same assertion helpers, budgets from the shared low-effort presets that `examples/` also uses. Split into `test_run_go_ts_gpu_example_smoke_mace` / `_upet` so each Kaggle suite collects only its six cases.

**API wiring (mocked, fast):** [`tests/integration/test_run_api.py`](integration/test_run_api.py) — validation, coherence errors, 6-type optimizer wiring matrix.

## Physics helpers

Shared assertions live in [`helpers.py`](helpers.py):

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

`requires_mace` marks tests that import the MACE stack at runtime; UMA and UPET
CI jobs install only their own extras and exclude these tests by marker, since
the calculators are mutually exclusive install targets on disk-limited runners.
`requires_upet` is the analogous marker for the UPET / metatomic-torchsim stack.

UMA tests that only import helpers / use mocks run on the UMA CPU job. Tests that
construct a real FairChem relaxer may skip when HuggingFace weights are
unavailable (no `HF_TOKEN` on Actions).

## GPU CI (Kaggle weekly + Modal monthly)

GPU tests are **not** run on GitHub-hosted CPU runners. Two remote workflows
cover them:

| Workflow | Hardware | Default mode | Schedule |
|----------|----------|--------------|----------|
| [kaggle-gpu.yml](../.github/workflows/kaggle-gpu.yml) | Kaggle Tesla T4 | smoke | Weekly (Sunday 06:00 UTC) |
| [modal-gpu.yml](../.github/workflows/modal-gpu.yml) | Modal H100 | full | Monthly (1st 06:00 UTC) |

**UMA is omitted** from both (HuggingFace auth for fairchem / UMA weights is
typically unavailable). MACE and UPET always run as separate jobs (mutually
exclusive extras).

### Kaggle GPU CI (weekly smoke)

| Mode | When | Marker (per suite) | Kernel timeout |
|------|------|--------------------|----------------|
| **smoke** (default) | Manual dispatch default; weekly cron | `gpu_smoke and requires_{mace\|upet} and not benchmark` | 1 h |
| **full** | Manual only — prefer Modal for scheduled full runs | `requires_cuda and requires_{mace\|upet} and not benchmark` | 3 h |

1. GitHub → Actions → **Kaggle GPU tests** → **Run workflow**
2. Choose `mode=smoke` unless you need the full GO+TS example matrix / parallel NEB suite
3. Leave `ref=main` and empty `marker` unless testing a branch or overriding selection
4. Requires repo secret `KAGGLE_API_TOKEN` (single-line API token from Kaggle Settings → API Tokens, or legacy `kaggle.json` pasted as one secret — the workflow normalizes both)

The workflow uploads a source tarball to the private Kaggle dataset
`rlaplaza/scgocisrc` so the GPU kernel can run without GitHub network access;
pip installs still require internet on the kernel. The kernel requests a
Tesla T4 (Kaggle's fallback P100 is incompatible with the cu124 wheels used
here).

```bash
# Prefer smoke while iterating
gh workflow run kaggle-gpu.yml -f ref=<branch> -f mode=smoke

# Full suite on Kaggle (expensive quota); Modal monthly is the usual full run
gh workflow run kaggle-gpu.yml -f ref=<branch> -f mode=full
```

### Modal GPU CI (monthly full on H100)

| Mode | When | Marker (per suite) | Function timeout |
|------|------|--------------------|------------------|
| **full** (default) | Manual default; monthly cron (1st 06:00 UTC) | `requires_cuda and requires_{mace\|upet} and not benchmark` | 4 h |
| **smoke** | Manual only | `gpu_smoke and requires_{mace\|upet} and not benchmark` | 4 h |

1. GitHub → Actions → **Modal GPU tests** → **Run workflow**
2. Leave `mode=full` for the full CUDA suites; use `smoke` only to spend fewer credits
3. Requires repo secrets:
   - `MODAL_TOKEN_ID` — Modal token id (`ak-…`)
   - `MODAL_API_TOKEN` — Modal token secret (mapped to `MODAL_TOKEN_SECRET` in the job)
4. Modal may place an `H100` request on an H200 at the same price; that is fine
   for these tests. A payment method must still be on the Modal account even when
   monthly free credits cover the run.

```bash
gh workflow run modal-gpu.yml -f ref=<branch> -f mode=full
```

### Coverage notes

Smoke coverage:

- MACE: [`tests/integration/test_gpu_mace_smoke.py`](integration/test_gpu_mace_smoke.py)
- UPET: [`tests/integration/test_gpu_upet_smoke.py`](integration/test_gpu_upet_smoke.py)

Full coverage also includes the six-type example matrix
([`test_gpu_examples_integration.py`](integration/test_gpu_examples_integration.py))
and other `requires_cuda` tests pinned to one MLIP suite. Full-mode MACE
coverage includes an example-mimic matrix: all six `system_type` values built
from the same low-effort presets the `examples/` scripts use, with shared e2e
bars (run-dir `metadata.json`, SCGO-stamped `*.db`, and per-case TS-success /
barrier-range requirements).

### Local equivalents

```bash
# MACE GPU smoke
pytest tests/ -m "gpu_smoke and requires_mace and not benchmark" -v

# UPET GPU smoke
pytest tests/ -m "gpu_smoke and requires_upet and not benchmark" -v

# Full MACE / UPET GPU suites
pytest tests/ -m "requires_cuda and requires_mace and not benchmark" -v
pytest tests/ -m "requires_cuda and requires_upet and not benchmark" -v

# Example-mimic GPU integration only (MACE)
pytest tests/integration/test_gpu_examples_integration.py -k mace -v
```
