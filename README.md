# SCGO: Simple Cluster Global Optimization

[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/) [![PyPI](https://img.shields.io/pypi/v/scgo.svg)](https://pypi.org/project/scgo/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![SCGO Logo](docs/source/_static/scgo_logo.svg)

A compact toolkit for global optimization of atomic clusters using ASE. SCGO provides a focused API for Basin Hopping (BH) and Genetic Algorithm (GA) workflows with practical defaults.

**Documentation**: Comprehensive API documentation is available in the `docs/` directory. For online documentation, see [ReadTheDocs](https://scgo.readthedocs.io/).

## Install

SCGO has a small core dependency set plus two mutually exclusive MLIP extras:

- `[mace]` for MACE + TorchSim + `nvalchemi-toolkit-ops`
- `[uma]` for `fairchem-core` UMA checkpoints

Install only one of `[mace]` or `[uma]` per environment.

**From PyPI (recommended):**

```bash
pip install "scgo[mace]"   # or: pip install "scgo[uma]"
```

For development or editable installs from source, clone the repository and use `pip install -e ".[mace]"` (see [installation docs](https://scgo.readthedocs.io/en/latest/installation.html)).

Conda (full dev stack from source):

```bash
git clone https://github.com/rlaplaza-lab/scgo.git
cd scgo
conda env create -f environment.yml
conda activate scgo
```

`environment.yml` installs the package editable with **`[mace,dev]`** (MACE/TorchSim + test/lint tooling).

The conda env uses `torch-sim-atomistic[mace]` with `nvalchemi-toolkit-ops` for TorchSim neighbor lists. Do not install `vesin` or `vesin-torch`—they conflict with the TorchSim stack we use.

Note: SCGO requires SQLite with the JSON1 extension (for `json_extract` and related functions). If you installed using conda, ensure `sqlite` from `conda-forge` is available in your environment (e.g., `conda install -c conda-forge sqlite`). If you use pip-only installs, consider installing `pysqlite3-binary` (e.g., `pip install pysqlite3-binary`) so that the Python `sqlite3` module exposes JSON1. This repository's CI enforces JSON1 availability.

Sella is not required by the core SCGO package and has been removed from the default pip constraints to avoid heavy native builds during dependency resolution. If you need Sella for advanced optimization features, install it manually (it builds C extensions and may require a C toolchain and Cython).

Editable install from source (alternative):

```bash
git clone https://github.com/rlaplaza-lab/scgo.git
cd scgo
pip install -e ".[mace]"   # or: pip install -e ".[uma]"
```

For pip installs, the same TorchSim stack applies: ensure `nvalchemi-toolkit-ops` is available; uninstall `vesin` and `vesin-torch` if you see TorchSim-related errors.

Dependency note: SCGO now allows `scipy>=1.14,<3` so the UMA/fairchem extra can resolve cleanly with `torch-sim-atomistic[fairchem]` (which constrains SciPy to `<1.17` in current releases).

For development with tests and linting (after a **runtime-only** `pip install -e .`):

```bash
pip install -e ".[mace,dev]"   # or UMA: pip install -e ".[uma,dev]"
pre-commit install
```

### Running on HPC (Slurm, shared filesystem)

- **SQLite**: SCGO keeps WAL mode off by default (fewer `-wal`/`-shm` issues on Lustre/GPFS/NFS). Prefer writing active `*.db` files under job-local scratch (`$SLURM_TMPDIR` or site-specific scratch) when you can, then copying results back to project storage.
- **Parallel jobs**: SCGO creates unique `run_<timestamp>_<microseconds>` folders, so jobs sharing a parent output directory usually write different DB files. Lock contention is still possible if jobs touch the same `*.db` (for example, reused explicit `run_id`/path) or if shared filesystems serialize lock files; for high parallelism, prefer one output directory per job.
- **Registry**: Discovery may write `.scgo_db_registry.json` and `.scgo_db_registry.lock` (with `flock` on Linux) for fast DB listing. When your run lives under a directory whose name ends in `_searches`, the index is kept at that parent only (not beside every `trial_*` folder). If your filesystem does not honor `flock`, use separate output directories per job or avoid parallel registry updates.
- **Logging**: Batch-friendly defaults suppress noisy third-party loggers. For local debugging, set `SCGO_LOCAL_DEV=1` or call `configure_logging(..., hpc_mode=False)`.

---

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- **Installation**: `docs/source/installation.rst` - Setup instructions for conda and pip
- **Quick Start**: `docs/source/quickstart.rst` - Basic usage examples and workflows
- **API Reference**: 
  - `docs/source/api/runner_api.rst` - High-level API entry points
  - `docs/source/api/surface.rst` - Slab configuration and deposition
  - `docs/source/api/cluster_adsorbate.rst` - Adsorbate placement and GA repositioning
  - `docs/source/api/param_presets.rst` - Parameter presets
  - `docs/source/api/system_types.rst` - System type definitions

To build the documentation locally:

```bash
pip install -e .
pip install -r docs/source/requirements.txt
cd docs && make html
```

The built documentation will be available in `docs/build/html/index.html`.

---

## Quick start

```python
from scgo import run_go
from scgo.param_presets import get_testing_params
results = run_go(
    ["Pt"] * 4,
    params=get_testing_params(),
    seed=42,
    system_type="gas_cluster",
)
```

- `results` is a list of `(energy, Atoms)` for unique minima (sorted by energy by default).
- Sequential multi-composition GO uses `run_go_campaign([...], system_type=...)` from [`scgo.runner_api`](scgo/runner_api.py) (also re-exported from `scgo`).

### Explicit system types

SCGO supports exactly four explicit `system_type` values:

- `gas_cluster`: gas-phase cluster (no slab, no extra adsorbate constraints)
- `surface_cluster`: cluster supported on a slab (`surface_config` required)
- `gas_cluster_adsorbate`: gas-phase cluster that includes adsorbate-like species (no slab)
- `surface_cluster_adsorbate`: supported cluster + adsorbate species (`surface_config` required)

`system_type` must be passed to each `run_*` API call. Top-level `system_type` is rejected inside preset dicts (`go_params` / `ts_params`); use the run function argument instead. For surface workflows, `surface_config` may appear in presets (e.g. `get_torchsim_ga_params`, `get_ts_search_params`) and on the `run_*` call—values must agree when both are set.
For adsorbate system types (`gas_cluster_adsorbate`, `surface_cluster_adsorbate`),
high-level runners require core-only `composition` and `adsorbates` (one ASE `Atoms`
fragment or a list of fragments). SCGO flattens adsorbate symbols in provided fragment
order and constructs the full mobile composition as
`core_composition + flattened_adsorbate_symbols` (mobile region after any slab).
Hierarchical initialization is the only supported adsorbate layout.

---

## What to expect on disk (output)

When you run a search for composition `Pt4`, SCGO writes into `Pt4_searches/` with the following structure:

### Global optimization (`{formula}_searches/`)

- `Pt4_searches/run_<YYYYMMDD_HHMMSS_ffffff>/trial_<N>/`
  - `bh_go.db` or `ga_go.db` (ASE database with candidates and relaxed structures)
  - `population.log` (GA runs)
- `Pt4_searches/results_summary.json` — campaign-level snapshot after the latest run. Top-level keys include:
  - **Provenance** (same convention as other SCGO JSON sidecars): `schema_version` (currently **3**), `scgo_version`, `created_at` (UTC ISO8601), `python_version`
  - **Run summary**: `composition` (formula string, e.g. `"Pt4"`), `total_unique_minima`, `minima_by_run` (map of `run_id` → count), `current_run_id`, `params` (JSON-safe snapshot aligned with `run_*/metadata.json`), `run_metadata_relpath` (e.g. `run_<id>/metadata.json`)
- `Pt4_searches/final_unique_minima/` — final XYZ files, named like `Pt4_minimum_01_run_YYYYMMDD_HHMMSS_ffffff_trial_1.xyz`
- `Pt4_searches/run_<...>/metadata.json` — per-run record: provenance header above plus `run_id`, `timestamp`, `composition` (symbol list), `formula`, `params`, and related run fields
- `Pt4_searches/validation/` — optional; created when `validate_with_hessian=True` to run vibrational checks
- `Pt4_searches/.scgo_db_registry.json` and `.scgo_db_registry.lock` — optional DB index and lock (see *Running on HPC* above)

Notes:
- If `clean=False`, SCGO will merge previous runs by scanning `run_*` directories and `trial_*/` DB files.
- `.db` files are ignored by the project `.gitignore`.

### Transition state search (`ts_results_{formula}/`)

`run_ts_search` (from `scgo`, wrapping [`scgo.runner_api`](scgo/runner_api.py)) reads minima from `{formula}_searches/` (or the `output_dir` you pass) and writes **under the same tree** into a dedicated folder:

- `{formula}_searches/ts_results_{formula}/`
  - **Per pair** `pair_id` (e.g. `0_1`): `ts_{pair_id}.xyz`, `reactant_{pair_id}.xyz`, `product_{pair_id}.xyz` (when geometries exist), and `neb_{pair_id}_metadata.json`
  - **`ts_search_summary_{formula}.json`** — full run: provenance header, NEB settings (`calculator_name`, `neb_fmax`, `neb_steps_resolved`, `neb_backend` `torchsim` or `ase`, `use_parallel_neb`, climb/interpolation flags, image count, spring constant, etc.), `composition`, `formula`, `num_total_pairs`, `num_successful`, `num_converged`, `results` (list of per-pair records), and `statistics` (`total_ts_found`, `converged_ts`, `successful_ts`, `min_barrier` / `max_barrier` / `avg_barrier` over successes)
  - **`ts_network_metadata_{formula}.json`** — graph-oriented view: `ts_connections[]` (each edge: `pair_id`, `minima_indices`, energies, `barrier_height`, optional `barrier_forward` / `barrier_reverse`, `neb_converged`, `n_images`, optional `minima_provenance`), `num_minima`, `statistics`, optional `minima_base_dir`
  - **`final_unique_ts/`** — after deduplication: `final_unique_ts_summary_{formula}.json` (provenance + `unique_ts[]` with `connected_edges`, `connected_minima`, `filename`, energies, etc.) and one `.xyz` per deduplicated TS (names may include `pair_…` when a single edge maps to that file)

**Per-pair entries** in `ts_search_summary_*.json` (and overlapping fields in `neb_*_metadata.json`) typically include: `pair_id`, `status` (`success` / `failed`), `neb_converged`, `n_images`, `spring_constant`, `reactant_energy`, `product_energy`, `ts_energy`, `barrier_height`, `error`, and on success `ts_image_index`. When traceability is available, `minima_indices` and **`minima_provenance`** appear: each endpoint lists `run_id`, `trial_id`, `source_db`, `source_db_relpath`, `systems_row_id`, `confid`, `gaid`, `unique_id`, `final_id`, `energy` (see `scgo/ts_search/transition_state_io.py`).

**`neb_{pair_id}_metadata.json`** merges the provenance header with pair fields above plus, when present: `final_fmax`, `steps_taken`, `force_calls`, and NEB-parameter echoes (`use_torchsim`, `neb_backend`, `interpolation_method`, `climb`, `align_endpoints`, `perturb_sigma`, `neb_interpolation_mic`, `neb_surface_cell_remap`, `neb_surface_lattice_rotation`, `neb_surface_max_lattice_shift`, `fmax`, `neb_steps`, etc.).

---

## Key options (short)

- **Global optimization (`params` for `run_go` / `run_go_campaign`)** is merged with `get_default_params()` via `initialize_params`: any preset that omits keys inherits defaults. Common entry points: `get_default_params()`, `get_minimal_ga_params()`, `get_testing_params()`, `get_high_energy_params()`, `get_diversity_params()`, `get_default_uma_params()` (fairchem UMA), and `get_torchsim_ga_params(system_type=..., surface_config=..., seed=..., model_name=...)` (MACE + TorchSim GA benchmark stack; requires `scgo[mace]`).
- **Transition-state search (`ts_params` for `run_ts_search` / `run_go_ts`)** is **not** merged with GO defaults. Build a flat dict with `get_ts_search_params(...)` (e.g. `calculator="UMA"` for UMA) and pass it explicitly alongside `go_params` when using `run_go_ts` / `run_go_ts_campaign`.

**NEB endpoint alignment (on by default):** Presets set `neb_align_endpoints=True` for all system types. Before ASE path interpolation (`idpp` or `linear`), SCGO reorders product atoms to match the reactant, then rigidly aligns endpoints so interior images start from a sensible band:

- **Gas clusters** — 3D Kabsch on the mobile region (or whole structure when no slab prefix).
- **Slab / periodic systems** — lattice-compatible PBC alignment (`neb_interpolation_mic=True` on surface types): MIC-aware atom matching, collective mobile lattice-image selection, per-atom MIC snapping, optional integer in-plane lattice shifts (`neb_surface_cell_remap`, search span `neb_surface_max_lattice_shift` default `1`), and global in-plane rotation evaluated jointly with each shift (`neb_surface_lattice_rotation`). Slab/`FixAtoms` anchors stay registered; mobile atoms are not rotated independently of the lattice frame (avoids energy-inequivalent distortions).

Path interpolation always uses the **aligned** reactant and product copies as band endpoints; only interior images are filled by `NEB.interpolate`. Disable with `ts_params["neb_align_endpoints"] = False` only when you intentionally want raw GO minima as endpoints.

**Surface GO final XYZ alignment:** For `surface_cluster` and `surface_cluster_adsorbate`, before writing `final_unique_minima/*.xyz`, SCGO PBC-aligns each minimum to the lowest-energy trial using the same slab protocol as NEB (`neb_surface_cell_remap`, `neb_surface_lattice_rotation`, `neb_surface_max_lattice_shift` from optimizer kwargs, gated by `SystemPolicy`). This keeps stored frames comparable across trials; TS/NEB endpoint alignment is unchanged.

**Initialization clash factor:** Default `min_distance_factor` is **0.4** (was 0.5) in `scgo.initialization.initialization_config.MIN_DISTANCE_FACTOR_DEFAULT`; override per run if you need stricter placement.

Preset-vs-runtime split in `runner_api`:

- Put scientific/tuning knobs in preset dicts (`go_params`/`ts_params`): calculator choice, optimizer settings, NEB settings, pairing thresholds, adsorbate placement (`cluster_adsorbate_config`), connectivity (`connectivity_factor`), etc.
- Keep run-control knobs on the `run_*` call itself: `verbosity`, `output_dir`, `output_root`, `output_stem`, `seed`, `log_summary`.
- Keep system-definition inputs on the `run_*` call itself: `system_type`, core-only `composition`, and `adsorbates` when applicable.
- For surface system types, pass `surface_config` on the `run_*` call and in preset builders (`get_torchsim_ga_params`, `get_ts_search_params`); SCGO validates coherence across GO/TS presets and run arguments.

**GA timing JSON** (set in `go_params` / `params` only):

```python
go_params["optimizer_params"]["ga"].update(
    write_timing_json=True,   # writes timing.json under each trial
    detailed_timing=True,     # include per_generation rows
)
```

Inspect -> edit -> run pattern:

```python
from scgo import run_go_ts
from scgo.param_presets import get_default_params, get_ts_search_params

go_params = get_default_params()
ts_params = get_ts_search_params(system_type="gas_cluster")

print(go_params["optimizer_params"]["ga"]["niter"])
go_params["optimizer_params"]["ga"]["niter"] = 8
ts_params["max_pairs"] = 12

summary = run_go_ts(
    "Pt5",
    go_params=go_params,
    ts_params=ts_params,
    system_type="gas_cluster",
    seed=7,
    verbosity=1,
)
```

After writing final XYZ files, SCGO can optionally tag the corresponding database records with metadata ("final_unique_minimum": true, "final_rank", and "final_written") so downstream tools can find final minima without re-scanning databases. This is enabled by default; disable with `params['tag_final_minima'] = False`.

- `fitness_strategy`: `low_energy` (default), `high_energy`, or `diversity`. The `diversity` strategy requires a `diversity_reference_db` glob (e.g., `"Pt*_searches/**/*.db"`).
- `validate_with_hessian` (bool): run force + Hessian checks (uses ASE vibrational analysis).
- GA backend: MLIPs use TorchSim batched GA; classical calculators use ASE GA.
- Database/perf knobs:
  - `db_enable_expression_indexes` (GA/BH, default `False`): enable extra SQLite JSON expression indexes for frequent metadata predicates/sorts.
  - `ga_adaptive_retry_enabled` (default `True`): adapt offspring attempt budget to recent acceptance rate instead of fixed `10*n_offspring`.
  - `ga_retry_floor_multiplier` / `ga_retry_ceiling_multiplier` (defaults `4` / `15`): lower/upper bounds for adaptive retry budget.
  - `ga_fast_prefilter_enabled` (default `True`): cheap severe-clash prefilter before full system-type validation.
  - **Timing:** set `optimizer_params['ga']['write_timing_json']=True` in `go_params` to write `timing.json` (`timings_s`, `counters`, `retry_failures`). Add `detailed_timing=True` for `per_generation` rows. TS timing uses `write_timing_json` in `ts_params` when needed.

---

## Surface workflows (supported clusters)

SCGO can run **genetic-algorithm** global optimization for a small **adsorbate cluster** on a periodic **slab**. The GA explores the adsorbate degrees of freedom (`composition`); the slab supplies the cell and controls which substrate atoms move during **local** relaxations via [`SurfaceSystemConfig`](scgo/surface/config.py) (`FixAtoms` under the hood, including on the TorchSim GA path).

### How to run

Build (or load) any ASE `Atoms` slab and pass it through the generic surface helper:

```python
from ase.build import fcc111
from scgo.surface import make_surface_config

slab = fcc111("Pt", size=(3, 3, 3), vacuum=10.0)
surface_config = make_surface_config(slab)
```

For the **graphite preset** used in example runners, use [`scgo.surface.make_graphite_surface_config`](scgo/surface/presets.py) (or `from scgo import make_graphite_surface_config`) instead of building a slab by hand.

Then build GO/TS presets and pass the same `surface_config` to the runner:

```python
from scgo.param_presets import get_torchsim_ga_params, get_ts_search_params
from scgo import run_go_ts

go_params = get_torchsim_ga_params(
    system_type="surface_cluster",
    surface_config=surface_config,
    seed=42,
)
ts_params = get_ts_search_params(
    system_type="surface_cluster",
    surface_config=surface_config,
    seed=42,
)
```

For the bundled graphite preset, `make_graphite_surface_config(slab_layers=3)` controls slab thickness (see `examples/example_pt5_graphite.py`).

- **Direct API** (any adsorbate size): `from scgo import ga_go, SurfaceSystemConfig` and pass `surface_config=...`.
- **`run_go`**: pass `surface_config=...` directly to `run_go(...)`; it is copied into each **present** `optimizer_params` entry among `simple` / `bh` / `ga` so the active algorithm sees the slab. Automatic algorithm choice follows mobile atom count (see **Algorithm selection** under Global optimization).
- For slab workflows, choose `system_type="surface_cluster"` (supported cluster only) or `system_type="surface_cluster_adsorbate"` (supported cluster with explicit adsorbate-mode policies). Use `scgo.surface.make_surface_config` for a custom ASE slab; use `scgo.surface.make_graphite_surface_config` for the bundled graphite template.

Adsorbate inputs and initial structures: For both `gas_cluster_adsorbate` and `surface_cluster_adsorbate`, use core-only `composition` plus `adsorbates` as the primary API (`Atoms` for one fragment, or `list[Atoms]` for multiple fragments). SCGO derives a strict mobile partition in order (`core_symbols == composition`, then flattened adsorbate symbols); slab atoms are not part of `composition`. SCGO uses hierarchical initialization only: build the core, place rigid fragment(s) on convex-hull adsorption sites (vertex/edge/facet), then (for surface) deposit the combined cluster on the slab. Placement ranks candidate poses by steric deficit (covalent-radius `blmin`) and progressively relaxes height/clash thresholds on retry. Optional fragment placement and validation tuning lives in `go_params` (`cluster_adsorbate_config=ClusterAdsorbateConfig(...)`, or set `connectivity_factor` alone for the common case). Use `scgo.surface.describe_surface_config` to log effective slab and height settings. GA and basin-hopping attach `n_core_atoms` and per-role symbol JSON in metadata for round-trip checks (including fragment-length metadata for multi-fragment adsorbates). When adsorbate metadata is present, [`validate_structure_for_system_type`](scgo/system_types.py) also asserts that the mobile region's chemical symbols match `core_symbols + adsorbate_symbols` in order (in addition to geometry checks). Input `adsorbates` fragments are validated to be connected geometries. `adsorbate_definition['adsorbate_fragment_lengths']` is optional for manual definitions: when set, integrity is enforced per fragment; when omitted, integrity falls back to the full adsorbate block as one connected subgraph. Set `freeze_adsorbate_internal_geometry=True` in GO params for strict template rigidity (Kabsch restore after mutations); the default (`False`) still preserves intra-fragment bonds via tag-rigid GA operators.

### Adsorbate GA behavior (`*_adsorbate`)

For adsorbate system types, the GA uses ASE tags to partition the mobile region: **core** (tag `0`) vs **adsorbate fragments** (tags `1..N`).

| Mechanism | Behavior |
|-----------|----------|
| Crossover | Cut-and-splice on the **core only**; adsorbate fragments stay on parent 0. |
| Rattle / overlap relief | Tag-rigid: each fragment moves as a unit (intra-fragment geometry preserved). |
| Rotational / mirror / flattening / breathing | Core-targeted variants (`*_core`); adsorbate distortions omitted or scoped to adsorbate tags when freeze is off. |
| `fragment_reposition` | Re-place one adsorbate fragment on fresh hull sites (same placement engine as init). |
| `in_plane_slide` | Surface-only; core and adsorbate variants (`in_plane_slide_core`, `in_plane_slide_ads`). |
| Validation | `enforce_adsorbate_subgraph_integrity=True` (default) rejects dissociated fragments post-operator and post-relax. |

Clash tables (`blmin`) and placement use gap-filled covalent radii via [`build_blmin`](scgo/initialization/atomic_radii.py) (`BLMIN_RATIO_DEFAULT=0.7`). Structure validation uses `connectivity_factor` (default `1.4`) — typically stricter than operator sterics, so borderline disconnections are caught at validation rather than during mutation.

### Slab motion during local relaxation

| Mode | Settings |
|------|----------|
| Entire slab frozen | `fix_all_slab_atoms=True` (default) |
| Relax only the top **N** slab layers (along `surface_normal_axis`) | `fix_all_slab_atoms=False`, `n_relax_top_slab_layers=N` |
| Same intent, using bottom layer count | `fix_all_slab_atoms=False`, `n_fix_bottom_slab_layers=L - N` where `L` is the number of distinct slab layers along that axis |
| Slab fully free to relax | `fix_all_slab_atoms=False` and leave `n_relax_top_slab_layers` and `n_fix_bottom_slab_layers` unset (`None`) |

Do not set `n_relax_top_slab_layers` together with `n_fix_bottom_slab_layers`, or together with `fix_all_slab_atoms=True`. For typical `ase.build.fcc111` slabs with vacuum along **z**, use `surface_normal_axis=2` (the default).

Run metadata records a JSON-safe summary of these flags (no embedded `Atoms`) under the sanitized `surface_config` key.

### Surface mobile connectivity (validation)

For `surface_cluster` and `surface_cluster_adsorbate`, GO/GA/BH and TS validate that the mobile region is slab-bound. Defaults in `get_default_params()` and `get_ts_search_params()`:

| Flag | Default | Effect when `True` |
|------|---------|-------------------|
| `allow_cluster_fragmentation` | `False` | Multiple disconnected core/mixed mobile subgroups allowed (each must touch the slab). |
| `allow_adsorbate_surface_detachment` | `False` | Adsorbate-only mobile subgroups on the slab without cluster contact (requires exactly one core/mixed subgroup when fragmentation is off). |
| `enforce_adsorbate_subgraph_integrity` | `True` | Keep adsorbate subgraphs connected (non-dissociative). With `adsorbate_fragment_lengths`, this is per fragment; otherwise it applies to the whole adsorbate block. External bonds to core/other fragments are still allowed. |

**Typical modes:** both `False` (strict single connected mobile cluster); fragmentation only (`True`, `False`); both `True` (any mobile split, each subgroup slab-connected). For `surface_cluster_adsorbate`, core vs adsorbate subgroups are classified using `adsorbate_definition['core_symbols']`.

**Breaking rename:** `allow_dissociative_adsorption` was removed. Migration:

| Old | New |
|-----|-----|
| `False` | `allow_cluster_fragmentation=False`, `allow_adsorbate_surface_detachment=False` |
| `True` | both flags `True` |
| (no old equivalent) | fragmentation only or detachment only — use the partial combos above |

**TS pair selection:** When `surface_config` is set, structural similarity and pair filtering use `n_slab=len(slab)` so displacements in frozen slab atoms alone do not create spurious distinct-minima pairs.

---

## Testing

```bash
# Fast default
pytest tests/ -m "not slow"

# Integration-only
pytest tests/ -m integration

# Slow-only
pytest tests/ -m slow
```

Fast EMT "benchmark" smoke tests (initialization and dimers) live under [`tests/benchmarks/`](tests/benchmarks/); long MLIP regression sweeps live under the top-level [`benchmark/`](benchmark/) package (see [`benchmark/README.md`](benchmark/README.md)).

For long GA/TorchSim tests, run in foreground with live output (`-s`) and an explicit timeout to avoid "looks stalled" sessions:

```bash
timeout 5400 pytest tests/ -m "not slow" -vv -s
```

---

## High-Level API

Canonical workflow entry points are defined in [`scgo/runner_api.py`](scgo/runner_api.py) and imported from the `scgo` package. Composition arguments may be a **formula string** (`"Pt3Au"`), a **symbol list**, or **`ase.Atoms`** (only symbols are used for GO).

### Global optimization

#### `run_go(composition, params=None, seed=None, ...)`

Single composition; returns a list of `(energy, Atoms)` unique minima.

```python
from scgo import run_go
from scgo.param_presets import get_default_params

results = run_go(
    ["Pt", "Pt", "Pt", "Pt"],
    params=get_default_params(),
    seed=42,
    verbosity=1,
    clean=False,
    output_dir=None,
    system_type="gas_cluster",
)
```

**Algorithm selection** (mobile atom count): 1–2 → simple (plain `gas_cluster` only); 3 → basin hopping; 4+ → genetic algorithm. Adsorbate system types skip `simple` (two-atom mobile regions use GA). For `*_adsorbate` types, pass core-only `composition` and `adsorbates=` on the `run_*` call; SCGO builds fragment templates and hierarchical initial structures internally.

#### `run_go_campaign(compositions, ..., system_type=...)`

Run GO for each composition **sequentially**; returns `dict[formula, list[(energy, Atoms)]]`.

For element or binary size scans, build composition lists with helpers from `scgo.runner_api`, then pass them to `run_go_campaign`:

```python
from scgo import run_go_campaign
from scgo.param_presets import get_testing_params
from scgo.runner_api import build_one_element_compositions, build_two_element_compositions

params = get_testing_params()
pt_scan = build_one_element_compositions("Pt", min_atoms=2, max_atoms=6)
au_pt_scan = build_two_element_compositions("Au", "Pt", min_atoms=2, max_atoms=4)
results = run_go_campaign(pt_scan, params=params, seed=42, system_type="gas_cluster")
```

### Transition state search

`run_ts_search` and `run_ts_campaign` take a **flat `ts_params` dict** from [`get_ts_search_params`](scgo/param_presets.py) (or edit a copy). TorchSim use is resolved from the calculator; pass `system_type` on the `run_*` call.

Per-system NEB defaults from `get_ts_search_params` include:

| Key | Gas types | Surface types |
|-----|-----------|---------------|
| `neb_align_endpoints` | `True` | `True` |
| `neb_interpolation_mic` | `False` | `True` (forced by policy) |
| `neb_surface_cell_remap` | `False` | `True` |
| `neb_surface_lattice_rotation` | `False` | `True` |
| `neb_surface_max_lattice_shift` | `1` | `1` |

For `*_adsorbate` runs, pass `adsorbates=` to `run_go_ts` so TS can use blockwise slab / core / adsorbate endpoint matching when alignment is enabled.

```python
from scgo import run_ts_search
from scgo.param_presets import get_ts_search_params

ts_params = get_ts_search_params(system_type="gas_cluster", seed=42)
ts_results = run_ts_search(
    ["Pt", "Pt", "Pt"],
    output_dir="Pt3_searches",
    ts_params=ts_params,
    seed=42,
    system_type="gas_cluster",
)
```

`run_ts_campaign` forwards the same `ts_params` to each composition.

### GO then TS

`run_go_ts` / `run_go_ts_campaign` use **`go_params=`** (merged like other GO runs) and **`ts_params=`** (same flat shape as above; **not** deep-merged with `get_default_params()`). For `*_adsorbate` system types, pass core-only `composition` and `adsorbates` on the `run_*` call (same as `run_go`); SCGO builds the full mobile composition internally. For surface workflows, pass `surface_config` on the `run_*` call and in preset builders (`get_torchsim_ga_params`, `get_ts_search_params`); values must agree when both are set. For MACE + TorchSim GA, start from [`get_torchsim_ga_params`](scgo/param_presets.py) with `system_type=...` and `seed` (optional `surface_config=` / `model_name=`), set `go_params["calculator"] = "MACE"` and `optimizer_params["ga"]` as needed; pair with `get_ts_search_params(...)` and set `ts_params["max_pairs"]`, etc. For UMA NEB defaults, use `get_ts_search_params(calculator="UMA", ...)`. See `examples/example_pt5_gas.py` for a minimal end-to-end example. Default output if `output_dir` is omitted is under `scgo_runs/<stem>_<mace|uma>/` (set `output_root` / `output_stem` to change).

Benchmarks comparing MACE vs UMA on the same GA structure can use [`get_uma_ga_benchmark_params`](scgo/param_presets.py) (re-exported from `scgo`). See `benchmark/` for long-running MLIP regression sweeps.

### Advanced / internals

- `from scgo.runner_api import _run_go_trials`, `_run_go_campaign_compositions`, `build_one_element_compositions`, `build_two_element_compositions`, …
- `from scgo import make_graphite_surface_config, make_surface_config` for preset graphite or custom ASE slabs
- `from scgo.cluster_adsorbate import ClusterAdsorbateConfig, place_fragment_on_cluster` — adsorbate placement; see `docs/source/api/cluster_adsorbate.rst`
- `from scgo.initialization.atomic_radii import build_blmin` — covalent-radius clash tables for GA and placement
- Low-level `scgo(...)` / `run_trials(...)`: pass `system_type` in `global_optimizer_kwargs` (required for `scgo`; `run_trials` defaults missing values to `"gas_cluster"`)
- `from scgo.ts_search.transition_state_run import run_transition_state_search` for a flat keyword API without the `ts_params` dict.

---

## Notes

- TorchSim is an optional tool that provides GPU-accelerated batched optimization when available; SCGO works with EMT (CPU) out of the box for quick tests.
- For reproducible results, pass `seed=` to the workflow functions above.
- Optional scripts in [`examples/`](examples/) are minimal, no-CLI examples that call [`run_go_ts`](scgo/runner_api.py). Each is tuned for MACE + TorchSim (edit calculator in the script if needed):

| Script | `system_type` | Notes |
|--------|----------------|-------|
| [`examples/example_pt5_gas.py`](examples/example_pt5_gas.py) | `gas_cluster` | Gas-phase `Pt5` only |
| [`examples/example_pt5_graphite.py`](examples/example_pt5_graphite.py) | `surface_cluster` | `Pt5` on preset graphite |
| [`examples/example_pt5_oh_gas.py`](examples/example_pt5_oh_gas.py) | `gas_cluster_adsorbate` | core-only `Pt5` + one OH; tag-aware GA, optional `freeze_adsorbate_internal_geometry` |
| [`examples/example_pt5_2oh_graphite.py`](examples/example_pt5_2oh_graphite.py) | `surface_cluster_adsorbate` | core-only `Pt5` + two OH on graphite; hull-site placement + `fragment_reposition` |

For multi-size MLIP sweeps, see [`benchmark/`](benchmark/) (e.g. [`benchmark/benchmark_Pt.py`](benchmark/benchmark_Pt.py), [`benchmark/benchmark_Pt_surface_graphite.py`](benchmark/benchmark_Pt_surface_graphite.py)), not `tests/benchmarks/`.

See `tests/` for concrete usage patterns and acceptance tests for adsorbate GA operators.

---

MIT License — see `LICENSE`.