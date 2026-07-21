# SCGO: Simple Cluster Global Optimization

[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/) [![PyPI](https://img.shields.io/pypi/v/scgo.svg)](https://pypi.org/project/scgo/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![SCGO Logo](docs/source/_static/scgo_logo.svg)

A compact toolkit for global optimization of atomic clusters using ASE. SCGO provides a focused API for Basin Hopping (BH) and Genetic Algorithm (GA) workflows with practical defaults.

**Documentation:** [Read the Docs](https://scgo.readthedocs.io/)

## Features

- **Basin Hopping and Genetic Algorithm** global optimization with automatic algorithm selection by cluster size
- **MLIP support** — MACE, UMA (fairchem), and UPET (metatomic) for GPU-accelerated optimization via TorchSim
- **Surface workflows** — slab-supported clusters and adsorbates with hull-site placement and tag-aware GA operators
- **Transition state search** — NEB-based TS search with automated pair selection and PBC-aware endpoint alignment
- **Flexible API** — high-level runners (`run_go`, `run_go_ts`, …) and low-level control for custom workflows
- **Reproducible initialization** — composition-canonical atom ordering for multi-element GA runs; mass-biased placement with per-structure RNG threading

## Install

Install with exactly one MLIP extra per environment (`[mace]`, `[uma]`, or `[upet]`):

```bash
pip install "scgo[mace]"   # or: pip install "scgo[uma]" or pip install "scgo[upet]"
```

Requires Python 3.12+ and SQLite with the JSON1 extension. See the [installation guide](https://scgo.readthedocs.io/en/latest/installation.html) for conda, editable installs, development extras, and HPC notes.

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

`results` is a list of `(energy, Atoms)` unique minima, sorted by energy. For sequential multi-composition runs, use `run_go_campaign`.

## Workflows

| Goal | Entry point | Documentation |
|------|-------------|---------------|
| Single composition GO | `run_go` | [Quick start](https://scgo.readthedocs.io/en/latest/quickstart.html) |
| Multi-composition GO | `run_go_campaign` | [Quick start — Campaigns](https://scgo.readthedocs.io/en/latest/quickstart.html#campaigns) |
| TS from existing minima | `run_ts_search` | [Quick start — Transition States](https://scgo.readthedocs.io/en/latest/quickstart.html#transition-states) |
| GO then TS | `run_go_ts` | [Quick start — Transition States](https://scgo.readthedocs.io/en/latest/quickstart.html#transition-states) |
| Multi-composition TS | `run_ts_campaign` | [Quick start — Campaigns](https://scgo.readthedocs.io/en/latest/quickstart.html#campaigns) |
| Multi-composition GO+TS | `run_go_ts_campaign` | [Quick start — Campaigns](https://scgo.readthedocs.io/en/latest/quickstart.html#campaigns) |

Pass one of four `system_type` values on every run: `gas_cluster`, `surface_cluster`, `gas_cluster_adsorbate`, or `surface_cluster_adsorbate`. See [system types](https://scgo.readthedocs.io/en/latest/api/system_types.html) for when to use each.

Output layout depends on the runner: `run_go` writes directly to `{formula}_searches/` (default in the current directory); combined and TS workflows use a **campaign root** with sibling `{formula}_searches/` and `{formula}_ts_results/` subdirectories. See [output directories](https://scgo.readthedocs.io/en/latest/quickstart.html#output-directories) and [on-disk layout](https://scgo.readthedocs.io/en/latest/quickstart.html#on-disk-layout) (run IDs, provenance, timing).

## Examples

Runnable scripts in [`examples/`](examples/) (MACE + TorchSim by default):

| Script | `system_type` | Notes |
|--------|---------------|-------|
| [`examples/example_pt5_gas.py`](examples/example_pt5_gas.py) | `gas_cluster` | Gas-phase Pt5 |
| [`examples/example_pt5_graphite.py`](examples/example_pt5_graphite.py) | `surface_cluster` | Pt5 on preset graphite |
| [`examples/example_pt5_oh_gas.py`](examples/example_pt5_oh_gas.py) | `gas_cluster_adsorbate` | Pt5 + OH in gas phase |
| [`examples/example_pt5_2oh_graphite.py`](examples/example_pt5_2oh_graphite.py) | `surface_cluster_adsorbate` | Pt5 + 2 OH on graphite |

## Development

```bash
pip install -e ".[mace,dev]"   # or: pip install -e ".[uma,dev]"
pre-commit install
pytest tests/ -m "not slow"
```

Long-running MLIP regression sweeps live in [`benchmark/`](benchmark/) (see [`benchmark/README.md`](benchmark/README.md)).

---

MIT License — see [`LICENSE`](LICENSE).
