# SCGO: Simple Cluster Global Optimization

[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/) [![PyPI](https://img.shields.io/pypi/v/scgo.svg)](https://pypi.org/project/scgo/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![SCGO Logo](docs/source/_static/scgo_logo.svg)

A compact toolkit for global optimization of atomic clusters using ASE. SCGO provides a focused API for Basin Hopping (BH) and Genetic Algorithm (GA) workflows with practical defaults.

**Documentation:** [Read the Docs](https://scgo.readthedocs.io/)

## Features

- **Basin Hopping and Genetic Algorithm** global optimization with automatic algorithm selection by cluster size
- **MLIP support** — MACE and UMA (fairchem) for GPU-accelerated optimization via TorchSim
- **Surface workflows** — slab-supported clusters and adsorbates with hull-site placement and tag-aware GA operators
- **Transition state search** — NEB-based TS search with automated pair selection and PBC-aware endpoint alignment
- **Flexible API** — high-level runners (`run_go`, `run_go_ts`, …) and low-level control for custom workflows

## Install

Install with exactly one MLIP extra per environment (`[mace]` or `[uma]`):

```bash
pip install "scgo[mace]"   # or: pip install "scgo[uma]"
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

Pass one of four `system_type` values on every run: `gas_cluster`, `surface_cluster`, `gas_cluster_adsorbate`, or `surface_cluster_adsorbate`. See [system types](https://scgo.readthedocs.io/en/latest/api/system_types.html) for when to use each.

Output is written under `{formula}_searches/` (databases, XYZ minima, JSON summaries). See [output files](https://scgo.readthedocs.io/en/latest/quickstart.html#output-files) and [parameters](https://scgo.readthedocs.io/en/latest/parameters.html) for presets and tuning.

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
