# SCGO: Simple Cluster Global Optimization

[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/) [![PyPI](https://img.shields.io/pypi/v/scgo.svg)](https://pypi.org/project/scgo/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![SCGO Logo](docs/source/_static/scgo_logo.svg)

Global optimization of atomic clusters with ASE: Basin Hopping, Genetic Algorithms, NEB transition-state search, and MLIPs (MACE, UMA, UPET) via TorchSim. Covers gas-phase, surface, and adsorbate workflows.

**Documentation:** [Read the Docs](https://scgo.readthedocs.io/)

## Install

Exactly one MLIP extra per environment:

```bash
pip install "scgo[mace]"   # or [uma] / [upet]
# UPET only: pip install 'vesin==0.6.0' --force-reinstall --no-deps
```

Python 3.12+, SQLite JSON1. Details: [installation guide](https://scgo.readthedocs.io/en/latest/installation.html).

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

`results` is a list of `(energy, Atoms)` unique minima (energy-sorted). Use `run_go_campaign` for multi-composition runs.

## Workflows

| Goal | Entry point |
|------|-------------|
| Single composition GO | `run_go` |
| Multi-composition GO | `run_go_campaign` |
| TS from existing minima | `run_ts_search` |
| GO then TS | `run_go_ts` |
| Multi-composition TS / GO+TS | `run_ts_campaign` / `run_go_ts_campaign` |

`system_type` is always a run argument: `gas_cluster`, `surface_cluster`, `gas_cluster_adsorbate`, or `surface_cluster_adsorbate`. Surfaces need `surface_config=`; adsorbates need `adsorbates=`.

Output: `run_go` writes `{formula}_searches/`; GO+TS/TS use a campaign root with sibling `{formula}_searches/` and `{formula}_ts_results/`. See [quickstart](https://scgo.readthedocs.io/en/latest/quickstart.html).

## Examples

[`examples/`](examples/) — MACE + TorchSim smoke scripts for all four system types (`example_pt5_*.py`).

## Development

```bash
pip install -e ".[mace,dev]"   # or [uma,dev] / [upet,dev]
pre-commit install
pytest tests/ -m "not slow"
```

Long MLIP sweeps: [`benchmark/`](benchmark/).

---

MIT License — see [`LICENSE`](LICENSE).
