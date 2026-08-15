# SCGO: Simple Cluster Global Optimization

[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/) [![PyPI](https://img.shields.io/pypi/v/scgo.svg)](https://pypi.org/project/scgo/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![SCGO Logo](docs/source/_static/scgo_logo.svg)

Global optimization of atomic clusters with ASE: Basin Hopping, Genetic Algorithms, NEB transition-state search, and MLIPs (MACE, UMA, UPET) via TorchSim. Supports six system types: `gas_cluster`, `surface_cluster`, `gas_cluster_adsorbate`, `surface_cluster_adsorbate`, `surface`, and `surface_adsorbate`.

**Documentation:** [Read the Docs](https://scgo.readthedocs.io/)

## Install

Exactly one MLIP extra per environment:

```bash
pip install "scgo[mace]"   # or [uma] / [upet]
```

**UPET note:** After `pip install "scgo[upet]"`, manually install `vesin==0.6.0`:
```bash
pip install 'vesin==0.6.0' --force-reinstall --no-deps
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

`results` is a list of `(energy, Atoms)` unique minima (energy-sorted).

**Algorithms:** SCGO auto-selects based on system size:
- ≤2 mobile atoms: Simple relaxation
- 3 atoms, no adsorbate: Basin Hopping
- 3+ atoms with adsorbate: Genetic Algorithm  
- ≥4 atoms: Genetic Algorithm

## Workflows

| Goal | Entry point |
|------|-------------|
| Single composition GO | `run_go` |
| Multi-composition GO | `run_go_campaign` |
| TS from existing minima | `run_ts_search` |
| GO then TS | `run_go_ts` |
| Multi-composition TS / GO+TS | `run_ts_campaign` / `run_go_ts_campaign` |

`system_type` is always a run argument (never inside `go_params` /
`optimizer_params` slots). Surfaces need `surface_config=`, adsorbates need
`adsorbates=`. Top-level `surface_config` in presets must agree with the run
argument when both are set.

**Output:** `run_go` writes `{path_key}_searches/` with datetime-tagged `run_*/` subdirectories. GO+TS creates sibling `{path_key}_ts_results/`. The `path_key` combines nanoparticle formula, adsorbate fragments, and surface name (e.g., `Pt5`, `Pt5_OH_OH_graphite`). See [quickstart](https://scgo.readthedocs.io/en/latest/quickstart.html).

## Examples

[`examples/`](examples/) — MACE + TorchSim smoke scripts for all six system types. See [`examples/README.md`](examples/README.md) for details.

## Development

```bash
pip install -e ".[mace,dev]"   # or [uma,dev] / [upet,dev]
pre-commit install
pytest tests/ -m "not slow and not integration and not requires_cuda and not requires_upet and not requires_uma"
```

Long MLIP sweeps: [`benchmark/`](benchmark/).

---

MIT License — see [`LICENSE`](LICENSE).
