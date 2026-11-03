# SCGO: Simple Cluster Global Optimization

[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/) [![PyPI](https://img.shields.io/pypi/v/scgo.svg)](https://pypi.org/project/scgo/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![SCGO Logo](docs/source/_static/scgo_logo.svg)

SCGO searches for low-energy arrangements of atoms. That can be a metal cluster, a molecule on a cluster, a cluster on a surface, or the top layers of a slab. The search uses basin hopping or a genetic algorithm. Transition states use nudged elastic band. MACE, UMA, and UPET are optional and run through ASE and TorchSim.

**Documentation:** [Read the Docs](https://scgo.readthedocs.io/)

## Install

The example below needs only the base package. It uses ASE's EMT potential:

```bash
pip install scgo
```

For a machine-learning potential, install one extra. Do not mix `[mace]`, `[uma]`, and `[upet]` in the same environment:

```bash
pip install "scgo[mace]"   # or [uma] / [upet]
```

**UPET only.** After `pip install "scgo[upet]"`, install this neighbor-list package as well:

```bash
pip install 'vesin==0.6.0' --force-reinstall --no-deps
```

Python 3.12+, with SQLite JSON1. Details: [installation guide](https://scgo.readthedocs.io/en/latest/installation.html).

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

for energy, atoms in results:
    print(f"{energy:.3f} eV  {atoms.get_chemical_formula()}")
```

`results` is a list of `(energy, Atoms)` pairs, lowest energy first. The energies are EMT, and the preset is a small search (2 generations, population 5). Files are written to `Pt4_searches/` in the current directory.

`system_type` is required. SCGO picks the algorithm from the number of moving atoms. One or two atoms in vacuum are relaxed. Three atoms and no molecule use basin hopping, in vacuum or on a slab. Every other search uses a genetic algorithm. The other system types are in the [quickstart](https://scgo.readthedocs.io/en/latest/quickstart.html).

## Next

- [A real MACE potential](https://scgo.readthedocs.io/en/latest/quickstart.html#quickstart-mace)
- [A cluster on graphite](https://scgo.readthedocs.io/en/latest/quickstart.html#quickstart-surface)
- [A molecule on the cluster](https://scgo.readthedocs.io/en/latest/quickstart.html#quickstart-molecule)
- [The top layers of a slab](https://scgo.readthedocs.io/en/latest/quickstart.html#quickstart-slab)
- [Transition states](https://scgo.readthedocs.io/en/latest/quickstart.html#quickstart-ts)
- [Many compositions at once](https://scgo.readthedocs.io/en/latest/quickstart.html#quickstart-campaign)

## Examples

[`examples/`](examples/) has short scripts for all six system types. See [`examples/README.md`](examples/README.md).

## Development

```bash
pip install -e ".[mace,dev]"   # or [uma,dev] / [upet,dev]
pre-commit install
pytest tests/ -m "not slow and not integration and not requires_cuda and not requires_upet and not requires_uma"
```

Longer MLIP runs are in [`benchmark/`](benchmark/).

---

MIT License. See [`LICENSE`](LICENSE).
