#!/usr/bin/env python3
"""Pt5+OH gas-phase: GO + TS via ``run_go_ts``.

``system_type="gas_cluster_adsorbate"``: core-only ``COMPOSITION`` plus one
``adsorbates`` ASE ``Atoms`` fragment.

Initialization places OH on convex-hull sites of the Pt core (ranked by steric
deficit). The GA preserves intra-fragment bonds via tag-rigid operators; crossover
splices the core only. Optional tuning in ``go_params``:

- ``connectivity_factor`` — validation threshold (default 1.4 scale on covalent radii)
- ``cluster_adsorbate_config`` — placement height range, retries, clash checks
- ``freeze_adsorbate_internal_geometry=True`` — strict Kabsch restore (this example
  enables it; default is ``False`` and still keeps fragments rigid as units)
"""

from __future__ import annotations

from pathlib import Path

from ase import Atoms

from scgo.param_presets import get_torchsim_ga_params, get_ts_search_params
from scgo.runner_api import run_go_ts

COMPOSITION = ["Pt", "Pt", "Pt", "Pt", "Pt"]
SEED = 42
SYSTEM_TYPE = "gas_cluster_adsorbate"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "results"
OUTPUT_STEM = "pt5_oh_gas"

NITER = 8
POPULATION_SIZE = 40
MAX_PAIRS = 12
ADSORBATES = Atoms(
    symbols=["O", "H"],
    positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.96]],
)


def _build_go_params() -> dict:
    go_params = get_torchsim_ga_params(system_type=SYSTEM_TYPE, seed=SEED)
    go_params["calculator"] = "MACE"
    go_params["connectivity_factor"] = 1.8
    go_params["optimizer_params"]["ga"].update(
        niter=NITER,
        population_size=POPULATION_SIZE,
    )
    # Optional: strict template restore after mutations (default False uses tag-rigid GA)
    go_params["freeze_adsorbate_internal_geometry"] = True
    return go_params


def _build_ts_params() -> dict:
    ts_params = get_ts_search_params(system_type=SYSTEM_TYPE, seed=SEED)
    ts_params["max_pairs"] = MAX_PAIRS
    ts_params["connectivity_factor"] = 1.8
    return ts_params


def main() -> None:
    run_go_ts(
        COMPOSITION,
        go_params=_build_go_params(),
        ts_params=_build_ts_params(),
        seed=SEED,
        output_root=DEFAULT_OUTPUT_ROOT,
        output_stem=OUTPUT_STEM,
        system_type=SYSTEM_TYPE,
        adsorbates=ADSORBATES,
    )


if __name__ == "__main__":
    main()
