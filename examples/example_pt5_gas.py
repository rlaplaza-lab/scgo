#!/usr/bin/env python3
"""Pt5 gas-phase: GO + TS via ``run_go_ts``.

``system_type="gas_cluster"`` — gas-phase cluster only (no slab, no ``adsorbates``).
"""

from __future__ import annotations

from pathlib import Path

from scgo.param_presets import get_torchsim_ga_params, get_ts_search_params
from scgo.runner_api import run_go_ts

N_ATOMS = 5
ELEMENT = "Pt"
SEED = 42
SYSTEM_TYPE = "gas_cluster"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "results"
OUTPUT_STEM = "pt5_gas"

NITER = 10
POPULATION_SIZE = 50
MAX_PAIRS = 15


def _build_go_params() -> dict:
    go_params = get_torchsim_ga_params(system_type=SYSTEM_TYPE, seed=SEED)
    go_params["calculator"] = "MACE"
    go_params["connectivity_factor"] = 1.4
    go_params["optimizer_params"]["ga"].update(
        niter=NITER,
        population_size=POPULATION_SIZE,
    )
    return go_params


def _build_ts_params() -> dict:
    ts_params = get_ts_search_params(system_type=SYSTEM_TYPE, seed=SEED)
    ts_params["max_pairs"] = MAX_PAIRS
    ts_params["connectivity_factor"] = 1.4
    return ts_params


def main() -> None:
    run_go_ts(
        [ELEMENT] * N_ATOMS,
        go_params=_build_go_params(),
        ts_params=_build_ts_params(),
        seed=SEED,
        output_root=DEFAULT_OUTPUT_ROOT,
        output_stem=OUTPUT_STEM,
        system_type=SYSTEM_TYPE,
    )


if __name__ == "__main__":
    main()
