#!/usr/bin/env python3
"""Pt5+OH gas-phase GO + TS (``gas_cluster_adsorbate``).

Sets ``freeze_adsorbate_internal_geometry=True`` (the default is ``False``).
"""

from __future__ import annotations

import os
from pathlib import Path

from ase import Atoms

from scgo import (
    get_low_effort_torchsim_ga_params,
    get_low_effort_ts_search_params,
    run_go_ts,
)

COMPOSITION = "Pt5"
SEED = 42
SYSTEM_TYPE = "gas_cluster_adsorbate"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "results"
OUTPUT_STEM = "pt5_oh_gas"


def _resolve_output_stem() -> str:
    return (
        os.environ.get("SCGO_EXAMPLE_OUTPUT_STEM", OUTPUT_STEM).strip() or OUTPUT_STEM
    )


# GA/NEB budgets come from the low-effort presets. More pairs than the
# surface-adsorbate example: gas OH hops need a wider IDPP pool and the cell is
# small, so each band is cheap.
MAX_PAIRS = 6
ADSORBATES = Atoms(
    symbols=["O", "H"],
    positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.96]],
)


def _build_go_params() -> dict:
    go_params = get_low_effort_torchsim_ga_params(system_type=SYSTEM_TYPE, seed=SEED)
    go_params["connectivity_factor"] = 1.8
    go_params["optimizer_params"]["ga"].update(
        write_timing_json=True,
        detailed_timing=True,
    )
    go_params["n_jobs"] = -2
    go_params["freeze_adsorbate_internal_geometry"] = True
    return go_params


def _build_ts_params() -> dict:
    ts_params = get_low_effort_ts_search_params(system_type=SYSTEM_TYPE, seed=SEED)
    ts_params["max_pairs"] = MAX_PAIRS
    ts_params["connectivity_factor"] = 1.8
    ts_params["write_timing_json"] = True
    return ts_params


def main() -> None:
    run_go_ts(
        COMPOSITION,
        go_params=_build_go_params(),
        ts_params=_build_ts_params(),
        seed=SEED,
        verbosity=1,
        output_root=DEFAULT_OUTPUT_ROOT,
        output_stem=_resolve_output_stem(),
        system_type=SYSTEM_TYPE,
        adsorbates=ADSORBATES,
    )


if __name__ == "__main__":
    main()
