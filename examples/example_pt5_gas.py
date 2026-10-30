#!/usr/bin/env python3
"""Pt5 gas-phase GO + TS (``gas_cluster``).

Presets keep no-climb NEB. This example only sets ``max_pairs``.
"""

from __future__ import annotations

import os
from pathlib import Path

from scgo import (
    get_low_effort_torchsim_ga_params,
    get_low_effort_ts_search_params,
    run_go_ts,
)

COMPOSITION = "Pt5"
SEED = 42
SYSTEM_TYPE = "gas_cluster"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "results"
OUTPUT_STEM = "pt5_gas"


def _resolve_output_stem() -> str:
    return (
        os.environ.get("SCGO_EXAMPLE_OUTPUT_STEM", OUTPUT_STEM).strip() or OUTPUT_STEM
    )


# GA/NEB budgets come from the low-effort presets; only the TS pair cap is a
# per-example knob (it is the dominant TS cost lever).
MAX_PAIRS = 6


def _build_go_params() -> dict:
    go_params = get_low_effort_torchsim_ga_params(system_type=SYSTEM_TYPE, seed=SEED)
    go_params["optimizer_params"]["ga"].update(
        write_timing_json=True,
        detailed_timing=True,
    )
    go_params["n_jobs"] = -2
    return go_params


def _build_ts_params() -> dict:
    ts_params = get_low_effort_ts_search_params(system_type=SYSTEM_TYPE, seed=SEED)
    ts_params["max_pairs"] = MAX_PAIRS
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
    )


if __name__ == "__main__":
    main()
