#!/usr/bin/env python3
"""OH on N-doped graphite GO + TS (``surface_adsorbate``).

Uses ``make_n_doped_graphite_surface_config``.
"""

from __future__ import annotations

import os
from pathlib import Path

from ase import Atoms

from scgo import (
    SurfaceSystemConfig,
    get_low_effort_torchsim_ga_params,
    get_low_effort_ts_search_params,
    make_n_doped_graphite_surface_config,
    run_go_ts,
)

COMPOSITION: list[str] = []
SEED = 42
SYSTEM_TYPE = "surface_adsorbate"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "results"
OUTPUT_STEM = "n_doped_graphite_oh"

# GA/NEB budgets come from the low-effort presets; only the TS pair cap is a
# per-example knob (it is the dominant TS cost lever).
MAX_PAIRS = 4
N_DOPANTS = 2


def _resolve_output_stem() -> str:
    return (
        os.environ.get("SCGO_EXAMPLE_OUTPUT_STEM", OUTPUT_STEM).strip() or OUTPUT_STEM
    )


def _oh_fragment() -> Atoms:
    return Atoms("OH", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.97]])


def _build_go_params(surface_config: SurfaceSystemConfig) -> dict:
    go_params = get_low_effort_torchsim_ga_params(
        system_type=SYSTEM_TYPE,
        surface_config=surface_config,
        seed=SEED,
    )
    go_params["connectivity_factor"] = 1.8
    go_params["optimizer_params"]["ga"].update(
        write_timing_json=True,
        detailed_timing=True,
    )
    go_params["n_jobs"] = -2
    go_params["freeze_adsorbate_internal_geometry"] = True
    return go_params


def _build_ts_params(surface_config: SurfaceSystemConfig) -> dict:
    ts_params = get_low_effort_ts_search_params(
        system_type=SYSTEM_TYPE,
        surface_config=surface_config,
        seed=SEED,
    )
    ts_params["max_pairs"] = MAX_PAIRS
    ts_params["connectivity_factor"] = 1.8
    ts_params["write_timing_json"] = True
    return ts_params


def main() -> None:
    surface_config = make_n_doped_graphite_surface_config(
        n_dopants=N_DOPANTS, seed=SEED
    )
    run_go_ts(
        COMPOSITION,
        go_params=_build_go_params(surface_config),
        ts_params=_build_ts_params(surface_config),
        seed=SEED,
        system_type=SYSTEM_TYPE,
        surface_config=surface_config,
        adsorbates=_oh_fragment(),
        verbosity=1,
        output_root=DEFAULT_OUTPUT_ROOT,
        output_stem=_resolve_output_stem(),
    )


if __name__ == "__main__":
    main()
