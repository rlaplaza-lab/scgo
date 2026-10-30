#!/usr/bin/env python3
"""Pt5 on graphite GO + TS (``surface_cluster``).

Presets keep no-climb NEB and no lattice rotation. This example raises
``max_pairs`` and enables climb so a low-effort run can still land an interior
saddle under the 12 eV spurious-barrier screen.
"""

from __future__ import annotations

import os
from pathlib import Path

from scgo import (
    SurfaceSystemConfig,
    get_low_effort_torchsim_ga_params,
    get_low_effort_ts_search_params,
    make_hopg_5x5_graphite_surface_config,
    run_go_ts,
)

COMPOSITION = "Pt5"
SEED = 42
SYSTEM_TYPE = "surface_cluster"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "results"
OUTPUT_STEM = "pt5_graphite"


def _resolve_output_stem() -> str:
    return (
        os.environ.get("SCGO_EXAMPLE_OUTPUT_STEM", OUTPUT_STEM).strip() or OUTPUT_STEM
    )


# Graphite-supported clusters often need more IDPP-screened pairs than gas:
# many near-copies fail the surface_cluster discontinuous-path gate. Use the
# production NEB step budget so climb can finish on the survivors.
MAX_PAIRS = 24


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
    return go_params


def _build_ts_params(surface_config: SurfaceSystemConfig) -> dict:
    ts_params = get_low_effort_ts_search_params(
        system_type=SYSTEM_TYPE,
        surface_config=surface_config,
        seed=SEED,
    )
    ts_params["max_pairs"] = MAX_PAIRS
    ts_params["neb_climb"] = True
    ts_params["neb_steps"] = 2000
    ts_params["connectivity_factor"] = 1.8
    ts_params["write_timing_json"] = True
    return ts_params


def main() -> None:
    surface_config = make_hopg_5x5_graphite_surface_config()
    run_go_ts(
        COMPOSITION,
        go_params=_build_go_params(surface_config),
        ts_params=_build_ts_params(surface_config),
        seed=SEED,
        verbosity=1,
        output_root=DEFAULT_OUTPUT_ROOT,
        output_stem=_resolve_output_stem(),
        surface_config=surface_config,
        system_type=SYSTEM_TYPE,
    )


if __name__ == "__main__":
    main()
