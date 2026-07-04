#!/usr/bin/env python3
"""Pt5 on graphite: GO + TS via ``run_go_ts``.

``system_type="surface_cluster"`` — supported Pt5 cluster on the preset graphite slab
(no ``adsorbates``).

Requires ``scgo[mace]``. Pass the same ``surface_config`` to the preset builders
and ``run_go_ts`` (values must agree when both are set). See
``docs/source/parameters.rst`` (*Parameter resolution*) for merge rules.

Output: ``results/pt5_graphite_mace/`` with ``Pt5_searches/``, ``Pt5_ts_results/``,
and optional ``go_ts_timing.json`` (see docs quickstart, *On-disk layout*).
"""

from __future__ import annotations

from pathlib import Path

from scgo import (
    SurfaceSystemConfig,
    get_torchsim_ga_params,
    get_ts_search_params,
    make_graphite_surface_config,
    run_go_ts,
)

COMPOSITION = "Pt5"
SEED = 42
SYSTEM_TYPE = "surface_cluster"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "results"
OUTPUT_STEM = "pt5_graphite"

NITER = 6
POPULATION_SIZE = 24
MAX_PAIRS = 10
SLAB_LAYERS = 3


def _build_go_params(surface_config: SurfaceSystemConfig) -> dict:
    go_params = get_torchsim_ga_params(
        system_type=SYSTEM_TYPE,
        surface_config=surface_config,
        seed=SEED,
    )
    go_params["connectivity_factor"] = 1.8
    go_params["optimizer_params"]["ga"].update(
        niter=NITER,
        population_size=POPULATION_SIZE,
        write_timing_json=True,
        detailed_timing=True,
    )
    return go_params


def _build_ts_params(surface_config: SurfaceSystemConfig) -> dict:
    ts_params = get_ts_search_params(
        system_type=SYSTEM_TYPE,
        surface_config=surface_config,
        seed=SEED,
    )
    ts_params["max_pairs"] = MAX_PAIRS
    ts_params["connectivity_factor"] = 1.8
    ts_params["write_timing_json"] = True
    return ts_params


def main() -> None:
    surface_config = make_graphite_surface_config(slab_layers=SLAB_LAYERS)
    run_go_ts(
        COMPOSITION,
        go_params=_build_go_params(surface_config),
        ts_params=_build_ts_params(surface_config),
        seed=SEED,
        verbosity=1,
        output_root=DEFAULT_OUTPUT_ROOT,
        output_stem=OUTPUT_STEM,
        surface_config=surface_config,
        system_type=SYSTEM_TYPE,
    )


if __name__ == "__main__":
    main()
