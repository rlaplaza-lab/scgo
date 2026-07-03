#!/usr/bin/env python3
"""Pt5+2OH on graphite: GO + TS via ``run_go_ts``.

``system_type="surface_cluster_adsorbate"``: core-only ``COMPOSITION`` plus two
``adsorbates`` OH fragments. Pass the same ``surface_config`` to ``run_go_ts``
and the preset builders.

Workflow: build Pt5 core, place each OH on distinct hull sites, deposit the
combined cluster on graphite with surface-biased orientation, then run tag-aware
GA (core crossover, ``fragment_reposition`` for adsorbate diversity). See
``docs/source/api/system_types.rst`` for operator details.
"""

from __future__ import annotations

from pathlib import Path

from ase import Atoms

from scgo import (
    SurfaceSystemConfig,
    get_torchsim_ga_params,
    get_ts_search_params,
    make_graphite_surface_config,
    run_go_ts,
)

COMPOSITION = "Pt5"
SEED = 42
SYSTEM_TYPE = "surface_cluster_adsorbate"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "results"
OUTPUT_STEM = "pt5_2oh_graphite"

NITER = 6
POPULATION_SIZE = 24
MAX_PAIRS = 10
SLAB_LAYERS = 3
ADSORBATES = [
    Atoms(symbols=["O", "H"], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.96]]),
    Atoms(symbols=["O", "H"], positions=[[2.2, 0.0, 0.0], [2.2, 0.0, 0.96]]),
]


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
    go_params["freeze_adsorbate_internal_geometry"] = True
    return go_params


def _build_ts_params(surface_config: SurfaceSystemConfig) -> dict:
    ts_params = get_ts_search_params(
        system_type=SYSTEM_TYPE,
        surface_config=surface_config,
        seed=SEED,
    )
    ts_params["max_pairs"] = MAX_PAIRS
    ts_params["energy_gap_threshold"] = 1.0
    ts_params["neb_n_images"] = 7
    ts_params["neb_steps"] = 800
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
        adsorbates=ADSORBATES,
    )


if __name__ == "__main__":
    main()
