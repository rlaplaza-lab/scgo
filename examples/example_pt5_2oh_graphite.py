#!/usr/bin/env python3
"""Pt5+2OH on graphite: GO + TS via ``run_go_ts``.

``system_type="surface_cluster_adsorbate"``: core-only ``COMPOSITION`` plus two
``adsorbates`` OH fragments. Pass the same ``surface_config`` to ``run_go_ts``
and the preset builders.

Workflow: build Pt5 core, place each OH on distinct hull sites, deposit the
combined cluster on graphite with surface-biased orientation, then run tag-aware
GA (core crossover, ``fragment_reposition`` for adsorbate diversity). See
``docs/source/api/system_types.rst`` for operator details.
Params come from the reduced-budget (~25% of production)
:func:`~scgo.param_presets.get_low_effort_torchsim_ga_params` /
:func:`~scgo.param_presets.get_low_effort_ts_search_params`, which keep the
production calculator and NEB physics but shrink the GA and NEB step budgets.

TS: surface-adsorbate presets supply climb, spring ``0.5``, shared
``neb_fmax=0.20``, 7 images, parallel NEB, ``max_endpoint_mismatch=1.5`` Å,
``energy_gap_threshold=0.75``, and IDPP-profile pair ranking (prefer robust
interior maxima). This example only tightens ``max_pairs``.

Output: ``results/pt5_2oh_graphite_mace/`` with ``Pt5_OH_OH_graphite_searches/``,
``Pt5_OH_OH_graphite_ts_results/``, and optional ``go_ts_timing.json`` (see docs
quickstart, *On-disk layout*).
"""

from __future__ import annotations

import os
from pathlib import Path

from ase import Atoms

from scgo import (
    SurfaceSystemConfig,
    get_low_effort_torchsim_ga_params,
    get_low_effort_ts_search_params,
    make_graphite_surface_config,
    run_go_ts,
)

COMPOSITION = "Pt5"
SEED = 42
SYSTEM_TYPE = "surface_cluster_adsorbate"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "results"
OUTPUT_STEM = "pt5_2oh_graphite"


def _resolve_output_stem() -> str:
    """Prefer ``SCGO_EXAMPLE_OUTPUT_STEM`` for clean end-to-end runs."""
    return (
        os.environ.get("SCGO_EXAMPLE_OUTPUT_STEM", OUTPUT_STEM).strip() or OUTPUT_STEM
    )


# GA/NEB budgets come from the low-effort presets. Fewer close pairs here:
# adsorbate bands are the most expensive (7 images, two-stage climb).
MAX_PAIRS = 4
SLAB_LAYERS = 3
SLAB_REPEAT_XY = 3
ADSORBATES = [
    Atoms(symbols=["O", "H"], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.96]]),
    Atoms(symbols=["O", "H"], positions=[[2.2, 0.0, 0.0], [2.2, 0.0, 0.96]]),
]


def _build_go_params(surface_config: SurfaceSystemConfig) -> dict:
    go_params = get_low_effort_torchsim_ga_params(
        system_type=SYSTEM_TYPE,
        surface_config=surface_config,
        seed=SEED,
    )
    go_params["connectivity_factor"] = 1.8
    go_params["optimizer_params"]["ga"].update(
        n_jobs_population_init=-2,   # all but one CPU
        n_jobs_offspring=-2,
        write_timing_json=True,
        detailed_timing=True,
    )
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
    surface_config = make_graphite_surface_config(
        slab_layers=SLAB_LAYERS,
        slab_repeat_xy=SLAB_REPEAT_XY,
    )
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
        adsorbates=ADSORBATES,
    )


if __name__ == "__main__":
    main()
