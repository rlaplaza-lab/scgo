#!/usr/bin/env python3
"""Pt5 on graphite: GO + TS via ``run_go_ts``.

``system_type="surface_cluster"`` — supported Pt5 cluster on the preset graphite slab
(no ``adsorbates``).

Requires ``scgo[mace]``. Pass the same ``surface_config`` to the preset builders
and ``run_go_ts`` as a top-level value (they must agree when both are set).
See ``docs/source/parameters.rst`` (*Parameter resolution*) for merge rules.
Params come from the reduced-budget (~25% of production)
:func:`~scgo.param_presets.get_low_effort_torchsim_ga_params` /
:func:`~scgo.param_presets.get_low_effort_ts_search_params`, which keep the
production calculator and NEB physics but shrink the GA and NEB step budgets.

TS: bare surface presets keep no-climb NEB, shared ``neb_fmax=0.20``, spring
``0.1``, 5 images, MIC + cell remap + lattice rotation, and parallel NEB.
This example only sets ``max_pairs`` and ``connectivity_factor``.

Output: ``results/pt5_graphite_mace/`` with ``Pt5_graphite_searches/``,
``Pt5_graphite_ts_results/``, and optional ``go_ts_timing.json`` (see docs
quickstart, *On-disk layout*).
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
    """Prefer ``SCGO_EXAMPLE_OUTPUT_STEM`` for clean end-to-end runs."""
    return (
        os.environ.get("SCGO_EXAMPLE_OUTPUT_STEM", OUTPUT_STEM).strip() or OUTPUT_STEM
    )


# GA/NEB budgets come from the low-effort presets; only the TS pair cap is a
# per-example knob (it is the dominant TS cost lever).
MAX_PAIRS = 6


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
    go_params[
        "n_jobs"
    ] = -2  # one switch parallelizes population init, offspring, and validation
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
