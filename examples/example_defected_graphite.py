#!/usr/bin/env python3
"""Defected graphite surface search via ``run_go_ts``.

``system_type="surface"`` — GA/BH act on the top slab layer of a vacancy-defected
graphite slab (bottom layers fixed). Requires ``scgo[mace]``.
Params come from the reduced-budget (~25% of production)
:func:`~scgo.param_presets.get_low_effort_torchsim_ga_params` /
:func:`~scgo.param_presets.get_low_effort_ts_search_params`, which keep the
production calculator and NEB physics but shrink the GA and NEB step budgets.

Output: ``results/defected_graphite_mace/`` with searches / TS result trees.
"""

from __future__ import annotations

import os
from pathlib import Path

from scgo import (
    SurfaceSystemConfig,
    get_low_effort_torchsim_ga_params,
    get_low_effort_ts_search_params,
    make_defected_graphite_surface_config,
    run_go_ts,
)

COMPOSITION: list[str] = []
SEED = 42
SYSTEM_TYPE = "surface"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "results"
OUTPUT_STEM = "defected_graphite"

# GA/NEB budgets come from the low-effort presets; only the TS pair cap is a
# per-example knob (it is the dominant TS cost lever).
MAX_PAIRS = 4
SLAB_LAYERS = 3
SLAB_REPEAT_XY = 3
N_VACANCIES = 1


def _resolve_output_stem() -> str:
    return (
        os.environ.get("SCGO_EXAMPLE_OUTPUT_STEM", OUTPUT_STEM).strip() or OUTPUT_STEM
    )


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
    surface_config = make_defected_graphite_surface_config(
        slab_layers=SLAB_LAYERS,
        slab_repeat_xy=SLAB_REPEAT_XY,
        n_vacancies=N_VACANCIES,
        seed=SEED,
    )
    run_go_ts(
        COMPOSITION,
        go_params=_build_go_params(surface_config),
        ts_params=_build_ts_params(surface_config),
        seed=SEED,
        system_type=SYSTEM_TYPE,
        surface_config=surface_config,
        verbosity=1,
        output_root=DEFAULT_OUTPUT_ROOT,
        output_stem=_resolve_output_stem(),
    )


if __name__ == "__main__":
    main()
