#!/usr/bin/env python3
"""Pt5 gas-phase: GO + TS via ``run_go_ts``.

``system_type="gas_cluster"`` — gas-phase cluster only (no slab, no ``adsorbates``).

Requires ``scgo[mace]`` (MACE + TorchSim). Start from
:func:`~scgo.param_presets.get_torchsim_ga_params` and
:func:`~scgo.param_presets.get_ts_search_params`, override keys, then pass the
dicts as ``go_params`` / ``ts_params``. Runners deep-merge partial dicts with
preset defaults at call time. Pass ``system_type`` on the ``run_go_ts`` call
(not inside the param dicts). Keep ``seed`` consistent across ``seed=``,
``go_params['seed']``, and ``ts_params['seed']``.
"""

from __future__ import annotations

from pathlib import Path

from scgo import get_torchsim_ga_params, get_ts_search_params, run_go_ts

COMPOSITION = "Pt5"
SEED = 42
SYSTEM_TYPE = "gas_cluster"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "results"
OUTPUT_STEM = "pt5_gas"

NITER = 10
POPULATION_SIZE = 50
MAX_PAIRS = 15


def _build_go_params() -> dict:
    go_params = get_torchsim_ga_params(system_type=SYSTEM_TYPE, seed=SEED)
    go_params["optimizer_params"]["ga"].update(
        niter=NITER,
        population_size=POPULATION_SIZE,
        write_timing_json=True,
        detailed_timing=True,
    )
    return go_params


def _build_ts_params() -> dict:
    ts_params = get_ts_search_params(system_type=SYSTEM_TYPE, seed=SEED)
    ts_params["max_pairs"] = MAX_PAIRS
    return ts_params


def main() -> None:
    run_go_ts(
        COMPOSITION,
        go_params=_build_go_params(),
        ts_params=_build_ts_params(),
        seed=SEED,
        verbosity=1,
        output_root=DEFAULT_OUTPUT_ROOT,
        output_stem=OUTPUT_STEM,
        system_type=SYSTEM_TYPE,
    )


if __name__ == "__main__":
    main()
