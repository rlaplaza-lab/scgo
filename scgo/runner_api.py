"""High-level SCGO workflows: GO, TS, GO+TS, and campaigns.

``go_params`` = global-optimization params; ``ts_params`` = flat TS preset
(:func:`scgo.param_presets.get_ts_search_params`). The run ``seed`` and
``go_params["seed"]`` / ``ts_params["seed"]`` must agree when more than one is set
(:func:`resolve_workflow_seed`). System mode is set only by the run function
``system_type=...`` argument together with explicit ``surface_config=...`` and,
for ``*_adsorbate`` modes, core-only ``composition`` plus ``adsorbates=...``
(single or multiple ASE ``Atoms`` fragments).
System-definition keys in ``go_params`` are partly restricted:
``system_type`` remains rejected, while top-level ``surface_config`` is allowed
and fanned out into optimizer slots. Adsorbate placement tuning
(``cluster_adsorbate_config``, ``connectivity_factor``, ``freeze_adsorbate_internal_geometry``)
belongs in ``go_params`` only—not as separate ``run_*`` keywords. For
``ts_params``, ``system_type`` remains rejected while ``surface_config`` is
allowed and validated against the run argument.

GA/BH timing JSON is configured in ``params``/``go_params`` under
``optimizer_params["ga"]`` (or ``bh``): ``write_timing_json`` and ``detailed_timing``.
TS uses ``write_timing_json`` in ``ts_params``. ``run_go_ts`` may also write
``go_ts_timing.json`` at the campaign root. See :mod:`scgo.utils.timing_report`.

This module is intentionally thin: the actual implementations live in
:mod:`scgo.runner_composition` (composition parsing), :mod:`scgo.runner_params`
(param merging/allowlisting/coherence and context dataclasses),
:mod:`scgo.runner_go` (GO trials/campaigns), and :mod:`scgo.runner_ts`
(TS and GO+TS pipelines). Imports flow one way only
(``runner_api -> runner_go``/``runner_ts -> ...``), so the runner import graph
is acyclic and each helper is called through the module that defines it.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from time import perf_counter

from ase import Atoms
from ase.calculators.calculator import Calculator

from scgo import runner_go
from scgo.exceptions import SCGOValidationError
from scgo.runner_composition import (
    CompositionInput,
    build_one_element_compositions,
    build_two_element_compositions,
    parse_composition_arg,
)
from scgo.runner_go import select_scgo_minima_algorithm
from scgo.runner_params import (
    RunGOCampaignContext,
    RunGOContext,
    _log_completion,
    _log_validation_error,
    _prepare_run_go_campaign_context,
    _prepare_run_go_context,
    resolve_workflow_seed,
)
from scgo.runner_ts import (
    log_go_ts_summary,
    run_go_ts,
    run_go_ts_campaign,
    run_ts_campaign,
    run_ts_search,
)
from scgo.surface.config import SurfaceSystemConfig
from scgo.system_types import AdsorbatesInput, SystemType
from scgo.utils.logging import get_logger

logger = get_logger(__name__)


def _execute_run_go(context: RunGOContext) -> list[tuple[float, Atoms]]:
    return runner_go._run_go_trials(
        context.composition,
        context.system_type,
        params=context.params,
        seed=context.seed,
        verbosity=context.verbosity,
        run_id=context.run_id,
        clean=context.clean,
        output_dir=context.output_dir,
        calculator_for_global_optimization=context.calculator_for_global_optimization,
    )


def run_go(
    composition: CompositionInput,
    params: dict | None = None,
    seed: int | None = None,
    verbosity: int = 1,
    run_id: str | None = None,
    clean: bool = False,
    output_dir: str | Path | None = None,
    calculator_for_global_optimization: Calculator | None = None,
    surface_config: SurfaceSystemConfig | None = None,
    system_type: SystemType | None = None,
    adsorbates: AdsorbatesInput | None = None,
    log_summary: bool = True,
) -> list[tuple[float, Atoms]]:
    """Run global optimization trials for one composition."""
    try:
        context = _prepare_run_go_context(
            composition,
            params=params,
            seed=seed,
            verbosity=verbosity,
            run_id=run_id,
            clean=clean,
            output_dir=output_dir,
            calculator_for_global_optimization=calculator_for_global_optimization,
            surface_config=surface_config,
            system_type=system_type,
            adsorbates=adsorbates,
        )
    except SCGOValidationError as exc:
        _log_validation_error(exc)
        raise
    t0 = perf_counter()
    minima = _execute_run_go(context)
    if log_summary:
        _log_completion(
            "run_go",
            elapsed_s=perf_counter() - t0,
            details=f"minima={len(minima)} output_dir={context.output_summary_dir}",
        )
    return minima


def _execute_run_go_campaign(
    context: RunGOCampaignContext,
) -> dict[str, list[tuple[float, Atoms]]]:
    return runner_go._run_go_campaign_compositions(
        context.compositions,
        context.system_type,
        params=context.params,
        seed=context.seed,
        verbosity=context.verbosity,
        run_id=context.run_id,
        clean=context.clean,
        output_dir=context.output_dir,
    )


def run_go_campaign(
    compositions: Iterable[CompositionInput],
    params: dict | None = None,
    seed: int | None = None,
    verbosity: int = 1,
    run_id: str | None = None,
    clean: bool = False,
    output_dir: str | Path | None = None,
    surface_config: SurfaceSystemConfig | None = None,
    system_type: SystemType | None = None,
    adsorbates: AdsorbatesInput | None = None,
    log_summary: bool = True,
) -> dict[str, list[tuple[float, Atoms]]]:
    """Run global optimization for multiple compositions.

    Each composition gets a reproducible sub-seed derived from ``seed`` /
    ``params["seed"]``. If a composition fails (``ValueError``, ``RuntimeError``,
    ``SCGOValidationError``, I/O, or database errors), the error is logged, that
    formula maps to an empty list, and remaining compositions continue.
    """
    try:
        context = _prepare_run_go_campaign_context(
            compositions,
            params=params,
            seed=seed,
            verbosity=verbosity,
            run_id=run_id,
            clean=clean,
            output_dir=output_dir,
            surface_config=surface_config,
            system_type=system_type,
            adsorbates=adsorbates,
        )
    except SCGOValidationError as exc:
        _log_validation_error(exc)
        raise
    t0 = perf_counter()
    campaign = _execute_run_go_campaign(context)
    if log_summary:
        _log_completion(
            "run_go_campaign",
            elapsed_s=perf_counter() - t0,
            details=f"compositions={len(campaign)} output_dir={context.output_summary_dir}",
        )
    return campaign


__all__ = [
    "CompositionInput",
    "build_one_element_compositions",
    "build_two_element_compositions",
    "log_go_ts_summary",
    "parse_composition_arg",
    "resolve_workflow_seed",
    "run_go",
    "run_go_campaign",
    "run_go_ts",
    "run_go_ts_campaign",
    "run_ts_campaign",
    "run_ts_search",
    "select_scgo_minima_algorithm",
]
