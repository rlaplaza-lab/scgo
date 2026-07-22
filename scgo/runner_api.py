"""High-level SCGO workflows: GO, TS, GO+TS, and campaigns.

``go_params`` = global-optimization params; ``ts_params`` = flat TS preset
(:func:`scgo.param_presets.get_ts_search_params`). The run ``seed`` and
``go_params['seed']`` / ``ts_params['seed']`` must agree when more than one is set
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
``optimizer_params['ga']`` (or ``bh``): ``write_timing_json`` and ``detailed_timing``.
TS uses ``write_timing_json`` in ``ts_params``. ``run_go_ts`` may also write
``go_ts_timing.json`` at the campaign root. See :mod:`scgo.utils.timing_report`.

This module is intentionally thin: the actual implementations live in
:mod:`scgo.runner_composition` (composition parsing), :mod:`scgo.runner_params`
(param merging/allowlisting/coherence and context dataclasses),
:mod:`scgo.runner_go` (GO trials/campaigns), and :mod:`scgo.runner_ts`
(TS and GO+TS pipelines). Everything that was previously importable from
``scgo.runner_api`` remains importable from here, including private helpers
used by tests via ``monkeypatch.setattr("scgo.runner_api.<name>", ...)``
(e.g. ``run_trials``, ``get_calculator_class``, ``_run_go_trials``,
``_run_go_campaign_compositions``, ``_ts_search``, ``_ts_campaign``,
``_run_go_ts_pipeline``). ``scgo.runner_go`` and ``scgo.runner_ts`` call back
into this module (via function-local imports, to avoid a top-level import
cycle) specifically so that patching those attributes here is honored
regardless of which module physically defines the calling code.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from time import perf_counter

from ase import Atoms
from ase.calculators.calculator import Calculator

# Re-exported purely so callers/tests can `from scgo.runner_api import <name>`
# and `monkeypatch.setattr("scgo.runner_api.<name>", ...)`; the redundant
# `as <name>` aliases mark these as intentional re-exports (not unused
# imports) for the linter, since this module's own code does not call them
# directly.
from scgo.minima_search import run_trials as run_trials
from scgo.param_presets import get_default_params as get_default_params
from scgo.exceptions import SCGOValidationError
from scgo.runner_composition import (
    CompositionInput,
    build_one_element_compositions,
    build_two_element_compositions,
    parse_composition_arg,
)
from scgo.runner_composition import _as_composition as _as_composition
from scgo.runner_composition import _as_composition_list as _as_composition_list
from scgo.runner_composition import _compact_formula_error as _compact_formula_error
from scgo.runner_composition import (
    _parse_lowercase_single_element as _parse_lowercase_single_element,
)
from scgo.runner_go import (
    _run_go_campaign_compositions,
    _run_go_trials,
    select_scgo_minima_algorithm,
)
from scgo.runner_go import ScgoMinimaAlgorithm as ScgoMinimaAlgorithm
from scgo.runner_params import RunGOCampaignContext, RunGOContext, resolve_workflow_seed
from scgo.runner_params import RunGOTSContext as RunGOTSContext
from scgo.runner_params import RunTSContext as RunTSContext
from scgo.runner_params import _ALGO_KEYS as _ALGO_KEYS
from scgo.runner_params import _DEFAULT_GO_PARAMS as _DEFAULT_GO_PARAMS
from scgo.runner_params import _as_int_seed as _as_int_seed
from scgo.runner_params import (
    _calculator_slug_from_go_params as _calculator_slug_from_go_params,
)
from scgo.runner_params import _coerce_ts_for_runner as _coerce_ts_for_runner
from scgo.runner_params import _default_go_ts_output_path as _default_go_ts_output_path
from scgo.runner_params import (
    _default_optimizer_system_type as _default_optimizer_system_type,
)
from scgo.runner_params import (
    _log_completion,
    _log_validation_error,
    _prepare_run_go_campaign_context,
    _prepare_run_go_context,
)
from scgo.runner_params import (
    _merge_adsorbate_context_into_params as _merge_adsorbate_context_into_params,
)
from scgo.runner_params import (
    _optimizer_write_timing_json_enabled as _optimizer_write_timing_json_enabled,
)
from scgo.runner_params import _prepare_run_context as _prepare_run_context
from scgo.runner_params import _prepare_run_go_ts_context as _prepare_run_go_ts_context
from scgo.runner_params import (
    _prepare_run_ts_search_context as _prepare_run_ts_search_context,
)
from scgo.runner_params import _reject_system_keys as _reject_system_keys
from scgo.runner_params import _require_system_type as _require_system_type
from scgo.runner_params import _resolve_go_params as _resolve_go_params
from scgo.runner_params import _resolve_go_ts_params as _resolve_go_ts_params
from scgo.runner_params import _resolve_ts_params as _resolve_ts_params
from scgo.runner_params import _resolved_path as _resolved_path
from scgo.runner_params import (
    _validate_go_ts_param_coherence as _validate_go_ts_param_coherence,
)
from scgo.runner_params import (
    _validate_go_ts_surface_config as _validate_go_ts_surface_config,
)
from scgo.runner_params import (
    _with_adsorbate_in_optimizers as _with_adsorbate_in_optimizers,
)
from scgo.runner_params import (
    _with_surface_in_optimizers as _with_surface_in_optimizers,
)
from scgo.runner_params import (
    _with_system_type_in_optimizer_params as _with_system_type_in_optimizer_params,
)
from scgo.runner_ts import (
    log_go_ts_summary,
    run_go_ts,
    run_go_ts_campaign,
    run_ts_campaign,
    run_ts_search,
)
from scgo.runner_ts import _execute_run_go_ts as _execute_run_go_ts
from scgo.runner_ts import _run_go_ts_pipeline as _run_go_ts_pipeline
from scgo.runner_ts import (
    _run_one_element_go_ts_pipeline as _run_one_element_go_ts_pipeline,
)
from scgo.surface.config import SurfaceSystemConfig
from scgo.system_types import AdsorbatesInput, SystemType
from scgo.ts_search.transition_state_run import (
    run_transition_state_campaign as _ts_campaign,  # noqa: F401
)
from scgo.ts_search.transition_state_run import (
    run_transition_state_search as _ts_search,  # noqa: F401
)
from scgo.utils.logging import get_logger
from scgo.utils.run_helpers import get_calculator_class as get_calculator_class

_LOGGER = get_logger(__name__)


def _execute_run_go(context: RunGOContext) -> list[tuple[float, Atoms]]:
    return _run_go_trials(
        context.composition,
        context.system_type,
        params=context.params,
        seed=context.seed,
        verbosity=context.verbosity,
        run_id=context.run_id,
        clean=context.clean,
        output_dir=context.output_dir,
        calculator_for_global_optimization=context.calculator_for_global_optimization,
        params_already_merged=True,
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
    return _run_go_campaign_compositions(
        context.compositions,
        context.system_type,
        params=context.params,
        seed=context.seed,
        verbosity=context.verbosity,
        run_id=context.run_id,
        clean=context.clean,
        output_dir=context.output_dir,
        params_already_merged=True,
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
    ``params['seed']``. If a composition fails (``ValueError``, ``RuntimeError``,
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
