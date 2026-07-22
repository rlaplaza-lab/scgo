"""Transition-state (TS) and GO+TS pipeline runners.

Implements the GO->TS pipeline plumbing and the public ``run_go_ts`` /
``run_go_ts_campaign`` / ``run_ts_search`` / ``run_ts_campaign`` API.

Note on the local ``scgo.runner_api`` imports scattered through this module:
``scgo.runner_api`` re-exports ``_run_go_trials``, ``get_calculator_class``,
``_ts_search``, and ``_ts_campaign`` as its own module attributes specifically
so tests can do e.g. ``monkeypatch.setattr("scgo.runner_api._ts_search", ...)``.
Since ``scgo.runner_api`` imports from this module at top level, importing it
back here at module load time would be circular; calls to those names are
therefore routed through a function-local import of ``scgo.runner_api`` so a
patched attribute on ``scgo.runner_api`` is honored regardless of where the
call originates.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from time import perf_counter
from typing import Any

from scgo.cluster_adsorbate.config import ClusterAdsorbateConfig
from scgo.exceptions import SCGOValidationError
from scgo.runner_composition import CompositionInput, _as_composition_list
from scgo.runner_params import (
    RunGOTSContext,
    _coerce_ts_for_runner,
    _default_go_ts_output_path,
    _log_completion,
    _log_validation_error,
    _merge_adsorbate_context_into_params,
    _optimizer_write_timing_json_enabled,
    _prepare_run_go_ts_context,
    _prepare_run_ts_search_context,
    _reject_system_keys,
    _require_system_type,
    _resolve_go_ts_params,
    _resolve_ts_params,
    _resolved_path,
    _validate_go_ts_param_coherence,
    _validate_go_ts_surface_config,
    _with_surface_in_optimizers,
    _with_system_type_in_optimizer_params,
    resolve_workflow_seed,
)
from scgo.surface.config import SurfaceSystemConfig
from scgo.system_types import (
    AdsorbateDefinition,
    AdsorbatesInput,
    SystemType,
    extract_adsorbate_definition_from_params,
    resolve_adsorbate_run_composition,
    resolve_connectivity_factor,
    validate_system_type_settings,
)
from scgo.utils.helpers import get_cluster_formula
from scgo.utils.logging import configure_logging, get_logger
from scgo.utils.output_paths import resolve_go_ts_pipeline_paths
from scgo.utils.run_helpers import (
    cleanup_torch_cuda,
    initialize_ts_params,
    log_ts_configuration,
)
from scgo.utils.run_tracking import get_run_directories, get_run_id_from_dir
from scgo.utils.timing_report import (
    GO_TS_TIMING_JSON_FILENAME,
    build_timing_payload,
    log_timing_summary,
    sum_neb_seconds_from_ts_results,
    write_timing_file,
)
from scgo.utils.validation import validate_composition

_LOGGER = get_logger(__name__)


def _run_go_ts_pipeline(
    composition: list[str],
    system_type: SystemType,
    *,
    go_params: dict[str, Any],
    ts_kwargs: dict[str, Any],
    adsorbate_definition: AdsorbateDefinition | None = None,
    seed: int | None = None,
    verbosity: int = 1,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run global optimization then transition-state search; return a compact run summary.

    ``go_params`` is the same global-optimization dict as ``run_go`` / ``run_go_ts``'s
    ``go_params=``. Minima and TS artifacts are sibling ``{formula}_searches/`` and
    ``{formula}_ts_results/`` directories under ``output_path`` (see
    :mod:`scgo.utils.output_paths`).
    ``adsorbate_definition`` (when provided) is forwarded to TS search so endpoint
    alignment can use explicit core/adsorbate block sizes.
    For high-level entry points see :mod:`scgo.runner_api`.
    """
    from scgo import runner_api as _runner_api

    configure_logging(verbosity)
    logger = get_logger(__name__)

    validate_composition(composition, allow_empty=False, allow_tuple=False)

    formula = get_cluster_formula(composition)
    output_path = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else _default_go_ts_output_path(
            composition,
            go_params=go_params,
            output_stem=formula,
            output_root=None,
        )
    )
    output_path.mkdir(parents=True, exist_ok=True)
    searches_dir, ts_results_dir = resolve_go_ts_pipeline_paths(output_path, formula)

    pipeline_t0 = perf_counter()
    merged_ga = go_params
    calculator_name = merged_ga["calculator"]
    calculator_kwargs = merged_ga.get("calculator_kwargs", {})
    _ = _runner_api.get_calculator_class(calculator_name)
    calculator_for_global_optimization = _runner_api.get_calculator_class(
        calculator_name
    )(
        **calculator_kwargs,
    )
    try:
        go_t0 = perf_counter()
        minima_list = _runner_api._run_go_trials(
            composition,
            system_type,
            params=merged_ga,
            seed=seed,
            verbosity=verbosity,
            output_dir=str(searches_dir),
            calculator_for_global_optimization=calculator_for_global_optimization,
            params_already_merged=True,
        )
    finally:
        go_wall_s = perf_counter() - go_t0
        del calculator_for_global_optimization
        cleanup_torch_cuda(logger=logger)

    minima_by_formula = {formula: minima_list}

    ts_kwargs_local = dict(ts_kwargs)
    ts_kwargs_local.pop("base_dir", None)
    ts_kwargs_local.pop("seed", None)
    ts_kwargs_local.pop("verbosity", None)
    ts_kwargs_local.pop("system_type", None)
    write_ts_json = bool(ts_kwargs_local.pop("write_timing_json", False))

    connectivity_factor_raw: float | None = ts_kwargs_local.pop(
        "connectivity_factor", None
    )
    surface_config_ts = ts_kwargs_local.get("surface_config")
    surface_cfg = (
        surface_config_ts
        if isinstance(surface_config_ts, SurfaceSystemConfig)
        else None
    )
    cluster_cfg = go_params.get("cluster_adsorbate_config")
    if not isinstance(cluster_cfg, ClusterAdsorbateConfig):
        cluster_cfg = None
    connectivity_factor = resolve_connectivity_factor(
        connectivity_factor_raw,
        cluster_adsorbate_config=cluster_cfg,
        surface_config=surface_cfg,
    )

    ts_results = _runner_api._ts_search(
        composition,
        output_dir=output_path,
        seed=seed,
        verbosity=verbosity,
        write_timing_json=write_ts_json,
        connectivity_factor=connectivity_factor,
        adsorbate_definition=adsorbate_definition,
        system_type=system_type,
        **ts_kwargs_local,
    )
    ts_success = sum(1 for result in ts_results if result.get("status") == "success")

    ts_neb = sum_neb_seconds_from_ts_results(ts_results)
    elapsed_s = perf_counter() - pipeline_t0
    go_ts_timings: dict[str, float] = {
        "total_wall_s": elapsed_s,
        "go_phase_s": go_wall_s,
        "ts_neb_sum_s": ts_neb,
        "cpu_non_relax_s": max(0.0, elapsed_s - go_wall_s - ts_neb),
    }
    log_timing_summary(logger, "go_ts", go_ts_timings, verbosity=verbosity)
    write_go_json = _optimizer_write_timing_json_enabled(merged_ga)
    if write_ts_json or write_go_json:
        go_run_dirs = get_run_directories(str(searches_dir))
        ts_run_dirs = get_run_directories(str(ts_results_dir))
        current_go_run_id = (
            get_run_id_from_dir(go_run_dirs[-1]) if go_run_dirs else None
        )
        current_ts_run_id = (
            get_run_id_from_dir(ts_run_dirs[-1]) if ts_run_dirs else None
        )
        go_timing_relpath = None
        if current_go_run_id:
            go_timing_path = searches_dir / current_go_run_id / "timing.json"
            if go_timing_path.is_file():
                go_timing_relpath = os.path.relpath(go_timing_path, output_path)
        ts_timing_relpath = None
        if current_ts_run_id:
            ts_timing_path = ts_results_dir / current_ts_run_id / "timing.json"
            if ts_timing_path.is_file():
                ts_timing_relpath = os.path.relpath(ts_timing_path, output_path)
        write_timing_file(
            str(output_path),
            build_timing_payload(
                backend="go_ts",
                timings_s=go_ts_timings,
                extra={
                    "formula": formula,
                    "counters": {
                        "ts_success": ts_success,
                        "ts_total": len(ts_results),
                    },
                    "current_go_run_id": current_go_run_id,
                    "current_ts_run_id": current_ts_run_id,
                    "go_run_timing_relpath": go_timing_relpath,
                    "ts_run_timing_relpath": ts_timing_relpath,
                },
            ),
            filename=GO_TS_TIMING_JSON_FILENAME,
        )
    logger.info(
        "Completed GO->TS pipeline for %s: successful NEBs=%d/%d, wall_time=%.2f s",
        formula,
        ts_success,
        len(ts_results),
        elapsed_s,
    )
    return {
        "formula": formula,
        "output_dir": output_path,
        "searches_dir": searches_dir,
        "ts_results_dir": ts_results_dir,
        "minima_by_formula": minima_by_formula,
        "ts_results": ts_results,
        "ts_success_count": ts_success,
        "ts_total_count": len(ts_results),
        "wall_time_s": elapsed_s,
        "timings_s": go_ts_timings,
    }


def _run_one_element_go_ts_pipeline(
    element: str,
    n_atoms: int,
    system_type: SystemType,
    *,
    go_params: dict[str, Any],
    ts_kwargs: dict[str, Any],
    seed: int | None = None,
    verbosity: int = 1,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run one-element GO then TS and return a compact run summary."""
    if not element or not isinstance(element, str):
        raise SCGOValidationError("element must be a non-empty string")
    if n_atoms < 1:
        raise SCGOValidationError("n_atoms must be >= 1")
    composition = [element] * n_atoms
    return _run_go_ts_pipeline(
        composition,
        system_type,
        go_params=go_params,
        ts_kwargs=ts_kwargs,
        seed=seed,
        verbosity=verbosity,
        output_dir=output_dir,
    )


def _execute_run_go_ts(context: RunGOTSContext) -> dict[str, Any]:
    from scgo import runner_api as _runner_api

    return _runner_api._run_go_ts_pipeline(
        context.composition,
        context.system_type,
        go_params=context.go_params,
        ts_kwargs=context.ts_kwargs,
        adsorbate_definition=context.adsorbate_definition,
        seed=context.seed,
        verbosity=context.verbosity,
        output_dir=context.output_dir,
    )


def run_go_ts(
    composition: CompositionInput,
    *,
    go_params: dict[str, Any] | None = None,
    ts_params: dict[str, Any] | None = None,
    seed: int | None = None,
    verbosity: int = 1,
    output_dir: str | Path | None = None,
    output_root: str | Path | None = None,
    output_stem: str | None = None,
    surface_config: SurfaceSystemConfig | None = None,
    system_type: SystemType | None = None,
    adsorbates: AdsorbatesInput | None = None,
    log_summary: bool = True,
) -> dict[str, Any]:
    """Run global optimization then transition-state search for one composition."""
    try:
        context = _prepare_run_go_ts_context(
            composition,
            go_params=go_params,
            ts_params=ts_params,
            seed=seed,
            verbosity=verbosity,
            output_dir=output_dir,
            output_root=output_root,
            output_stem=output_stem,
            surface_config=surface_config,
            system_type=system_type,
            adsorbates=adsorbates,
        )
    except SCGOValidationError as exc:
        _log_validation_error(exc)
        raise
    t0 = perf_counter()
    summary = _execute_run_go_ts(context)
    if log_summary:
        log_go_ts_summary(_LOGGER, summary, wall_time_s=perf_counter() - t0)
    return summary


def run_go_ts_campaign(
    compositions: Iterable[CompositionInput],
    *,
    go_params: dict[str, Any] | None = None,
    ts_params: dict[str, Any] | None = None,
    seed: int | None = None,
    verbosity: int = 1,
    output_dir: str | Path | None = None,
    output_root: str | Path | None = None,
    output_stem: str | None = None,
    surface_config: SurfaceSystemConfig | None = None,
    system_type: SystemType | None = None,
    adsorbates: AdsorbatesInput | None = None,
    log_summary: bool = True,
) -> dict[str, dict[str, Any]]:
    """Run GO+TS for multiple compositions."""
    try:
        st = _require_system_type(system_type, "run_go_ts_campaign")
        validate_system_type_settings(system_type=st, surface_config=surface_config)
        if go_params is not None:
            _reject_system_keys(go_params, context="run_go_ts_campaign")
        if ts_params is not None:
            _reject_system_keys(ts_params, context="run_go_ts_campaign", kind="ts")
    except SCGOValidationError as exc:
        _log_validation_error(exc)
        raise
    go_mat, ts_mat = _resolve_go_ts_params(
        system_type=st,
        surface_config=surface_config,
        go_params=go_params,
        ts_params=ts_params,
    )
    eff_seed = resolve_workflow_seed(seed_kw=seed, go_params=go_mat, ts_params=ts_mat)
    go_prep = _with_surface_in_optimizers(go_mat, surface_config=surface_config)
    _validate_go_ts_param_coherence(
        go_prepared=go_prep,
        ts_params=ts_mat,
        system_type=st,
        surface_config=surface_config,
    )

    full_compositions: list[list[str]] = []
    ads_def, ads_temp = None, None
    preset_ads = (
        extract_adsorbate_definition_from_params(go_mat) if adsorbates is None else None
    )
    for core_comp in _as_composition_list(compositions):
        ads_def, ads_temp, full_comp = resolve_adsorbate_run_composition(
            system_type=st,
            composition=core_comp,
            adsorbates=adsorbates,
            preset_adsorbate_definition=preset_ads,
            context="run_go_ts_campaign",
        )
        full_compositions.append(full_comp)
        _validate_go_ts_surface_config(
            go_prep,
            system_type=st,
            surface_config=surface_config,
            adsorbate_composition=full_comp,
        )

    go_local = _with_system_type_in_optimizer_params(go_prep, system_type=st)
    go_local = _merge_adsorbate_context_into_params(
        go_local,
        adsorbate_definition=ads_def,
        adsorbate_fragment_template=ads_temp,
    )
    ts_kwargs = _coerce_ts_for_runner(
        ts_mat,
        fn_name="run_go_ts_campaign",
        system_type=st,
        surface_config=surface_config,
    )
    parent = _resolved_path(output_dir) or _default_go_ts_output_path(
        full_compositions[0],
        go_params=go_mat,
        output_stem=output_stem or "go_ts_campaign",
        output_root=output_root,
    )
    out: dict[str, dict[str, Any]] = {}
    t0 = perf_counter()
    for comp in full_compositions:
        formula = get_cluster_formula(comp)
        context = RunGOTSContext(
            composition=comp,
            system_type=st,
            go_params=go_local,
            ts_kwargs=ts_kwargs,
            seed=eff_seed,
            verbosity=verbosity,
            output_dir=parent / f"{formula}_campaign",
            adsorbate_definition=ads_def,
        )
        out[formula] = _execute_run_go_ts(context)
    if log_summary:
        total = sum(int(s.get("ts_total_count") or 0) for s in out.values())
        ok = sum(int(s.get("ts_success_count") or 0) for s in out.values())
        _log_completion(
            "run_go_ts_campaign",
            elapsed_s=perf_counter() - t0,
            details=f"compositions={len(out)} successful_nebs={ok}/{total}",
        )
    return out


def run_ts_search(
    composition: CompositionInput,
    *,
    ts_params: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
    searches_dir: str | Path | None = None,
    seed: int | None = None,
    verbosity: int = 1,
    surface_config: SurfaceSystemConfig | None = None,
    system_type: SystemType | None = None,
    adsorbates: AdsorbatesInput | None = None,
    log_summary: bool = True,
) -> list[dict[str, Any]]:
    """Run transition-state search for one composition.

    ``output_dir`` is the campaign root. Minima are loaded from
    ``{formula}_searches/`` (or from ``searches_dir`` when provided). TS
    artifacts are written to sibling ``{formula}_ts_results/`` with
    ``run_*/pair_*/`` subdirectories. If ``output_dir`` points at an existing
    ``*_searches`` directory, its parent is treated as the campaign root.
    """
    from scgo import runner_api as _runner_api

    try:
        context = _prepare_run_ts_search_context(
            composition,
            ts_params=ts_params,
            output_dir=output_dir,
            searches_dir=searches_dir,
            seed=seed,
            verbosity=verbosity,
            surface_config=surface_config,
            system_type=system_type,
            adsorbates=adsorbates,
        )
    except SCGOValidationError as exc:
        _log_validation_error(exc)
        raise
    configure_logging(verbosity)
    log_ts_configuration(
        context.ts_params,
        context.ts_kwargs,
        verbosity=verbosity,
        user_params=ts_params,
        base=context.ts_base,
    )
    t0 = perf_counter()
    results = _runner_api._ts_search(
        context.composition,
        output_dir=context.output_dir,
        searches_dir=context.searches_dir,
        seed=context.seed,
        verbosity=verbosity,
        adsorbate_definition=context.adsorbate_definition,
        system_type=context.system_type,
        **context.ts_kwargs,
    )
    if log_summary:
        ok = sum(1 for r in results if r.get("status") == "success")
        _log_completion(
            "run_ts_search",
            elapsed_s=perf_counter() - t0,
            details=f"successful_nebs={ok}/{len(results)} output_dir={context.output_dir}",
        )
    return results


def run_ts_campaign(
    compositions: Iterable[CompositionInput],
    *,
    ts_params: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
    seed: int | None = None,
    verbosity: int = 1,
    surface_config: SurfaceSystemConfig | None = None,
    system_type: SystemType | None = None,
    adsorbates: AdsorbatesInput | None = None,
    log_summary: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    from scgo import runner_api as _runner_api

    try:
        st = _require_system_type(system_type, "run_ts_campaign")
        validate_system_type_settings(system_type=st, surface_config=surface_config)
        if ts_params is not None:
            _reject_system_keys(ts_params, context="run_ts_campaign", kind="ts")
    except SCGOValidationError as exc:
        _log_validation_error(exc)
        raise
    ts_mat = _resolve_ts_params(
        ts_params, system_type=st, surface_config=surface_config
    )
    ts_base = initialize_ts_params(None, system_type=st, surface_config=surface_config)
    eff_seed = resolve_workflow_seed(seed_kw=seed, ts_params=ts_mat)
    ts_kwargs = _coerce_ts_for_runner(
        ts_mat, fn_name="run_ts_campaign", system_type=st, surface_config=surface_config
    )
    ts_kwargs.pop("system_type", None)  # passed as positional arg below
    configure_logging(verbosity)
    log_ts_configuration(
        ts_mat,
        ts_kwargs,
        verbosity=verbosity,
        user_params=ts_params,
        base=ts_base,
    )

    full_compositions: list[list[str]] = []
    ads_def: AdsorbateDefinition | None = None
    preset_ads = (
        extract_adsorbate_definition_from_params(ts_mat) if adsorbates is None else None
    )
    for core in _as_composition_list(compositions):
        ads_def, _, full = resolve_adsorbate_run_composition(
            system_type=st,
            composition=core,
            adsorbates=adsorbates,
            preset_adsorbate_definition=preset_ads,
            context="run_ts_campaign",
        )
        full_compositions.append(full)
    out_path = _resolved_path(output_dir)
    t0 = perf_counter()
    if ads_def:
        ts_kwargs["adsorbate_definition"] = ads_def
    campaign = _runner_api._ts_campaign(
        full_compositions,
        st,
        output_dir=out_path,
        seed=eff_seed,
        verbosity=verbosity,
        ts_kwargs=ts_kwargs,
    )
    if log_summary:
        total = sum(len(v) for v in campaign.values())
        ok = sum(
            1 for rl in campaign.values() for r in rl if r.get("status") == "success"
        )
        _log_completion(
            "run_ts_campaign",
            elapsed_s=perf_counter() - t0,
            details=f"compositions={len(campaign)} successful_nebs={ok}/{total}",
        )
    return campaign


def log_go_ts_summary(
    logger: Any,
    summary: dict[str, Any],
    *,
    wall_time_s: float | None = None,
) -> None:
    """Log NEB success counts from a ``run_go_ts*`` summary dict."""
    ts_results = summary.get("ts_results") or []
    ok = sum(1 for r in ts_results if r.get("status") == "success")
    logger.info("Successful NEBs: %d/%d", ok, len(ts_results))
    if wall_time_s is not None:
        logger.info("Total wall time: %.2f s", wall_time_s)
