"""Param merging, allowlists, coherence checks, and run-context dataclasses.

These helpers resolve/merge ``go_params`` / ``ts_params`` against preset
defaults, validate consistency between run-level ``system_type`` /
``surface_config`` and the params dicts, and build the frozen context
dataclasses consumed by :mod:`scgo.runner_api`'s public run functions.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.calculators.calculator import Calculator

from scgo.exceptions import SCGOValidationError
from scgo.param_presets import get_default_params
from scgo.runner_composition import (
    CompositionInput,
    _as_composition,
    _as_composition_list,
)
from scgo.runner_go import select_scgo_minima_algorithm
from scgo.surface.config import SurfaceSystemConfig
from scgo.system_types import (
    AdsorbateDefinition,
    AdsorbatesInput,
    SystemType,
    extract_adsorbate_definition_from_params,
    get_system_policy,
    resolve_adsorbate_run_composition,
    validate_system_type_settings,
)
from scgo.utils.helpers import get_cluster_formula
from scgo.utils.logging import get_logger
from scgo.utils.output_paths import resolve_go_searches_dir
from scgo.utils.run_helpers import initialize_params, initialize_ts_params
from scgo.utils.ts_runner_kwargs import coerce_ts_params_to_runner_kwargs

_ALGO_KEYS = ("simple", "bh", "ga")
_LOGGER = get_logger(__name__)
_VALIDATION_LOGGER = get_logger("scgo.validation")
_DEFAULT_GO_PARAMS: dict[str, Any] | None = None


def _log_validation_error(exc: SCGOValidationError) -> None:
    """Emit user-facing ERROR for validation failures at the runner API boundary."""
    _VALIDATION_LOGGER.error("Validation error: %s", exc)


@dataclass(frozen=True)
class RunGOContext:
    composition: list[str]
    system_type: SystemType
    params: dict[str, Any]
    seed: int | None
    run_id: str | None
    clean: bool
    output_dir: Path | None
    verbosity: int
    calculator_for_global_optimization: Calculator | None
    output_summary_dir: str


@dataclass(frozen=True)
class RunGOCampaignContext:
    compositions: list[list[str]]
    system_type: SystemType
    params: dict[str, Any]
    seed: int | None
    run_id: str | None
    clean: bool
    output_dir: Path | None
    verbosity: int
    output_summary_dir: str


@dataclass(frozen=True)
class RunGOTSContext:
    composition: list[str]
    system_type: SystemType
    go_params: dict[str, Any]
    ts_kwargs: dict[str, Any]
    seed: int | None
    verbosity: int
    output_dir: Path
    adsorbate_definition: AdsorbateDefinition | None


@dataclass(frozen=True)
class RunTSContext:
    composition: list[str]
    system_type: SystemType
    ts_params: dict[str, Any]
    ts_base: dict[str, Any]
    ts_kwargs: dict[str, Any]
    seed: int | None
    verbosity: int
    output_dir: Path | None
    searches_dir: Path | None
    adsorbate_definition: AdsorbateDefinition | None


def _optimizer_write_timing_json_enabled(params: dict[str, Any]) -> bool:
    """Return True when any GO optimizer slot requests ``write_timing_json``."""
    opt = params.get("optimizer_params") or {}
    for algo in _ALGO_KEYS:
        slot = opt.get(algo)
        if isinstance(slot, dict) and slot.get("write_timing_json"):
            return True
    return False


def _default_optimizer_system_type(algo: str) -> SystemType | None:
    global _DEFAULT_GO_PARAMS
    if _DEFAULT_GO_PARAMS is None:
        _DEFAULT_GO_PARAMS = get_default_params()
    slot = _DEFAULT_GO_PARAMS.get("optimizer_params", {}).get(algo, {})
    if isinstance(slot, dict):
        return slot.get("system_type")
    return None


def _resolved_path(path: str | Path | None) -> Path | None:
    return Path(path).expanduser().resolve() if path is not None else None


def _require_system_type(system_type: SystemType | None, fn_name: str) -> SystemType:
    if system_type is None:
        raise SCGOValidationError(f"system_type is required for {fn_name}.")
    return system_type


def _prepare_run_context(
    composition: CompositionInput,
    *,
    system_type: SystemType | None,
    surface_config: SurfaceSystemConfig | None,
    params: dict[str, Any] | None,
    adsorbates: AdsorbatesInput | None,
    context: str,
) -> tuple[
    SystemType,
    dict[str, Any] | None,
    AdsorbateDefinition | None,
    Atoms | None,
    list[str],
]:
    st = _require_system_type(system_type, context)
    validate_system_type_settings(system_type=st, surface_config=surface_config)
    if params is not None:
        _reject_system_keys(params, context=context, kind="go")
    comp = _as_composition(composition)
    preset_ads = (
        extract_adsorbate_definition_from_params(params)
        if adsorbates is None and params is not None
        else None
    )
    ads_def, ads_template, full_comp = resolve_adsorbate_run_composition(
        system_type=st,
        composition=comp,
        adsorbates=adsorbates,
        preset_adsorbate_definition=preset_ads,
        context=context,
    )
    params_prep = params or {}
    if params:
        params_prep = _with_surface_in_optimizers(params, surface_config=surface_config)
    if params_prep is not None:
        params_prep = _with_adsorbate_in_optimizers(
            params_prep,
            adsorbate_definition=ads_def,
            adsorbate_fragment_template=ads_template,
        )
    return st, params_prep, ads_def, ads_template, full_comp


def _validate_go_ts_surface_config(
    go_prepared: dict[str, Any],
    *,
    system_type: SystemType,
    surface_config: SurfaceSystemConfig | None,
    adsorbate_composition: list[str],
) -> None:
    """For surface system types, ensure active GO slot does not conflict."""
    if not get_system_policy(system_type).uses_surface:
        return
    if not isinstance(surface_config, SurfaceSystemConfig):
        raise SCGOValidationError(
            f"system_type={system_type!r} requires the run surface_config argument "
            "to be a SurfaceSystemConfig."
        )
    chosen = select_scgo_minima_algorithm(len(adsorbate_composition), system_type)
    op = go_prepared.get("optimizer_params") or {}
    go_slot = op.get(chosen)
    if not isinstance(go_slot, dict):
        go_slot = {}
    go_sc = go_slot.get("surface_config")
    if go_sc is not None and go_sc != surface_config:
        raise SCGOValidationError(
            "run surface_config and go_params['optimizer_params']["
            f"'{chosen}']['surface_config'] disagree."
        )


def _validate_go_ts_param_coherence(
    *,
    go_prepared: dict[str, Any],
    ts_params: dict[str, Any],
    system_type: SystemType,
    surface_config: SurfaceSystemConfig | None,
) -> None:
    """Validate GO/TS params coherence against run-level system definition."""
    policy = get_system_policy(system_type)
    go_surface_config = go_prepared.get("surface_config") or surface_config
    if policy.uses_surface:
        if not isinstance(go_surface_config, SurfaceSystemConfig):
            raise SCGOValidationError(
                "GO/TS coherence error: surface system types require "
                "go_params['surface_config'] or run surface_config=."
            )
        if (
            surface_config is not None
            and go_prepared.get("surface_config") is not None
            and go_prepared.get("surface_config") != surface_config
        ):
            raise SCGOValidationError(
                "GO/TS coherence error: go_params['surface_config'] disagrees with "
                "run surface_config."
            )
    elif go_surface_config is not None:
        raise SCGOValidationError(
            "GO/TS coherence error: go_params['surface_config'] is set but "
            f"run system_type={system_type!r} is non-surface."
        )

    optimizer_params = go_prepared.get("optimizer_params") or {}
    for algo in _ALGO_KEYS:
        slot = optimizer_params.get(algo)
        if slot is None:
            continue
        if not isinstance(slot, dict):
            raise SCGOValidationError(
                f"go_params['optimizer_params']['{algo}'] must be a dict."
            )
        slot_system_type = slot.get("system_type")
        default_slot_st = _default_optimizer_system_type(algo)
        if (
            slot_system_type is not None
            and slot_system_type != system_type
            and slot_system_type != default_slot_st
        ):
            raise SCGOValidationError(
                "GO/TS coherence error: "
                f"go_params['optimizer_params']['{algo}']['system_type']="
                f"{slot_system_type!r} disagrees with run system_type={system_type!r}."
            )
        slot_surface_config = slot.get("surface_config")
        if policy.uses_surface:
            if (
                slot_surface_config is not None
                and surface_config is not None
                and slot_surface_config != surface_config
            ):
                raise SCGOValidationError(
                    "GO/TS coherence error: "
                    f"go_params['optimizer_params']['{algo}']['surface_config'] "
                    "disagrees with run surface_config."
                )
        elif slot_surface_config is not None:
            raise SCGOValidationError(
                "GO/TS coherence error: go_params surface_config is set but "
                f"run system_type={system_type!r} is non-surface."
            )

    ts_surface_config = ts_params.get("surface_config") or surface_config
    if policy.uses_surface:
        if not isinstance(ts_surface_config, SurfaceSystemConfig):
            raise SCGOValidationError(
                "GO/TS coherence error: surface system types require "
                "ts_params['surface_config'] or run surface_config=."
            )
        if (
            surface_config is not None
            and ts_params.get("surface_config") is not None
            and ts_params.get("surface_config") != surface_config
        ):
            raise SCGOValidationError(
                "GO/TS coherence error: ts_params['surface_config'] disagrees with "
                "run surface_config."
            )
    elif ts_surface_config is not None:
        raise SCGOValidationError(
            "GO/TS coherence error: ts_params['surface_config'] is set but "
            f"run system_type={system_type!r} is non-surface."
        )


def _merge_adsorbate_context_into_params(
    base: dict[str, Any] | None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Attach adsorbate/surface init context for :func:`_run_go_trials` / GA."""
    out = copy.deepcopy(base) if base is not None else {}
    out.update({k: v for k, v in kwargs.items() if v is not None})
    return out


def _with_system_type_in_optimizer_params(
    params: dict[str, Any] | None,
    *,
    system_type: SystemType,
) -> dict[str, Any]:
    """Attach ``system_type`` (and fan-out ``surface_config``) to optimizer slots."""
    out = copy.deepcopy(params or {})
    op = out.setdefault("optimizer_params", {})
    for algo in _ALGO_KEYS:
        cfg = op.setdefault(algo, {})
        cfg["system_type"] = system_type
    # Add surface_config to all optimizer slots if it's in params
    if "surface_config" in out:
        for algo in _ALGO_KEYS:
            op.setdefault(algo, {})["surface_config"] = out["surface_config"]
    return out


def _coerce_ts_for_runner(
    ts_params: dict[str, Any] | None,
    *,
    fn_name: str,
    system_type: SystemType,
    surface_config: SurfaceSystemConfig | None,
) -> dict[str, Any]:
    if not ts_params:
        raise SCGOValidationError(
            f"ts_params is required for {fn_name}. Build with get_ts_search_params(...)."
        )
    _reject_system_keys(ts_params, context=fn_name, kind="ts")
    return coerce_ts_params_to_runner_kwargs(
        ts_params, system_type=system_type, surface_config=surface_config
    )


def _resolve_go_params(
    go_params: dict[str, Any] | None,
    *,
    surface_config: SurfaceSystemConfig | None = None,
) -> dict[str, Any]:
    """Merge GO params with defaults and inject run-level ``surface_config`` when missing."""
    merged = initialize_params(go_params)
    if surface_config is not None and merged.get("surface_config") is None:
        merged = copy.deepcopy(merged)
        merged["surface_config"] = surface_config
    return merged


def _resolve_ts_params(
    ts_params: dict[str, Any] | None,
    *,
    system_type: SystemType,
    surface_config: SurfaceSystemConfig | None = None,
    go_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge TS params with defaults; align calculator with merged GO when provided."""
    merged_go = initialize_params(go_params) if go_params is not None else None
    merged = initialize_ts_params(
        ts_params,
        system_type=system_type,
        surface_config=surface_config,
        go_params=merged_go,
    )
    if surface_config is not None and merged.get("surface_config") is None:
        merged = copy.deepcopy(merged)
        merged["surface_config"] = surface_config
    return merged


def _resolve_go_ts_params(
    *,
    system_type: SystemType,
    surface_config: SurfaceSystemConfig | None,
    go_params: dict[str, Any] | None,
    ts_params: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return merged GO and TS param dicts using canonical preset defaults."""
    effective_go = _resolve_go_params(go_params, surface_config=surface_config)
    effective_ts = _resolve_ts_params(
        ts_params,
        system_type=system_type,
        surface_config=surface_config,
        go_params=effective_go,
    )
    return effective_go, effective_ts


def _calculator_slug_from_go_params(go_params: dict[str, Any] | None) -> str:
    c = str((go_params or {}).get("calculator", "MACE")).strip().upper()
    if c in ("MACE", "UMA"):
        return c.lower()
    return c.lower() or "calc"


def _default_go_ts_output_path(
    composition: list[str],
    *,
    go_params: dict[str, Any],
    output_stem: str | None,
    output_root: str | Path | None,
) -> Path:
    root = output_root if output_root is not None else Path.cwd() / "scgo_runs"
    p = Path(root).expanduser().resolve()
    stem = output_stem or get_cluster_formula(composition)
    path = (p / f"{stem}_{_calculator_slug_from_go_params(go_params)}").resolve()
    if output_root is None:
        _LOGGER.info("No output_dir provided; using default campaign root %s", path)
    return path


def _log_completion(kind: str, *, elapsed_s: float, details: str) -> None:
    _LOGGER.info("%s completed in %.2f s (%s)", kind, elapsed_s, details)


def _as_int_seed(label: str, value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as e:
        raise SCGOValidationError(f"{label} must be int-like, got {value!r}") from e


def resolve_workflow_seed(
    *,
    seed_kw: int | None = None,
    go_params: dict[str, Any] | None = None,
    ts_params: dict[str, Any] | None = None,
) -> int | None:
    """Unify run ``seed=...``, ``go_params['seed']``, and ``ts_params['seed']``; all non-null must agree."""
    parts: list[tuple[str, int]] = []
    if seed_kw is not None:
        parts.append(("run_kwd(seed=...)", _as_int_seed("run seed", seed_kw)))
    if go_params is not None and go_params.get("seed") is not None:
        parts.append(
            (
                "go_params['seed']",
                _as_int_seed("go_params['seed']", go_params.get("seed")),
            )
        )
    if ts_params is not None and ts_params.get("seed") is not None:
        parts.append(
            (
                "ts_params['seed']",
                _as_int_seed("ts_params['seed']", ts_params.get("seed")),
            )
        )
    if not parts:
        return None
    values = {v for _, v in parts}
    if len(values) > 1:
        desc = ", ".join(f"{name}={v}" for name, v in parts)
        raise SCGOValidationError(f"Inconsistent random seeds: {desc}")
    return next(iter(values))


def _with_surface_in_optimizers(
    go_params: dict[str, Any], *, surface_config: SurfaceSystemConfig | None
) -> dict[str, Any]:
    """Copy ``go_params``; fan out explicit run ``surface_config`` to optimizer slots."""
    out = copy.deepcopy(go_params)
    if surface_config is not None:
        if out.get("surface_config") is None:
            out["surface_config"] = surface_config
        elif out.get("surface_config") != surface_config:
            raise SCGOValidationError(
                "run argument surface_config must match go_params['surface_config'] "
                "when both are set."
            )
        op = out.setdefault("optimizer_params", {})
        for key in _ALGO_KEYS:
            if key not in op:
                continue
            slot = op[key]
            if not isinstance(slot, dict):
                raise SCGOValidationError(
                    f"optimizer_params['{key}'] must be a dict when using go_params['surface_config']"
                )
            ex = slot.get("surface_config")
            if ex is None:
                slot["surface_config"] = surface_config
            elif ex != surface_config:
                raise SCGOValidationError(
                    f"run argument surface_config must match "
                    f"go_params['optimizer_params']['{key}']['surface_config'] when both are set."
                )
    return out


def _with_adsorbate_in_optimizers(
    go_params: dict[str, Any] | None,
    *,
    adsorbate_definition: Any | None = None,
    adsorbate_fragment_template: Any | None = None,
) -> dict[str, Any]:
    """Copy ``go_params``; fan out derived adsorbate context to optimizer slots."""
    out = copy.deepcopy(go_params) if go_params is not None else {}
    cluster_adsorbate_config = out.get("cluster_adsorbate_config")

    # If any adsorbate param is set, distribute to all optimizer slots
    if (
        adsorbate_definition is not None
        or adsorbate_fragment_template is not None
        or cluster_adsorbate_config is not None
    ):
        op = out.setdefault("optimizer_params", {})
        for key in _ALGO_KEYS:
            slot = op.setdefault(key, {})
            if not isinstance(slot, dict):
                raise SCGOValidationError(
                    f"optimizer_params['{key}'] must be a dict when using adsorbate parameters"
                )
            if adsorbate_definition is not None:
                ex = slot.get("adsorbate_definition")
                if ex is None:
                    slot["adsorbate_definition"] = adsorbate_definition
                # Don't check for match, multiple definitions may be equivalent
            if adsorbate_fragment_template is not None:
                ex = slot.get("adsorbate_fragment_template")
                if ex is None:
                    slot["adsorbate_fragment_template"] = adsorbate_fragment_template
            if cluster_adsorbate_config is not None:
                ex = slot.get("cluster_adsorbate_config")
                if ex is None:
                    slot["cluster_adsorbate_config"] = cluster_adsorbate_config
    return out


def _reject_system_keys(
    params: dict[str, Any], *, context: str, kind: str = "go"
) -> None:
    """Reject ``system_type`` in go/ts params (use the run-function argument).

    Top-level ``surface_config`` in params is allowed and fanned into optimizer
    slots; only ``system_type`` remains forbidden here.
    """
    if params.get("system_type") is not None:
        guidance = "Use the run function argument instead."
        if kind == "ts":
            guidance = (
                "Use the run function system_type argument; "
                "ts_params['surface_config'] is allowed."
            )
        raise SCGOValidationError(
            f"{context} does not allow top-level {kind}_params['system_type']. "
            f"{guidance}"
        )


def _prepare_run_go_context(
    composition: CompositionInput,
    *,
    params: dict[str, Any] | None,
    seed: int | None,
    verbosity: int,
    run_id: str | None,
    clean: bool,
    output_dir: str | Path | None,
    calculator_for_global_optimization: Calculator | None,
    surface_config: SurfaceSystemConfig | None,
    system_type: SystemType | None,
    adsorbates: AdsorbatesInput | None,
) -> RunGOContext:
    st, params_prep, ads_def, ads_temp, comp = _prepare_run_context(
        composition,
        system_type=system_type,
        surface_config=surface_config,
        params=params,
        adsorbates=adsorbates,
        context="run_go",
    )
    eff_seed = resolve_workflow_seed(seed_kw=seed, go_params=params)
    eff_params = _with_system_type_in_optimizer_params(params_prep, system_type=st)
    eff_params = _merge_adsorbate_context_into_params(
        eff_params,
        adsorbate_definition=ads_def,
        adsorbate_fragment_template=ads_temp,
    )
    out_path = _resolved_path(output_dir)
    searches_dir = str(resolve_go_searches_dir(output_dir, get_cluster_formula(comp)))
    return RunGOContext(
        composition=comp,
        system_type=st,
        params=eff_params,
        seed=eff_seed,
        run_id=run_id,
        clean=clean,
        output_dir=out_path,
        verbosity=verbosity,
        calculator_for_global_optimization=calculator_for_global_optimization,
        output_summary_dir=searches_dir,
    )


def _prepare_run_go_campaign_context(
    compositions: Iterable[CompositionInput],
    *,
    params: dict[str, Any] | None,
    seed: int | None,
    verbosity: int,
    run_id: str | None,
    clean: bool,
    output_dir: str | Path | None,
    surface_config: SurfaceSystemConfig | None,
    system_type: SystemType | None,
    adsorbates: AdsorbatesInput | None,
) -> RunGOCampaignContext:
    st = _require_system_type(system_type, "run_go_campaign")
    validate_system_type_settings(system_type=st, surface_config=surface_config)
    if params is not None:
        _reject_system_keys(params, context="run_go_campaign")
    params_prep = (
        _with_surface_in_optimizers(params, surface_config=surface_config)
        if params
        else None
    )
    eff_seed = resolve_workflow_seed(seed_kw=seed, go_params=params)
    eff_params = _with_system_type_in_optimizer_params(params_prep, system_type=st)
    preset_ads_def = (
        extract_adsorbate_definition_from_params(eff_params)
        if adsorbates is None
        else None
    )
    full_compositions: list[list[str]] = []
    ads_def: AdsorbateDefinition | None = None
    ads_temp: list[Atoms] | Atoms | None = None
    for composition_item in _as_composition_list(compositions):
        comp = _as_composition(composition_item)
        ads_def, ads_temp, full_comp = resolve_adsorbate_run_composition(
            system_type=st,
            composition=comp,
            adsorbates=adsorbates,
            preset_adsorbate_definition=preset_ads_def,
            context="run_go_campaign",
        )
        full_compositions.append(full_comp)
        if ads_def is not None:
            eff_params["adsorbate_definition"] = ads_def
        if ads_temp is not None:
            eff_params["adsorbate_fragment_template"] = ads_temp
    if adsorbates is not None or preset_ads_def is not None:
        eff_params = _with_adsorbate_in_optimizers(
            eff_params,
            adsorbate_definition=eff_params.get("adsorbate_definition"),
            adsorbate_fragment_template=eff_params.get("adsorbate_fragment_template"),
        )
    out_path = _resolved_path(output_dir)
    campaign_root = (
        str(Path(out_path).expanduser().resolve())
        if out_path is not None
        else str(
            resolve_go_searches_dir(
                None, get_cluster_formula(full_compositions[0])
            ).parent
        )
    )
    return RunGOCampaignContext(
        compositions=full_compositions,
        system_type=st,
        params=eff_params,
        seed=eff_seed,
        run_id=run_id,
        clean=clean,
        output_dir=out_path,
        verbosity=verbosity,
        output_summary_dir=campaign_root,
    )


def _prepare_run_go_ts_context(
    composition: CompositionInput,
    *,
    go_params: dict[str, Any] | None,
    ts_params: dict[str, Any] | None,
    seed: int | None,
    verbosity: int,
    output_dir: str | Path | None,
    output_root: str | Path | None,
    output_stem: str | None,
    surface_config: SurfaceSystemConfig | None,
    system_type: SystemType | None,
    adsorbates: AdsorbatesInput | None,
) -> RunGOTSContext:
    context_name = "run_go_ts"
    st = _require_system_type(system_type, context_name)
    validate_system_type_settings(system_type=st, surface_config=surface_config)
    if go_params is not None:
        _reject_system_keys(go_params, context=context_name)
    if ts_params is not None:
        _reject_system_keys(ts_params, context=context_name, kind="ts")
    go_mat, ts_mat = _resolve_go_ts_params(
        system_type=st,
        surface_config=surface_config,
        go_params=go_params,
        ts_params=ts_params,
    )
    eff_seed = resolve_workflow_seed(seed_kw=seed, go_params=go_mat, ts_params=ts_mat)
    go_prep = _with_surface_in_optimizers(go_mat, surface_config=surface_config)
    core_comp = _as_composition(composition)
    preset_ads = (
        extract_adsorbate_definition_from_params(go_mat) if adsorbates is None else None
    )
    ads_def, ads_temp, comp = resolve_adsorbate_run_composition(
        system_type=st,
        composition=core_comp,
        adsorbates=adsorbates,
        preset_adsorbate_definition=preset_ads,
        context=context_name,
    )
    _validate_go_ts_param_coherence(
        go_prepared=go_prep,
        ts_params=ts_mat,
        system_type=st,
        surface_config=surface_config,
    )
    _validate_go_ts_surface_config(
        go_prep,
        system_type=st,
        surface_config=surface_config,
        adsorbate_composition=comp,
    )
    go_prep = _with_system_type_in_optimizer_params(go_prep, system_type=st)
    go_local = _merge_adsorbate_context_into_params(
        go_prep,
        adsorbate_definition=ads_def,
        adsorbate_fragment_template=ads_temp,
    )
    ts_kwargs = _coerce_ts_for_runner(
        ts_mat, fn_name=context_name, system_type=st, surface_config=surface_config
    )
    out_path = _resolved_path(output_dir) or _default_go_ts_output_path(
        comp, go_params=go_mat, output_stem=output_stem, output_root=output_root
    )
    return RunGOTSContext(
        composition=comp,
        system_type=st,
        go_params=go_local,
        ts_kwargs=ts_kwargs,
        seed=eff_seed,
        verbosity=verbosity,
        output_dir=out_path,
        adsorbate_definition=ads_def,
    )


def _prepare_run_ts_search_context(
    composition: CompositionInput,
    *,
    ts_params: dict[str, Any] | None,
    output_dir: str | Path | None,
    searches_dir: str | Path | None,
    seed: int | None,
    verbosity: int,
    surface_config: SurfaceSystemConfig | None,
    system_type: SystemType | None,
    adsorbates: AdsorbatesInput | None,
) -> RunTSContext:
    context_name = "run_ts_search"
    st, _, ads_def, _, comp = _prepare_run_context(
        composition,
        system_type=system_type,
        surface_config=surface_config,
        params=None,
        adsorbates=adsorbates,
        context=context_name,
    )
    if ts_params is not None:
        _reject_system_keys(ts_params, context=context_name, kind="ts")
    ts_mat = _resolve_ts_params(
        ts_params, system_type=st, surface_config=surface_config
    )
    ts_base = initialize_ts_params(None, system_type=st, surface_config=surface_config)
    eff_seed = resolve_workflow_seed(seed_kw=seed, ts_params=ts_mat)
    ts_kwargs = _coerce_ts_for_runner(
        ts_mat, fn_name=context_name, system_type=st, surface_config=surface_config
    )
    ts_kwargs.pop("system_type", None)
    return RunTSContext(
        composition=comp,
        system_type=st,
        ts_params=ts_mat,
        ts_base=ts_base,
        ts_kwargs=ts_kwargs,
        seed=eff_seed,
        verbosity=verbosity,
        output_dir=_resolved_path(output_dir),
        searches_dir=_resolved_path(searches_dir),
        adsorbate_definition=ads_def,
    )
