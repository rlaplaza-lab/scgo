"""Param merging, allowlists, coherence checks, and run-context dataclasses.

These helpers resolve/merge ``go_params`` / ``ts_params`` against preset
defaults, validate consistency between run-level ``system_type`` /
``surface_config`` and the params dicts, and build the frozen context
dataclasses consumed by :mod:`scgo.runner_api`'s public run functions.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.calculators.calculator import Calculator

from scgo.exceptions import SCGOValidationError
from scgo.runner_composition import (
    CompositionInput,
    _as_composition,
    _as_composition_list,
)
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
from scgo.utils.logging import get_logger
from scgo.utils.output_paths import (
    calculator_slug_from_go_params,
    formula_searches_dir,
    resolve_campaign_root_from_args,
    resolve_go_searches_dir,
)
from scgo.utils.path_keys import resolve_run_path_key
from scgo.utils.run_helpers import initialize_params, initialize_ts_params
from scgo.utils.ts_runner_kwargs import coerce_ts_params_to_runner_kwargs

_ALGO_KEYS = ("simple", "bh", "ga")
# Forbidden inside optimizer_params slots (run identity, not algo hyperparameters).
_SLOT_IDENTITY_KEYS = frozenset(
    {
        "system_type",
        "surface_config",
        "adsorbate_definition",
        "adsorbate_fragment_template",
        "cluster_adsorbate_config",
    }
)
logger = get_logger(__name__)
_VALIDATION_LOGGER = get_logger("scgo.validation")


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
    calculator_for_global_optimization: Calculator | None
    output_summary_dir: str
    # Parallel to ``compositions``.
    composition_adsorbate: list[
        tuple[AdsorbateDefinition | None, Atoms | list[Atoms] | None]
    ]


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


def _copy_params(params: dict[str, Any] | None) -> dict[str, Any]:
    """Shallow-copy params with isolated per-slot ``optimizer_params`` dicts.

    The top-level dict is copied shallowly, and each ``optimizer_params[algo]``
    slot is copied as a new dict so callers can mutate slot keys without leaking
    back into the source. Heavy, identity-sensitive values (``relaxer`` — which
    aliases under ``deepcopy`` — and a top-level frozen ``surface_config``) are
    shared by reference, matching the prior behavior and avoiding any cloning of
    torch models or slab configs.
    """
    if params is None:
        return {}
    out = dict(params)
    opt = out.get("optimizer_params")
    if isinstance(opt, dict):
        out["optimizer_params"] = {
            key: dict(slot) if isinstance(slot, dict) else slot
            for key, slot in opt.items()
        }
    return out


def _reject_slot_identity_keys(params: dict[str, Any] | None) -> None:
    """Reject run-identity keys inside ``optimizer_params`` slots."""
    if not params:
        return
    opt = params.get("optimizer_params")
    if not isinstance(opt, dict):
        return
    for algo, slot in opt.items():
        if not isinstance(slot, dict):
            continue
        bad = sorted(k for k in slot if k in _SLOT_IDENTITY_KEYS)
        if bad:
            raise SCGOValidationError(
                f"optimizer_params['{algo}'] must not contain identity keys {bad}. "
                "Use the run function arguments (system_type=, surface_config=, "
                "adsorbates=) and/or top-level go_params keys instead."
            )


def _with_surface_on_params(
    go_params: dict[str, Any] | None, *, surface_config: SurfaceSystemConfig | None
) -> dict[str, Any]:
    """Copy ``go_params``; set top-level ``surface_config`` from the run argument."""
    out = _copy_params(go_params)
    if surface_config is not None:
        if out.get("surface_config") is None:
            out["surface_config"] = surface_config
        elif out.get("surface_config") != surface_config:
            raise SCGOValidationError(
                "run argument surface_config must match go_params['surface_config'] "
                "when both are set."
            )
    _reject_slot_identity_keys(out)
    return out


def _resolved_path(path: str | Path | None) -> Path | None:
    return Path(path).expanduser().resolve() if path is not None else None


def _require_system_type(system_type: SystemType | None, fn_name: str) -> SystemType:
    if system_type is None:
        raise SCGOValidationError(f"system_type is required for {fn_name}.")
    return system_type


def _resolve_surface_config(
    surface_config: SurfaceSystemConfig | None,
    params: dict[str, Any] | None,
) -> SurfaceSystemConfig | None:
    """Prefer the run argument; else a top-level ``params['surface_config']``."""
    if surface_config is not None:
        return surface_config
    if params is not None:
        top_sc = params.get("surface_config")
        if isinstance(top_sc, SurfaceSystemConfig):
            return top_sc
    return None


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
    Atoms | list[Atoms] | None,
    list[str],
    SurfaceSystemConfig | None,
]:
    st = _require_system_type(system_type, context)
    resolved_surface = _resolve_surface_config(surface_config, params)
    if params is not None:
        _reject_system_keys(params, context=context, kind="go")
        _reject_slot_identity_keys(params)
    validate_system_type_settings(system_type=st, surface_config=resolved_surface)
    policy = get_system_policy(st)
    allow_empty = policy.slab_is_search_target
    comp = _as_composition(composition, allow_empty=allow_empty)
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
    params_prep = _with_surface_on_params(params, surface_config=resolved_surface)
    return st, params_prep, ads_def, ads_template, full_comp, resolved_surface


def _validate_go_ts_surface_config(
    *,
    system_type: SystemType,
    surface_config: SurfaceSystemConfig | None,
) -> None:
    """For surface system types, require a ``SurfaceSystemConfig`` run argument."""
    if not get_system_policy(system_type).uses_surface:
        return
    if not isinstance(surface_config, SurfaceSystemConfig):
        raise SCGOValidationError(
            f"system_type={system_type!r} requires the run surface_config argument "
            "to be a SurfaceSystemConfig."
        )


def _require_top_level_surface_consistency(
    *,
    explicit_config: Any,
    system_type: SystemType,
    surface_config: SurfaceSystemConfig | None,
    source_label: str,
) -> None:
    """Validate a top-level ``surface_config`` against run-level surface/system type.

    Shared by the go_params and ts_params top-level coherence checks, which are
    otherwise identical apart from the ``source_label`` that appears in messages.
    """
    policy = get_system_policy(system_type)
    resolved = explicit_config or surface_config
    if policy.uses_surface:
        if not isinstance(resolved, SurfaceSystemConfig):
            raise SCGOValidationError(
                "GO/TS coherence error: surface system types require "
                f"{source_label} or run surface_config=."
            )
        if (
            surface_config is not None
            and explicit_config is not None
            and explicit_config != surface_config
        ):
            raise SCGOValidationError(
                "GO/TS coherence error: "
                f"{source_label} disagrees with run surface_config."
            )
    elif resolved is not None:
        raise SCGOValidationError(
            "GO/TS coherence error: "
            f"{source_label} is set but "
            f"run system_type={system_type!r} is non-surface."
        )


def _validate_go_ts_param_coherence(
    *,
    go_prepared: dict[str, Any],
    ts_params: dict[str, Any],
    system_type: SystemType,
    surface_config: SurfaceSystemConfig | None,
) -> None:
    """Validate GO/TS params coherence against run-level system definition."""
    _reject_slot_identity_keys(go_prepared)
    _require_top_level_surface_consistency(
        explicit_config=go_prepared.get("surface_config"),
        system_type=system_type,
        surface_config=surface_config,
        source_label="go_params['surface_config']",
    )
    _require_top_level_surface_consistency(
        explicit_config=ts_params.get("surface_config"),
        system_type=system_type,
        surface_config=surface_config,
        source_label="ts_params['surface_config']",
    )


def _merge_adsorbate_context_into_params(
    base: dict[str, Any] | None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Attach adsorbate/surface init context for :func:`_run_go_trials` / GA."""
    out = _copy_params(base)
    out.update({k: v for k, v in kwargs.items() if v is not None})
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
    if go_params is not None:
        _reject_slot_identity_keys(go_params)
    merged = initialize_params(go_params)
    _reject_slot_identity_keys(merged)
    if surface_config is not None and merged.get("surface_config") is None:
        merged = _copy_params(merged)
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
    if merged_go is not None:
        _reject_slot_identity_keys(merged_go)
    merged = initialize_ts_params(
        ts_params,
        system_type=system_type,
        surface_config=surface_config,
        go_params=merged_go,
    )
    if surface_config is not None and merged.get("surface_config") is None:
        merged = _copy_params(merged)
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


def _default_go_ts_output_path(
    composition: list[str],
    *,
    go_params: dict[str, Any],
    output_stem: str | None,
    output_root: str | Path | None,
    system_type: SystemType | None = None,
    adsorbate_definition: AdsorbateDefinition | None = None,
    surface_config: SurfaceSystemConfig | None = None,
) -> Path:
    stem = output_stem or resolve_run_path_key(
        composition,
        system_type=system_type,
        adsorbate_definition=adsorbate_definition,
        surface_config=surface_config,
        params=go_params,
    )
    path = resolve_campaign_root_from_args(
        None,
        output_root=output_root,
        output_stem=stem,
        path_key=stem,
        calc_slug=calculator_slug_from_go_params(go_params),
    )
    if output_root is None:
        logger.info("No output_dir provided; using default campaign root %s", path)
    return path


def _log_completion(kind: str, *, elapsed_s: float, details: str) -> None:
    logger.info("%s completed in %.2f s (%s)", kind, elapsed_s, details)


def format_completion_details(
    *,
    compositions: int | None = None,
    minima: int | None = None,
    successful_nebs: tuple[int, int] | None = None,
    output_dir: str | Path | None = None,
) -> str:
    """Build a uniform ``details=`` string for ``_log_completion``.

    Fields are always rendered in the same order (``compositions``, ``minima``,
    ``successful_nebs``, ``output_dir``) and omitted when not applicable.
    """
    parts: list[str] = []
    if compositions is not None:
        parts.append(f"compositions={compositions}")
    if minima is not None:
        parts.append(f"minima={minima}")
    if successful_nebs is not None:
        ok, total = successful_nebs
        parts.append(f"successful_nebs={ok}/{total}")
    if output_dir is not None:
        parts.append(f"output_dir={output_dir}")
    return " ".join(parts)


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


def _reject_system_keys(
    params: dict[str, Any], *, context: str, kind: str = "go"
) -> None:
    """Reject ``system_type`` in go/ts params (use the run-function argument).

    Top-level ``surface_config`` in params is allowed; only ``system_type`` remains
    forbidden here.
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
    st, params_prep, ads_def, ads_temp, comp, resolved_surface = _prepare_run_context(
        composition,
        system_type=system_type,
        surface_config=surface_config,
        params=params,
        adsorbates=adsorbates,
        context="run_go",
    )
    eff_seed = resolve_workflow_seed(seed_kw=seed, go_params=params)
    eff_params = _merge_adsorbate_context_into_params(
        params_prep,
        adsorbate_definition=ads_def,
        adsorbate_fragment_template=ads_temp,
    )
    out_path = _resolved_path(output_dir)
    path_key = resolve_run_path_key(
        comp,
        system_type=st,
        adsorbate_definition=ads_def,
        surface_config=resolved_surface,
        params=eff_params,
    )
    # ``run_go``'s ``output_dir`` *is* the ``{path_key}_searches`` directory; the
    # shared campaign root is its parent.
    campaign_root = resolve_campaign_root_from_args(output_dir, path_key=path_key)
    searches_dir = str(
        resolve_go_searches_dir(output_dir, path_key)
        if output_dir is not None
        else formula_searches_dir(campaign_root, path_key)
    )
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
    calculator_for_global_optimization: Calculator | None,
    surface_config: SurfaceSystemConfig | None,
    system_type: SystemType | None,
    adsorbates: AdsorbatesInput | None,
) -> RunGOCampaignContext:
    st = _require_system_type(system_type, "run_go_campaign")
    resolved_surface = _resolve_surface_config(surface_config, params)
    if params is not None:
        _reject_system_keys(params, context="run_go_campaign")
        _reject_slot_identity_keys(params)
    validate_system_type_settings(system_type=st, surface_config=resolved_surface)
    params_prep = _with_surface_on_params(params, surface_config=resolved_surface)
    eff_seed = resolve_workflow_seed(seed_kw=seed, go_params=params)
    preset_ads_def = (
        extract_adsorbate_definition_from_params(params_prep)
        if adsorbates is None
        else None
    )
    full_compositions: list[list[str]] = []
    composition_adsorbate: list[
        tuple[AdsorbateDefinition | None, Atoms | list[Atoms] | None]
    ] = []
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
        composition_adsorbate.append((ads_def, ads_temp))
    out_path = _resolved_path(output_dir)
    first_ads, _first_temp = (
        composition_adsorbate[0] if composition_adsorbate else (None, None)
    )
    campaign_path_key = resolve_run_path_key(
        full_compositions[0],
        system_type=st,
        adsorbate_definition=first_ads,
        surface_config=resolved_surface,
        params=params_prep,
    )
    campaign_root = str(
        resolve_campaign_root_from_args(output_dir, path_key=campaign_path_key)
    )
    return RunGOCampaignContext(
        compositions=full_compositions,
        system_type=st,
        params=params_prep,
        seed=eff_seed,
        run_id=run_id,
        clean=clean,
        output_dir=out_path,
        verbosity=verbosity,
        calculator_for_global_optimization=calculator_for_global_optimization,
        output_summary_dir=campaign_root,
        composition_adsorbate=composition_adsorbate,
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
    if go_params is not None:
        _reject_system_keys(go_params, context=context_name)
        _reject_slot_identity_keys(go_params)
    if ts_params is not None:
        _reject_system_keys(ts_params, context=context_name, kind="ts")
    resolved_surface = _resolve_surface_config(surface_config, go_params)
    if resolved_surface is None:
        resolved_surface = _resolve_surface_config(None, ts_params)
    if surface_config is not None:
        validate_system_type_settings(system_type=st, surface_config=surface_config)
    go_mat, ts_mat = _resolve_go_ts_params(
        system_type=st,
        surface_config=resolved_surface,
        go_params=go_params,
        ts_params=ts_params,
    )
    eff_seed = resolve_workflow_seed(seed_kw=seed, go_params=go_mat, ts_params=ts_mat)
    go_prep = _with_surface_on_params(go_mat, surface_config=resolved_surface)
    policy = get_system_policy(st)
    core_comp = _as_composition(composition, allow_empty=policy.slab_is_search_target)
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
        surface_config=resolved_surface,
    )
    _validate_go_ts_surface_config(
        system_type=st,
        surface_config=resolved_surface,
    )
    go_local = _merge_adsorbate_context_into_params(
        go_prep,
        adsorbate_definition=ads_def,
        adsorbate_fragment_template=ads_temp,
    )
    ts_kwargs = _coerce_ts_for_runner(
        ts_mat, fn_name=context_name, system_type=st, surface_config=resolved_surface
    )
    out_path = _resolved_path(output_dir) or _default_go_ts_output_path(
        comp,
        go_params=go_mat,
        output_stem=output_stem,
        output_root=output_root,
        system_type=st,
        adsorbate_definition=ads_def,
        surface_config=resolved_surface,
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
    if ts_params is not None:
        _reject_system_keys(ts_params, context=context_name, kind="ts")
    resolved_surface = _resolve_surface_config(surface_config, ts_params)
    st, _, ads_def, _ads_temp, comp, resolved_surface = _prepare_run_context(
        composition,
        system_type=system_type,
        surface_config=resolved_surface,
        params=None,
        adsorbates=adsorbates,
        context=context_name,
    )
    ts_mat = _resolve_ts_params(
        ts_params, system_type=st, surface_config=resolved_surface
    )
    ts_base = initialize_ts_params(
        None, system_type=st, surface_config=resolved_surface
    )
    eff_seed = resolve_workflow_seed(seed_kw=seed, ts_params=ts_mat)
    ts_kwargs = _coerce_ts_for_runner(
        ts_mat, fn_name=context_name, system_type=st, surface_config=resolved_surface
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
