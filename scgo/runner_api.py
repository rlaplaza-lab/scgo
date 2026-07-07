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
"""

from __future__ import annotations

import copy
import os
import re
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from ase import Atoms

from scgo.cluster_adsorbate.config import ClusterAdsorbateConfig
from scgo.param_presets import get_default_params
from scgo.surface.config import SurfaceSystemConfig
from scgo.system_types import (
    AdsorbatesInput,
    AdsorbateDefinition,
    SystemType,
    build_adsorbate_definition_from_inputs,
    get_system_policy,
    resolve_connectivity_factor,
    validate_adsorbate_definition,
    validate_system_type_settings,
)
from scgo.ts_search.transition_state_run import (
    run_transition_state_campaign as _ts_campaign,
    run_transition_state_search as _ts_search,
)
from scgo.utils.helpers import get_cluster_formula
from scgo.utils.logging import configure_logging, get_logger
from scgo.utils.output_paths import (
    resolve_go_campaign_searches_dir,
    resolve_go_searches_dir,
    resolve_go_ts_pipeline_paths,
)
from scgo.utils.ts_runner_kwargs import coerce_ts_params_to_runner_kwargs

from scgo.minima_search import run_trials
from scgo.utils.rng_helpers import ensure_rng
from scgo.utils.run_helpers import (
    cleanup_torch_cuda,
    get_calculator_class,
    initialize_params,
    initialize_ts_params,
    log_configuration,
    log_ts_configuration,
    prepare_algorithm_kwargs,
    validate_algorithm_params,
)
from scgo.utils.run_tracking import (
    ensure_run_id,
    get_run_directories,
    get_run_id_from_dir,
)
from scgo.utils.timing_report import (
    GO_TS_TIMING_JSON_FILENAME,
    build_timing_payload,
    log_timing_summary,
    sum_neb_seconds_from_ts_results,
    write_timing_file,
)
from scgo.utils.validation import validate_composition

type CompositionInput = str | list[str] | Atoms
_ALGO_KEYS = ("simple", "bh", "ga")
_LOGGER = get_logger(__name__)
_DEFAULT_GO_PARAMS: dict[str, Any] | None = None


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
    calculator_for_global_optimization: Any | None
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


def _as_composition(composition: CompositionInput) -> list[str]:
    if isinstance(composition, Atoms):
        return list(composition.get_chemical_symbols())
    elif isinstance(composition, str):
        return parse_composition_arg(composition)
    elif isinstance(composition, list):
        if not composition:
            raise ValueError("composition list must not be empty")
        return [str(s) for s in composition]
    else:
        raise TypeError(
            f"composition must be str, list[str], or Atoms, got {type(composition).__name__}"
        )


def _as_composition_list(items: Iterable[CompositionInput]) -> list[list[str]]:
    out = [_as_composition(x) for x in items]
    if not out:
        raise ValueError("compositions iterable must not be empty")
    return out


def _resolved_path(path: str | Path | None) -> Path | None:
    return Path(path).expanduser().resolve() if path is not None else None


def _require_system_type(system_type: SystemType | None, fn_name: str) -> SystemType:
    if system_type is None:
        raise ValueError(f"system_type is required for {fn_name}.")
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
    ads_def, ads_template, full_comp = build_adsorbate_definition_from_inputs(
        system_type=st, composition=comp, adsorbates=adsorbates, context=context
    )
    validate_adsorbate_definition(
        system_type=st,
        composition=full_comp,
        adsorbate_definition=ads_def,
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
        raise ValueError(
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
        raise ValueError(
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
            raise ValueError(
                "GO/TS coherence error: surface system types require "
                "go_params['surface_config'] or run surface_config=."
            )
        if (
            surface_config is not None
            and go_prepared.get("surface_config") is not None
            and go_prepared.get("surface_config") != surface_config
        ):
            raise ValueError(
                "GO/TS coherence error: go_params['surface_config'] disagrees with "
                "run surface_config."
            )
    elif go_surface_config is not None:
        raise ValueError(
            "GO/TS coherence error: go_params['surface_config'] is set but "
            f"run system_type={system_type!r} is non-surface."
        )

    optimizer_params = go_prepared.get("optimizer_params") or {}
    for algo in _ALGO_KEYS:
        slot = optimizer_params.get(algo)
        if slot is None:
            continue
        if not isinstance(slot, dict):
            raise ValueError(f"go_params['optimizer_params']['{algo}'] must be a dict.")
        slot_system_type = slot.get("system_type")
        default_slot_st = _default_optimizer_system_type(algo)
        if (
            slot_system_type is not None
            and slot_system_type != system_type
            and slot_system_type != default_slot_st
        ):
            raise ValueError(
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
                raise ValueError(
                    "GO/TS coherence error: "
                    f"go_params['optimizer_params']['{algo}']['surface_config'] "
                    "disagrees with run surface_config."
                )
        elif slot_surface_config is not None:
            raise ValueError(
                "GO/TS coherence error: go_params surface_config is set but "
                f"run system_type={system_type!r} is non-surface."
            )

    ts_surface_config = ts_params.get("surface_config") or surface_config
    if policy.uses_surface:
        if not isinstance(ts_surface_config, SurfaceSystemConfig):
            raise ValueError(
                "GO/TS coherence error: surface system types require "
                "ts_params['surface_config'] or run surface_config=."
            )
        if (
            surface_config is not None
            and ts_params.get("surface_config") is not None
            and ts_params.get("surface_config") != surface_config
        ):
            raise ValueError(
                "GO/TS coherence error: ts_params['surface_config'] disagrees with "
                "run surface_config."
            )
    elif ts_surface_config is not None:
        raise ValueError(
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
        raise ValueError(
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
        raise TypeError(f"{label} must be int-like, got {value!r}") from e


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
        raise ValueError(f"Inconsistent random seeds: {desc}")
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
            raise ValueError(
                "run argument surface_config must match go_params['surface_config'] "
                "when both are set."
            )
        op = out.setdefault("optimizer_params", {})
        for key in _ALGO_KEYS:
            if key not in op:
                continue
            slot = op[key]
            if not isinstance(slot, dict):
                raise ValueError(
                    f"optimizer_params['{key}'] must be a dict when using go_params['surface_config']"
                )
            ex = slot.get("surface_config")
            if ex is None:
                slot["surface_config"] = surface_config
            elif ex != surface_config:
                raise ValueError(
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
                raise ValueError(
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
    forbidden = ("system_type", "surface_config")
    if kind == "go":
        forbidden = ("system_type",)
    if kind == "ts":
        forbidden = ("system_type",)
    for key in forbidden:
        if params.get(key) is not None:
            guidance = "Use the run function argument instead."
            if kind == "ts" and key == "system_type":
                guidance = (
                    "Use the run function system_type argument; "
                    "ts_params['surface_config'] is allowed."
                )
            raise ValueError(
                f"{context} does not allow top-level {kind}_params['{key}']. {guidance}"
            )


ScgoMinimaAlgorithm = Literal["simple", "bh", "ga"]


def select_scgo_minima_algorithm(
    n_atoms: int, system_type: SystemType
) -> ScgoMinimaAlgorithm:
    """Select global optimizer for composition size and system type.

    Uses the mobile-atom count (core + adsorbate symbols for adsorbate modes).
    Plain ``gas_cluster`` alone may use ``simple`` for 1–2 atoms; adsorbate and
    surface modes never select ``simple``.
    """
    policy = get_system_policy(system_type)
    simple_allowed = not policy.uses_surface and not policy.has_adsorbate
    if n_atoms <= 2 and simple_allowed:
        return "simple"
    if n_atoms == 3:
        if policy.has_adsorbate:
            return "ga"
        return "bh"
    return "ga"


def _run_go_trials(
    composition: list[str],
    system_type: SystemType,
    params: dict | None = None,
    seed: int | None = None,
    verbosity: int = 1,
    run_id: str | None = None,
    clean: bool = False,
    output_dir: str | Path | None = None,
    calculator_for_global_optimization: Any | None = None,
    *,
    params_already_merged: bool = False,
) -> list[tuple[float, Atoms]]:
    """Run global optimization for a composition; return unique minima sorted by energy."""
    configure_logging(verbosity)
    logger = get_logger(__name__)

    validate_composition(composition, allow_empty=False, allow_tuple=False)

    # Initialize and merge params with defaults
    if not params_already_merged:
        params = initialize_params(params)
    else:
        params = copy.deepcopy(params or {})

    # Validate calculator availability
    calculator_name = params.get("calculator", "MACE")
    _ = get_calculator_class(calculator_name)

    # Validate params structure - rng should not be in optimizer_params
    for algo in ["bh", "ga"]:
        algo_params = params["optimizer_params"].get(algo, {})
        if "rng" in algo_params:
            raise ValueError(
                f'"rng" should not be in params["optimizer_params"]["{algo}"]. '
                f'Use the "seed" parameter instead.'
            )

    # Prefer explicit function seed arg; fall back to params['seed'] if provided
    if seed is None:
        seed = params.get("seed", None)

    # Convert seed to generator at API boundary
    rng = ensure_rng(seed)

    n_atoms = len(composition)
    cluster_formula = get_cluster_formula(composition)
    main_output_dir = str(resolve_go_searches_dir(output_dir, cluster_formula))

    # Algorithm selection: Use simple optimization for 1-2 atoms, BH for 3, GA for larger
    chosen_go = select_scgo_minima_algorithm(n_atoms, system_type)
    if chosen_go == "simple":
        logger.info(
            f"Selected simple optimization for {n_atoms}-atom cluster (trivial structure)"
        )
    elif chosen_go == "bh":
        logger.info(
            f"Selected Basin Hopping for {n_atoms}-atom cluster (small cluster)"
        )
    else:
        logger.info(f"Selected Genetic Algorithm for {n_atoms}-atom cluster")

    # Extract algorithm-specific parameters without mutation
    algo_params = params["optimizer_params"].get(chosen_go, {})

    user_params = None if params_already_merged else params
    params_base = None if params_already_merged else get_default_params()

    # Validate algorithm-specific parameters
    validate_algorithm_params(algo_params, chosen_go, verbosity)

    # Get calculator kwargs if provided
    calculator_kwargs = params.get("calculator_kwargs", {})

    # Unified parameter preparation (resolves auto params, fitness strategy, diversity, etc.)
    global_optimizer_kwargs = prepare_algorithm_kwargs(
        algo_params=algo_params,
        params=params,
        composition=composition,
        chosen_go=chosen_go,
        system_type=system_type,
    )

    # Validate that no unexpected top-level keys were provided
    expected_top_level_keys = {
        "validate_with_hessian",
        "calculator",
        "calculator_kwargs",
        "surface_config",
        "fmax_threshold",
        "check_hessian",
        "imag_freq_threshold",
        "optimizer_params",
        "fitness_strategy",
        "diversity_reference_db",
        "diversity_max_references",
        "diversity_update_interval",
        "tag_final_minima",
        "connectivity_factor",
        "allow_cluster_fragmentation",
        "allow_adsorbate_surface_detachment",
        "enforce_adsorbate_subgraph_integrity",
        "freeze_adsorbate_internal_geometry",
        "adsorbate_definition",
        "adsorbate_fragment_template",
        "cluster_adsorbate_config",
        "seed",  # seed is handled separately at API boundary, not passed to algorithms
    }
    unexpected_keys = set(params.keys()) - expected_top_level_keys
    if unexpected_keys:
        raise ValueError(
            f"Unexpected parameter keys: {sorted(unexpected_keys)}. "
            f"Expected keys: {sorted(expected_top_level_keys)}"
        )

    # Log the final configuration being used
    log_configuration(
        params=params,
        chosen_go=chosen_go,
        cluster_formula=cluster_formula,
        n_atoms=n_atoms,
        global_optimizer_kwargs=global_optimizer_kwargs,
        verbosity=verbosity,
        user_params=user_params,
        params_base=params_base,
    )

    final_unique_minima = run_trials(
        composition=composition,
        global_optimizer=chosen_go,
        global_optimizer_kwargs=global_optimizer_kwargs,
        output_dir=main_output_dir,
        calculator_for_global_optimization=(
            calculator_for_global_optimization
            if calculator_for_global_optimization is not None
            else get_calculator_class(params["calculator"])(**calculator_kwargs)
        ),
        validate_with_hessian=params.get("validate_with_hessian", False),
        fmax_threshold=params.get("fmax_threshold", 0.05),
        check_hessian=params.get("check_hessian", True),
        imag_freq_threshold=params.get("imag_freq_threshold", 50.0),
        validation_n_jobs=params.get("validation_n_jobs", 1),
        tag_final_minima=params.get("tag_final_minima", True),
        rng=rng,
        run_id=run_id,
        clean=clean,
    )

    cleanup_torch_cuda(logger=logger)

    return final_unique_minima


def parse_composition_arg(comp_str: str) -> list[str]:
    """Supports two formats:
    - Comma-separated symbols: "Pt,Pt,Au"
    - Compact formula: "Pt3Au" or "AuPt2"
    """
    comp_str = comp_str.strip()
    if "," in comp_str:
        parts = [p.strip() for p in comp_str.split(",") if p.strip()]
        # Normalize element symbols (e.g., 'pt' -> 'Pt')
        normalized = [p[0].upper() + p[1:].lower() if len(p) > 0 else p for p in parts]
        return normalized

    # Parse compact formula, e.g., "Pt3Au" or "pt3au" -> [("Pt", "3"), ("Au", "")]
    # Accept lower- or upper-case element symbols and optional integer counts
    token_re = re.compile(r"([A-Za-z]{1,2})(\d*)", flags=re.IGNORECASE)
    matches = token_re.findall(comp_str)
    if not matches:
        raise ValueError(f"Unable to parse composition string: {comp_str}")

    reconstructed = "".join(elem + count for elem, count in matches)
    if reconstructed.lower() != comp_str.lower():
        raise ValueError(f"Unable to parse composition string: {comp_str}")

    composition: list[str] = []
    for elem, count_str in matches:
        # Normalize capitalization: first letter uppercase, rest lowercase
        elem_norm = elem[0].upper() + elem[1:].lower() if len(elem) > 0 else elem
        count = int(count_str) if count_str else 1
        if count == 0:
            raise ValueError(
                f"Element '{elem_norm}' has zero count in composition string: '{comp_str}'"
            )
        composition.extend([elem_norm] * count)

    return composition


def build_one_element_compositions(
    element: str, min_atoms: int, max_atoms: int
) -> list[list[str]]:
    """Composition list for mono-element size scans (min_atoms..max_atoms)."""
    if not element or not isinstance(element, str):
        raise ValueError("element must be a non-empty string")
    if min_atoms < 1:
        raise ValueError("min_atoms must be >= 1")
    if max_atoms < min_atoms:
        raise ValueError("max_atoms must be >= min_atoms")
    return [[element] * n_atoms for n_atoms in range(min_atoms, max_atoms + 1)]


def build_two_element_compositions(
    element1: str, element2: str, min_atoms: int, max_atoms: int
) -> list[list[str]]:
    """Composition list for bimetallic size scans (min_atoms..max_atoms)."""
    if not element1 or not isinstance(element1, str):
        raise ValueError("element1 must be a non-empty string")
    if not element2 or not isinstance(element2, str):
        raise ValueError("element2 must be a non-empty string")
    if min_atoms < 1:
        raise ValueError("min_atoms must be >= 1")
    if max_atoms < min_atoms:
        raise ValueError("max_atoms must be >= min_atoms")
    compositions: list[list[str]] = []
    for n_atoms in range(min_atoms, max_atoms + 1):
        for i in range(n_atoms + 1):
            compositions.append([element1] * i + [element2] * (n_atoms - i))
    return compositions


def _run_go_campaign_compositions(
    compositions: Iterable[list[str]],
    system_type: SystemType,
    params: dict | None = None,
    seed: int | None = None,
    verbosity: int = 1,
    run_id: str | None = None,
    clean: bool = False,
    output_dir: str | Path | None = None,
    *,
    params_already_merged: bool = False,
) -> dict[str, list[tuple[float, Atoms]]]:
    """Run optimizations for an iterable of compositions; return mapping formula->minima."""
    if params_already_merged:
        params = copy.deepcopy(params or {})
    else:
        params = initialize_params(params)
    configure_logging(verbosity)

    # Validate params structure early: 'rng' must not be present inside
    # optimizer-specific params. Raise ValueError so callers get immediate
    # feedback instead of having the error swallowed during campaign
    # iteration.
    for algo in ["bh", "ga"]:
        algo_params = params["optimizer_params"].get(algo, {})
        if "rng" in algo_params:
            raise ValueError(
                f'"rng" should not be in params["optimizer_params"]["{algo}"]. '
                f'Use the "seed" parameter instead.'
            )
    logger = get_logger(__name__)

    # Generate run_id once at campaign start if not provided
    run_id = ensure_run_id(run_id, verbosity=verbosity, logger=logger)

    # Prefer explicit function seed arg; fall back to params['seed'] if provided
    if seed is None:
        seed = params.get("seed", None)

    # Convert seed to generator at API boundary
    rng = ensure_rng(seed)

    all_results = {}
    compositions_list = list(compositions)
    if not compositions_list:
        raise ValueError("compositions iterable must not be empty")
    num_compositions = len(compositions_list)
    logger.info(f"Starting campaign for {num_compositions} compositions.")

    # Create calculator once and reuse it for all compositions to avoid file handle leaks
    calculator_kwargs = params.get("calculator_kwargs", {})
    calculator_for_global_optimization = get_calculator_class(params["calculator"])(
        **calculator_kwargs,
    )

    for i, composition in enumerate(compositions_list):
        formula_str = get_cluster_formula(composition)
        if verbosity >= 1:
            logger.info(f"\n{'=' * 60}")
            logger.info(
                f"Running minima search for {formula_str} ({i + 1}/{num_compositions})"
            )
            logger.info(f"{'=' * 60}")

        comp_seed = int(rng.integers(0, 2**63 - 1))
        trial_output_dir = resolve_go_campaign_searches_dir(output_dir, formula_str)
        trial_output_dir_str = (
            str(trial_output_dir) if trial_output_dir is not None else None
        )

        try:
            results = _run_go_trials(
                composition,
                system_type,
                params,
                seed=comp_seed,
                verbosity=verbosity,
                run_id=run_id,
                clean=clean,
                output_dir=trial_output_dir_str,
                calculator_for_global_optimization=calculator_for_global_optimization,
                params_already_merged=True,
            )
            # Always add results (possibly empty) so the API returns a key for each
            # requested composition; this makes the function predictable for
            # downstream consumers and tests.
            all_results[formula_str] = results
            if not results and verbosity >= 1:
                logger.warning(f"No minima found for {formula_str} (results empty)")
            if verbosity >= 1:
                logger.info(f"Finished processing {formula_str}.")
                logger.info(f"  Returned {len(results)} final minima for {formula_str}")
        except (RuntimeError, ValueError, OSError, sqlite3.DatabaseError) as e:
            # Enhanced error logging for HPC debugging
            error_details = [
                f"Failed to process {formula_str}: {e}",
                f"Working directory: {os.getcwd()}",
            ]
            if trial_output_dir:
                error_details.append(f"Output directory: {trial_output_dir}")
                if os.path.exists(trial_output_dir):
                    try:
                        files = os.listdir(trial_output_dir)
                        error_details.append(f"Output dir contents: {files}")
                    except OSError:
                        error_details.append(
                            "Output dir exists but cannot list contents"
                        )
                else:
                    error_details.append("Output directory does not exist")

            logger.error(" | ".join(error_details), exc_info=(verbosity >= 2))
            all_results[formula_str] = []
            if verbosity >= 1:
                logger.warning(
                    f"Skipping {formula_str} and continuing campaign "
                    f"({i + 1}/{num_compositions})"
                )
            continue

    # Best-effort: drop shared calculator reference and free CUDA memory to avoid
    # fragmentation when campaigns are run sequentially in the same process.
    del calculator_for_global_optimization
    cleanup_torch_cuda(logger=logger)

    return all_results


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
    _ = get_calculator_class(calculator_name)
    calculator_for_global_optimization = get_calculator_class(calculator_name)(
        **calculator_kwargs,
    )
    try:
        go_t0 = perf_counter()
        minima_list = _run_go_trials(
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

    ts_results = _ts_search(
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
        raise ValueError("element must be a non-empty string")
    if n_atoms < 1:
        raise ValueError("n_atoms must be >= 1")
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


def _prepare_run_go_context(
    composition: CompositionInput,
    *,
    params: dict[str, Any] | None,
    seed: int | None,
    verbosity: int,
    run_id: str | None,
    clean: bool,
    output_dir: str | Path | None,
    calculator_for_global_optimization: Any | None,
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
    full_compositions: list[list[str]] = []
    for composition_item in _as_composition_list(compositions):
        ads_def, ads_temp, full_comp = build_adsorbate_definition_from_inputs(
            system_type=st,
            composition=composition_item,
            adsorbates=adsorbates,
            context="run_go_campaign",
        )
        validate_adsorbate_definition(
            system_type=st,
            composition=full_comp,
            adsorbate_definition=ads_def,
            context="run_go_campaign",
        )
        full_compositions.append(full_comp)
        eff_params["adsorbate_definition"] = ads_def
        eff_params["adsorbate_fragment_template"] = ads_temp
    if adsorbates is not None:
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
    ads_def, ads_temp, comp = build_adsorbate_definition_from_inputs(
        system_type=st,
        composition=core_comp,
        adsorbates=adsorbates,
        context=context_name,
    )
    validate_adsorbate_definition(
        system_type=st,
        composition=comp,
        adsorbate_definition=ads_def,
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


def _execute_run_go_ts(context: RunGOTSContext) -> dict[str, Any]:
    return _run_go_ts_pipeline(
        context.composition,
        context.system_type,
        go_params=context.go_params,
        ts_kwargs=context.ts_kwargs,
        adsorbate_definition=context.adsorbate_definition,
        seed=context.seed,
        verbosity=context.verbosity,
        output_dir=context.output_dir,
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


def run_go(
    composition: CompositionInput,
    params: dict | None = None,
    seed: int | None = None,
    verbosity: int = 1,
    run_id: str | None = None,
    clean: bool = False,
    output_dir: str | Path | None = None,
    calculator_for_global_optimization: Any | None = None,
    surface_config: SurfaceSystemConfig | None = None,
    system_type: SystemType | None = None,
    adsorbates: AdsorbatesInput | None = None,
    log_summary: bool = True,
) -> list[tuple[float, Atoms]]:
    """Run global optimization trials for one composition."""
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
    t0 = perf_counter()
    minima = _execute_run_go(context)
    if log_summary:
        _log_completion(
            "run_go",
            elapsed_s=perf_counter() - t0,
            details=f"minima={len(minima)} output_dir={context.output_summary_dir}",
        )
    return minima


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
    I/O, or database errors), the error is logged, that formula maps to an empty
    list, and remaining compositions continue.
    """
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
    t0 = perf_counter()
    campaign = _execute_run_go_campaign(context)
    if log_summary:
        _log_completion(
            "run_go_campaign",
            elapsed_s=perf_counter() - t0,
            details=f"compositions={len(campaign)} output_dir={context.output_summary_dir}",
        )
    return campaign


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
    st = _require_system_type(system_type, "run_go_ts_campaign")
    validate_system_type_settings(system_type=st, surface_config=surface_config)
    if go_params is not None:
        _reject_system_keys(go_params, context="run_go_ts_campaign")
    if ts_params is not None:
        _reject_system_keys(ts_params, context="run_go_ts_campaign", kind="ts")
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
    for core_comp in _as_composition_list(compositions):
        ads_def, ads_temp, full_comp = build_adsorbate_definition_from_inputs(
            system_type=st,
            composition=core_comp,
            adsorbates=adsorbates,
            context="run_go_ts_campaign",
        )
        validate_adsorbate_definition(
            system_type=st,
            composition=full_comp,
            adsorbate_definition=ads_def,
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
    configure_logging(verbosity)
    log_ts_configuration(
        context.ts_params,
        context.ts_kwargs,
        verbosity=verbosity,
        user_params=ts_params,
        base=context.ts_base,
    )
    t0 = perf_counter()
    results = _ts_search(
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
    st = _require_system_type(system_type, "run_ts_campaign")
    validate_system_type_settings(system_type=st, surface_config=surface_config)
    if ts_params is not None:
        _reject_system_keys(ts_params, context="run_ts_campaign", kind="ts")
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
    for core in _as_composition_list(compositions):
        ads_def, _, full = build_adsorbate_definition_from_inputs(
            system_type=st,
            composition=core,
            adsorbates=adsorbates,
            context="run_ts_campaign",
        )
        validate_adsorbate_definition(
            system_type=st,
            composition=full,
            adsorbate_definition=ads_def,
            context="run_ts_campaign",
        )
        full_compositions.append(full)
    out_path = _resolved_path(output_dir)
    t0 = perf_counter()
    if ads_def:
        ts_kwargs["adsorbate_definition"] = ads_def
    campaign = _ts_campaign(
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


__all__ = [
    "CompositionInput",
    "log_go_ts_summary",
    "parse_composition_arg",
    "build_one_element_compositions",
    "build_two_element_compositions",
    "_run_go_trials",
    "_run_go_campaign_compositions",
    "_run_go_ts_pipeline",
    "_run_one_element_go_ts_pipeline",
    "resolve_workflow_seed",
    "run_go",
    "run_go_campaign",
    "run_go_ts",
    "run_go_ts_campaign",
    "run_ts_campaign",
    "run_ts_search",
]
