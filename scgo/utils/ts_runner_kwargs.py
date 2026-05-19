"""Flat TS dict → kwargs for :func:`run_transition_state_search`."""

from __future__ import annotations

from typing import Any

from scgo.constants import DEFAULT_ENERGY_TOLERANCE
from scgo.param_presets import get_ts_defaults
from scgo.surface.config import SurfaceSystemConfig
from scgo.system_types import SystemType, get_system_policy
from scgo.utils.torchsim_policy import resolve_ts_torchsim_flags


def coerce_ts_params_to_runner_kwargs(
    ts_params: dict[str, Any] | None,
    *,
    system_type: SystemType,
    surface_config: Any | None = None,
) -> dict[str, Any]:
    """Map ``get_ts_search_params`` output to ``run_transition_state_search`` kwargs.

    Missing NEB knobs (``neb_align_endpoints``, ``neb_interpolation_mic``, image
    counts, fmax, steps, ...) fall back to the per-system-type defaults in
    :data:`scgo.param_presets.TS_DEFAULTS_BY_SYSTEM_TYPE`, so a sparse
    user-built dict still gets policy-coherent values without relying on the
    runner's silent override.
    """
    if ts_params is None:
        raise ValueError(
            "ts_params is required. Build with get_ts_search_params(system_type=...)."
        )

    calc_name = str(ts_params["calculator"])
    if system_type not in SystemType.__args__:
        raise ValueError(
            f"Unsupported system_type={system_type!r}; "
            f"expected one of {SystemType.__args__!r}."
        )
    ts_defaults = get_ts_defaults(system_type)
    use_ts, use_pn = resolve_ts_torchsim_flags(
        calc_name,
        ts_params.get("use_torchsim"),
        ts_params.get("use_parallel_neb"),
    )
    ts_surface_config = ts_params.get("surface_config")
    if (
        surface_config is not None
        and ts_surface_config is not None
        and surface_config != ts_surface_config
    ):
        raise ValueError("run surface_config and ts_params['surface_config'] disagree.")
    resolved_surface_config = (
        surface_config if surface_config is not None else ts_surface_config
    )
    if get_system_policy(system_type).uses_surface and not isinstance(
        resolved_surface_config, SurfaceSystemConfig
    ):
        raise ValueError(
            f"system_type={system_type!r} requires surface_config in ts_params "
            "or as the run surface_config argument."
        )

    kwargs: dict[str, Any] = {
        "params": {
            "calculator": ts_params["calculator"],
            "calculator_kwargs": ts_params.get("calculator_kwargs") or {},
        },
        "system_type": system_type,
        "use_torchsim": use_ts,
        "use_parallel_neb": use_pn,
        "torchsim_params": {
            "force_tol": ts_params.get("torchsim_fmax", ts_defaults["torchsim_fmax"]),
            "max_steps": ts_params.get(
                "torchsim_max_steps", ts_defaults["torchsim_max_steps"]
            ),
        },
    }
    if str(ts_params.get("calculator", "")).strip().upper() == "UMA":
        ck = ts_params.get("calculator_kwargs", {}) or {}
        kwargs["torchsim_params"].update(
            {
                "model_kind": "fairchem",
                "fairchem_model_name": ck.get("model_name", "uma-s-1p2"),
                "fairchem_task_name": ck.get("task_name", "oc25"),
            }
        )

    # Keys without per-system defaults: pass through as-is (None when missing
    # is fine for the runner).
    passthrough_keys = (
        "write_timing_json",
        "max_pairs",
        "energy_gap_threshold",
        "similarity_tolerance",
        "similarity_pair_cor_max",
        "connectivity_factor",
        "allow_dissociative_adsorption",
    )
    for key in passthrough_keys:
        kwargs[key] = ts_params.get(key)
    kwargs["surface_config"] = resolved_surface_config

    # NEB knobs that vary per system_type: fall back to the defaults table.
    # torchsim_* defaults are consumed only in torchsim_params above; they are
    # not valid top-level kwargs for run_transition_state_search.
    for key in (
        "neb_align_endpoints",
        "neb_interpolation_mic",
        "neb_n_images",
        "neb_spring_constant",
        "neb_fmax",
        "neb_steps",
        "neb_climb",
        "neb_perturb_sigma",
        "neb_interpolation_method",
        "neb_tangent_method",
    ):
        kwargs[key] = ts_params.get(key, ts_defaults[key])

    # Generic (system-type-agnostic) defaults.
    generic_defaults = {
        "dedupe_minima": True,
        "minima_energy_tolerance": DEFAULT_ENERGY_TOLERANCE,
    }
    for key, def_val in generic_defaults.items():
        kwargs[key] = ts_params.get(key, def_val)

    return kwargs
