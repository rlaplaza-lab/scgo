"""Model-identity kwargs shared by TorchSim relaxer builders."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from scgo.calculators.dispersion import (
    MATPES_R2SCAN_MODEL,
    dispersion_xc_for_mace_kwargs,
)
from scgo.exceptions import SCGOValidationError

_CANONICAL_MODEL_KIND = {
    "mace": "mace",
    "fairchem": "fairchem",
    "uma": "fairchem",
    "upet": "upet",
    "metatomic": "upet",
}


def canonical_model_kind(name: str) -> str | None:
    """``mace``, ``fairchem`` (``uma``), or ``upet`` (``metatomic``). Unknown is ``None``."""
    return _CANONICAL_MODEL_KIND.get(str(name).strip().lower())


def _as_optional_str(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def build_torchsim_params(
    calc_name: str,
    calculator_kwargs: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Model fields for a calculator name or relaxer kind. No device or live model."""
    ck = calculator_kwargs or {}
    kind = canonical_model_kind(calc_name)
    if kind == "fairchem":
        return {
            "model_kind": "fairchem",
            "fairchem_model_name": str(ck["model_name"]),
            "fairchem_task_name": _as_optional_str(ck.get("task_name")),
        }
    if kind == "upet":
        return {
            "model_kind": "upet",
            "upet_model_name": _as_optional_str(ck.get("model_name")),
            "upet_version": ck.get("version"),
            "upet_checkpoint_path": ck.get("checkpoint_path"),
            "upet_non_conservative": bool(ck.get("non_conservative", False)),
        }
    if kind == "mace":
        return {
            "model_kind": "mace",
            "mace_model_name": ck.get("model_name", MATPES_R2SCAN_MODEL),
            "dispersion_xc": dispersion_xc_for_mace_kwargs(ck),
        }
    raise SCGOValidationError(f"Unknown TorchSim calculator {calc_name!r}")
