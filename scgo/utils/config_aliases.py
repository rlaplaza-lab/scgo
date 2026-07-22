"""Helpers for aliased dataclass construction kwargs."""

from __future__ import annotations

from typing import Any

from scgo.exceptions import SCGOValidationError

_UNSET: Any = object()


def resolve_aliased_float(
    canonical_name: str,
    canonical: Any,
    alias_name: str,
    alias: Any,
    default: float,
) -> float:
    """Pick ``canonical`` or ``alias`` (or ``default``); error if both disagree."""
    c_set = canonical is not _UNSET
    a_set = alias is not _UNSET
    if c_set and a_set and float(canonical) != float(alias):
        raise SCGOValidationError(
            f"conflicting {canonical_name}={canonical!r} and {alias_name}={alias!r}"
        )
    if c_set:
        return float(canonical)
    if a_set:
        return float(alias)
    return float(default)
