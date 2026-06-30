"""Atomic radii with ASE gap-filling, interpolation, and per-element caching.

ASE's ``vdw_radii`` table has NaN for many transition metals and lanthanides.
Missing values are resolved once per element via linear interpolation (or
extrapolation) along atomic number, with a scaled-covalent fallback for VdW.
Resolved values are cached so production runs do not repeat work or log noise.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from ase.data import atomic_numbers, chemical_symbols, covalent_radii, vdw_radii

if TYPE_CHECKING:
    from ase import Atoms

# NOTE: Keep stdlib logging here to avoid early package-import cycles during
# scgo bootstrap (initialization imports can execute before scgo.utils is ready).
logger = logging.getLogger(__name__)

VDW_COVALENT_FALLBACK_SCALE = 1.3

_patched_log_keys: set[tuple[str, str]] = set()


def _is_valid_radius(value: float) -> bool:
    return bool(np.isfinite(value) and value > 0)


def _interpolate_radius_at_z(z: int, radii: np.ndarray) -> float | None:
    """Linearly interpolate or extrapolate a radius at atomic number ``z``.

    Scans left and right for the nearest elements with finite positive radii.
    """
    n = len(radii)
    if z <= 0 or z >= n:
        return None

    left: tuple[int, float] | None = None
    for zz in range(z - 1, 0, -1):
        if zz < n and _is_valid_radius(float(radii[zz])):
            left = (zz, float(radii[zz]))
            break

    right: tuple[int, float] | None = None
    for zz in range(z + 1, n):
        if _is_valid_radius(float(radii[zz])):
            right = (zz, float(radii[zz]))
            break

    if left is not None and right is not None:
        left_z, left_val = left
        right_z, right_val = right
        weight = (z - left_z) / (right_z - left_z)
        return left_val + weight * (right_val - left_val)

    if left is not None:
        left_z, left_val = left
        for zz in range(left_z - 1, 0, -1):
            if zz < n and _is_valid_radius(float(radii[zz])):
                weight = (z - zz) / (left_z - zz)
                return float(radii[zz]) + weight * (left_val - float(radii[zz]))
        return left_val

    if right is not None:
        right_z, right_val = right
        for zz in range(right_z + 1, n):
            if _is_valid_radius(float(radii[zz])):
                weight = (z - right_z) / (zz - right_z)
                return right_val + weight * (float(radii[zz]) - right_val)
        return right_val

    return None


def _log_patch_once(kind: str, symbol: str, message: str) -> None:
    key = (kind, symbol)
    if key in _patched_log_keys:
        return
    _patched_log_keys.add(key)
    logger.info(message)


def _resolve_ase_radius(
    symbol: str,
    *,
    kind: str,
    radii_table: np.ndarray,
    fallback: float | None = None,
) -> float:
    try:
        z = atomic_numbers[symbol]
    except KeyError as exc:
        raise ValueError(
            f"Unknown element symbol: {symbol}. Could not find {kind} radius."
        ) from exc

    raw = float(radii_table[z])
    if _is_valid_radius(raw):
        return raw

    patched = _interpolate_radius_at_z(z, radii_table)
    if patched is not None:
        _log_patch_once(
            kind,
            symbol,
            f"{kind.capitalize()} radius for {symbol} is missing/NaN in ASE; "
            f"using interpolated value {patched:.3f} Å",
        )
        return patched

    if fallback is not None:
        _log_patch_once(
            kind,
            symbol,
            f"{kind.capitalize()} radius for {symbol} is missing/NaN in ASE; "
            f"using fallback value {fallback:.3f} Å",
        )
        return fallback

    raise ValueError(
        f"Could not resolve {kind} radius for {symbol}: ASE value is invalid "
        "and interpolation/extrapolation failed."
    )


@lru_cache(maxsize=256)
def get_covalent_radius(symbol: str) -> float:
    """Return the covalent radius for ``symbol`` in Angstroms."""
    return _resolve_ase_radius(symbol, kind="covalent", radii_table=covalent_radii)


@lru_cache(maxsize=256)
def get_vdw_radius(symbol: str) -> float:
    """Return the van-der-Waals radius for ``symbol`` in Angstroms."""
    try:
        z = atomic_numbers[symbol]
    except KeyError as exc:
        raise ValueError(
            f"Unknown element symbol: {symbol}. Could not find vdw radius."
        ) from exc

    raw = float(vdw_radii[z])
    if _is_valid_radius(raw):
        return raw

    return _resolve_ase_radius(
        symbol,
        kind="vdw",
        radii_table=vdw_radii,
        fallback=get_covalent_radius(symbol) * VDW_COVALENT_FALLBACK_SCALE,
    )


def clear_atomic_radii_cache() -> None:
    """Clear cached radii and one-shot patch logs (mainly for tests)."""
    get_covalent_radius.cache_clear()
    get_covalent_radius_by_z.cache_clear()
    get_vdw_radius.cache_clear()
    _patched_log_keys.clear()


@lru_cache(maxsize=256)
def get_covalent_radius_by_z(z: int) -> float:
    """Return the covalent radius for atomic number ``z`` in Angstroms."""
    return get_covalent_radius(chemical_symbols[int(z)])


def build_blmin_from_zs(
    zs: Iterable[int],
    ratio: float = 0.7,
) -> dict[tuple[int, int], float]:
    """Build an ASE-compatible blmin table using scgo gap-filled covalent radii."""
    unique = sorted({int(z) for z in zs})
    out: dict[tuple[int, int], float] = {}
    for i, zi in enumerate(unique):
        ri = get_covalent_radius_by_z(zi)
        for zj in unique[i:]:
            rj = get_covalent_radius_by_z(zj)
            dist = (ri + rj) * ratio
            out[(zi, zj)] = dist
            if zi != zj:
                out[(zj, zi)] = dist
    return out


def build_blmin(
    symbols: Iterable[str], ratio: float = 0.7
) -> dict[tuple[int, int], float]:
    """Build an ASE-compatible blmin table for the given element symbols."""
    zs = [atomic_numbers[str(s)] for s in symbols]
    return build_blmin_from_zs(zs, ratio)


def cluster_passes_ga_blmin(
    atoms: Atoms,
    blmin_ratio: float,
) -> bool:
    """Return True if ``atoms`` satisfies ASE GA steric checks at ``blmin_ratio``."""
    from ase_ga.utilities import atoms_too_close

    blmin = build_blmin_from_zs(atoms.get_atomic_numbers(), ratio=blmin_ratio)
    return not atoms_too_close(atoms, blmin, use_tags=False)


def resolve_steric_floor(
    min_distance_factor: float,
    blmin_ratio: float | None,
) -> float:
    """Minimum clash factor for placement: at least ``blmin_ratio`` when set."""
    if blmin_ratio is None:
        return min_distance_factor
    return max(min_distance_factor, blmin_ratio)
