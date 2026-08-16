"""Connectivity-factor specs: global float or per-element / per-pair multipliers.

Bonded means ``distance <= threshold``. Thresholds:

- float ``f``: ``(r_i + r_j) * f``
- element dict: ``r_i * f_i + r_j * f_j`` (missing symbols use ``CONNECTIVITY_FACTOR``)
- pair entry ``("Pt", "C")`` / ``"Pt-C"``: ``(r_i + r_j) * f_ij`` (order-independent)
- mixed dict: pair entries override the element-derived threshold

"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from scgo.exceptions import SCGOValidationError
from scgo.initialization.initialization_config import CONNECTIVITY_FACTOR

# User-facing input: float, or mapping with element and/or pair keys.
ConnectivityFactorInput = float | Mapping[Any, float]

# Canonical pair key after normalize: sorted chemical symbols.
PairKey = tuple[str, str]

SpecKind = Literal["global", "element", "pair", "mixed"]


@dataclass(frozen=True)
class NormalizedConnectivityFactor:
    """Resolved connectivity multiplier used by bonded-pair checks.

    Exactly one of:

    - ``global_factor`` set (float input) and empty element/pair maps, or
    - ``global_factor`` is ``None`` and element and/or pair maps hold overrides
      (missing element symbols fall back to ``CONNECTIVITY_FACTOR``).

    Element/pair maps are stored as sorted tuples so the object is hashable
    (safe for frozen configs and cache keys).
    """

    global_factor: float | None
    element_items: tuple[tuple[str, float], ...]
    pair_items: tuple[tuple[PairKey, float], ...]

    def is_global(self) -> bool:
        return (
            self.global_factor is not None
            and not self.element_items
            and not self.pair_items
        )

    def kind(self) -> SpecKind:
        if self.is_global():
            return "global"
        has_el = bool(self.element_items)
        has_pair = bool(self.pair_items)
        if has_pair and has_el:
            return "mixed"
        if has_pair:
            return "pair"
        return "element"

    @property
    def element_factors(self) -> dict[str, float]:
        return dict(self.element_items)

    @property
    def pair_factors(self) -> dict[PairKey, float]:
        return dict(self.pair_items)


def _parse_pair_key(key: object) -> PairKey | None:
    """Return a sorted symbol pair, or ``None`` if ``key`` is not a pair key."""
    if isinstance(key, tuple) and len(key) == 2:
        a, b = key
        if isinstance(a, str) and isinstance(b, str) and a and b:
            return (a, b) if a <= b else (b, a)
        return None
    if isinstance(key, str) and "-" in key:
        parts = key.split("-")
        if len(parts) == 2 and parts[0] and parts[1]:
            a, b = parts[0], parts[1]
            return (a, b) if a <= b else (b, a)
    return None


def _validate_positive_factor(name: str, value: object) -> float:
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as e:
        raise SCGOValidationError(
            f"connectivity_factor {name} must be a positive number, got {value!r}"
        ) from e
    if not np.isfinite(f) or f <= 0.0:
        raise SCGOValidationError(
            f"connectivity_factor {name} must be a positive finite number, got {f}"
        )
    return f


def normalize_connectivity_factor(
    spec: ConnectivityFactorInput | NormalizedConnectivityFactor | None,
    *,
    name: str = "connectivity_factor",
) -> NormalizedConnectivityFactor:
    """Validate and canonicalize a connectivity-factor spec.

    ``None`` becomes the module default global factor (``CONNECTIVITY_FACTOR``).
    """
    if isinstance(spec, NormalizedConnectivityFactor):
        return spec
    if spec is None:
        return NormalizedConnectivityFactor(
            global_factor=float(CONNECTIVITY_FACTOR),
            element_items=(),
            pair_items=(),
        )
    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        f = _validate_positive_factor(name, spec)
        return NormalizedConnectivityFactor(
            global_factor=f, element_items=(), pair_items=()
        )
    if not isinstance(spec, Mapping):
        raise SCGOValidationError(
            f"{name} must be a float or a mapping of element/pair multipliers, "
            f"got {type(spec).__name__}"
        )
    if len(spec) == 0:
        raise SCGOValidationError(f"{name} mapping must not be empty")

    element_factors: dict[str, float] = {}
    pair_factors: dict[PairKey, float] = {}
    for key, raw in spec.items():
        pair = _parse_pair_key(key)
        if pair is not None:
            pair_factors[pair] = _validate_positive_factor(f"{name}[{key!r}]", raw)
            continue
        if isinstance(key, str) and key and "-" not in key:
            element_factors[key] = _validate_positive_factor(f"{name}[{key!r}]", raw)
            continue
        raise SCGOValidationError(
            f"{name} key {key!r} must be an element symbol (e.g. 'Pt') or a "
            "pair ('Pt-C' or ('Pt', 'C'))"
        )
    return NormalizedConnectivityFactor(
        global_factor=None,
        element_items=tuple(sorted(element_factors.items())),
        pair_items=tuple(sorted(pair_factors.items())),
    )


def _element_scales_array(
    symbols: Sequence[str], spec: NormalizedConnectivityFactor
) -> np.ndarray:
    """Per-atom element scales (length ``len(symbols)``)."""
    if spec.global_factor is not None and not spec.element_items:
        return np.full(len(symbols), float(spec.global_factor), dtype=float)
    lookup = spec.element_factors
    default = float(CONNECTIVITY_FACTOR)
    return np.array([float(lookup.get(s, default)) for s in symbols], dtype=float)


def pair_bond_threshold(
    r_i: float,
    r_j: float,
    sym_i: str,
    sym_j: str,
    spec: NormalizedConnectivityFactor,
) -> float:
    """Bond distance threshold for one atom pair under ``spec``."""
    pair: PairKey = (sym_i, sym_j) if sym_i <= sym_j else (sym_j, sym_i)
    if pair in spec.pair_factors:
        return (r_i + r_j) * float(spec.pair_factors[pair])
    if spec.global_factor is not None and not spec.element_items:
        return (r_i + r_j) * float(spec.global_factor)
    f_i = float(spec.element_factors.get(sym_i, CONNECTIVITY_FACTOR))
    f_j = float(spec.element_factors.get(sym_j, CONNECTIVITY_FACTOR))
    return r_i * f_i + r_j * f_j


def _element_threshold_matrix(radii: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """``thresh[i,j] = r_i * s_i + r_j * s_j`` (equiv. ``(r_i+r_j)*f`` when scales equal)."""
    return radii[:, None] * scales[:, None] + radii[None, :] * scales[None, :]


def _apply_pair_overrides_matrix(
    thresh: np.ndarray,
    radii: np.ndarray,
    symbols: Sequence[str],
    pair_factors: Mapping[PairKey, float],
) -> None:
    """In-place: set ``thresh[i,j] = (r_i+r_j)*f_ij`` for matching symbol pairs."""
    if not pair_factors:
        return
    by_sym: dict[str, list[int]] = {}
    for i, sym in enumerate(symbols):
        by_sym.setdefault(sym, []).append(i)
    for (a, b), factor in pair_factors.items():
        f = float(factor)
        idx_a = by_sym.get(a)
        idx_b = by_sym.get(b)
        if not idx_a or not idx_b:
            continue
        if a == b:
            ia = np.asarray(idx_a, dtype=int)
            for i in ia:
                js = ia[ia > i]
                if js.size == 0:
                    continue
                vals = (radii[i] + radii[js]) * f
                thresh[i, js] = vals
                thresh[js, i] = vals
        else:
            ia = np.asarray(idx_a, dtype=int)
            ib = np.asarray(idx_b, dtype=int)
            vals = (radii[ia][:, None] + radii[ib][None, :]) * f
            thresh[np.ix_(ia, ib)] = vals
            thresh[np.ix_(ib, ia)] = vals.T


def bond_threshold_matrix(
    radii: np.ndarray,
    symbols: Sequence[str],
    spec: NormalizedConnectivityFactor,
) -> np.ndarray:
    """Return an ``(n, n)`` matrix of pair bond thresholds.

    Specializes by :meth:`NormalizedConnectivityFactor.kind`:

    - ``global``: one multiply ``(r_i + r_j) * f``
    - ``element``: ``r_i*s_i + r_j*s_j`` from a scales vector
    - ``pair`` / ``mixed``: element (or unit-scale) base, then sparse pair overrides
    """
    n = len(radii)
    if n == 0:
        return np.zeros((0, 0), dtype=float)

    kind = spec.kind()
    if kind == "global":
        thresh = (radii[:, None] + radii[None, :]) * float(spec.global_factor)
        np.fill_diagonal(thresh, 0.0)
        return thresh

    scales = _element_scales_array(symbols, spec)
    thresh = _element_threshold_matrix(radii, scales)
    if kind in ("pair", "mixed"):
        _apply_pair_overrides_matrix(thresh, radii, symbols, spec.pair_factors)
    np.fill_diagonal(thresh, 0.0)
    return thresh


def bond_thresholds_for_pairs(
    radii: np.ndarray,
    symbols: Sequence[str],
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    spec: NormalizedConnectivityFactor,
) -> np.ndarray:
    """Thresholds for explicit ``(i, j)`` index arrays (vectorized by kind)."""
    if i_idx.size == 0:
        return np.empty(0, dtype=float)

    kind = spec.kind()
    if kind == "global":
        return (radii[i_idx] + radii[j_idx]) * float(spec.global_factor)

    scales = _element_scales_array(symbols, spec)
    out = radii[i_idx] * scales[i_idx] + radii[j_idx] * scales[j_idx]
    if kind in ("pair", "mixed") and spec.pair_factors:
        pair_map = spec.pair_factors
        for k, (i, j) in enumerate(zip(i_idx.tolist(), j_idx.tolist(), strict=True)):
            sym_i, sym_j = symbols[i], symbols[j]
            pair: PairKey = (sym_i, sym_j) if sym_i <= sym_j else (sym_j, sym_i)
            if pair in pair_map:
                out[k] = (radii[i] + radii[j]) * float(pair_map[pair])
    return out


def bond_threshold_cross_matrix(
    radii_a: np.ndarray,
    symbols_a: Sequence[str],
    radii_b: np.ndarray,
    symbols_b: Sequence[str],
    spec: NormalizedConnectivityFactor,
) -> np.ndarray:
    """Return an ``(n_a, n_b)`` matrix of bond thresholds between two atom sets.

    Same specialization as :func:`bond_threshold_matrix`, without a zero diagonal
    (the two sets are disjoint).
    """
    n_a, n_b = len(radii_a), len(radii_b)
    if n_a == 0 or n_b == 0:
        return np.zeros((n_a, n_b), dtype=float)

    kind = spec.kind()
    if kind == "global":
        return (radii_a[:, None] + radii_b[None, :]) * float(spec.global_factor)

    scales_a = _element_scales_array(symbols_a, spec)
    scales_b = _element_scales_array(symbols_b, spec)
    thresh = radii_a[:, None] * scales_a[:, None] + radii_b[None, :] * scales_b[None, :]
    if kind in ("pair", "mixed") and spec.pair_factors:
        by_a: dict[str, list[int]] = {}
        by_b: dict[str, list[int]] = {}
        for i, sym in enumerate(symbols_a):
            by_a.setdefault(sym, []).append(i)
        for j, sym in enumerate(symbols_b):
            by_b.setdefault(sym, []).append(j)
        for (a, b), factor in spec.pair_factors.items():
            f = float(factor)
            for sym_left, sym_right in ((a, b), (b, a)) if a != b else ((a, b),):
                ia = by_a.get(sym_left)
                ib = by_b.get(sym_right)
                if not ia or not ib:
                    continue
                ia_arr = np.asarray(ia, dtype=int)
                ib_arr = np.asarray(ib, dtype=int)
                vals = (radii_a[ia_arr][:, None] + radii_b[ib_arr][None, :]) * f
                thresh[np.ix_(ia_arr, ib_arr)] = vals
                if a == b:
                    break
    return thresh


def bond_thresholds_for_cross_pairs(
    radii_a: np.ndarray,
    symbols_a: Sequence[str],
    i_idx: np.ndarray,
    radii_b: np.ndarray,
    symbols_b: Sequence[str],
    j_idx: np.ndarray,
    spec: NormalizedConnectivityFactor,
) -> np.ndarray:
    """Thresholds for explicit cross-set ``(i in A, j in B)`` index arrays."""
    if i_idx.size == 0:
        return np.empty(0, dtype=float)

    kind = spec.kind()
    if kind == "global":
        return (radii_a[i_idx] + radii_b[j_idx]) * float(spec.global_factor)

    scales_a = _element_scales_array(symbols_a, spec)
    scales_b = _element_scales_array(symbols_b, spec)
    out = radii_a[i_idx] * scales_a[i_idx] + radii_b[j_idx] * scales_b[j_idx]
    if kind in ("pair", "mixed") and spec.pair_factors:
        pair_map = spec.pair_factors
        for k, (i, j) in enumerate(zip(i_idx.tolist(), j_idx.tolist(), strict=True)):
            sym_i, sym_j = symbols_a[i], symbols_b[j]
            pair: PairKey = (sym_i, sym_j) if sym_i <= sym_j else (sym_j, sym_i)
            if pair in pair_map:
                out[k] = (radii_a[i] + radii_b[j]) * float(pair_map[pair])
    return out


def clash_threshold_matrix(radii: np.ndarray, min_distance_factor: float) -> np.ndarray:
    """Steric clash threshold matrix: ``(r_i + r_j) * min_distance_factor``."""
    return (radii[:, None] + radii[None, :]) * float(min_distance_factor)


def max_connectivity_scale(spec: NormalizedConnectivityFactor) -> float:
    """Largest multiplier in the spec (KDTree query radius / conservative bounds)."""
    if spec.is_global():
        assert spec.global_factor is not None
        return float(spec.global_factor)
    values: list[float] = []
    if spec.global_factor is not None:
        values.append(float(spec.global_factor))
    values.extend(float(v) for _, v in spec.element_items)
    values.extend(float(v) for _, v in spec.pair_items)
    if spec.element_items and spec.global_factor is None:
        values.append(float(CONNECTIVITY_FACTOR))
    if spec.pair_items and not spec.element_items:
        values.append(float(CONNECTIVITY_FACTOR))
    return max(values) if values else float(CONNECTIVITY_FACTOR)


def min_connectivity_scale(spec: NormalizedConnectivityFactor) -> float:
    """Smallest multiplier in the spec (steric-floor comparisons)."""
    if spec.is_global():
        assert spec.global_factor is not None
        return float(spec.global_factor)
    values: list[float] = []
    if spec.global_factor is not None:
        values.append(float(spec.global_factor))
    values.extend(float(v) for _, v in spec.element_items)
    values.extend(float(v) for _, v in spec.pair_items)
    if spec.element_items and spec.global_factor is None:
        values.append(float(CONNECTIVITY_FACTOR))
    if spec.pair_items and not spec.element_items:
        values.append(float(CONNECTIVITY_FACTOR))
    return min(values) if values else float(CONNECTIVITY_FACTOR)


def format_connectivity_factor(spec: NormalizedConnectivityFactor) -> str:
    """Human-readable form for logs and error messages."""
    if spec.is_global():
        return f"{spec.global_factor:.2f}"
    parts: list[str] = []
    for sym, val in spec.element_items:
        parts.append(f"{sym}:{val:.2f}")
    for (a, b), val in spec.pair_items:
        parts.append(f"{a}-{b}:{val:.2f}")
    return "{" + ", ".join(parts) + "}"


def connectivity_factor_cache_key(spec: NormalizedConnectivityFactor) -> tuple:
    """Hashable cache key for template / placement caches."""
    if spec.is_global():
        assert spec.global_factor is not None
        return ("global", float(spec.global_factor))
    return ("dict", spec.element_items, spec.pair_items)


def connectivity_factor_for_json(
    spec: ConnectivityFactorInput | NormalizedConnectivityFactor | None,
) -> float | dict[str, float]:
    """JSON-serializable form (pair keys as ``'A-B'``)."""
    norm = normalize_connectivity_factor(spec)
    if norm.is_global():
        assert norm.global_factor is not None
        return float(norm.global_factor)
    out: dict[str, float] = {sym: float(v) for sym, v in norm.element_items}
    for (a, b), v in norm.pair_items:
        out[f"{a}-{b}"] = float(v)
    return out
