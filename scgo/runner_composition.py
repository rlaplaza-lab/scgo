"""Composition parsing and composition-list builders for SCGO runners.

Pure helpers that turn a ``CompositionInput`` (compact formula string,
comma-separated symbol string, ``list[str]``, or an ASE ``Atoms``) into the
``list[str]`` composition representation used throughout :mod:`scgo.runner_api`
and friends, plus builders for common composition-list scans.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from ase import Atoms
from ase.data import atomic_numbers
from ase.formula import Formula

from scgo.exceptions import SCGOValidationError
from scgo.utils.helpers import get_composition_counts

type CompositionInput = str | list[str] | Atoms


def _compact_formula_error(comp_str: str, detail: str) -> SCGOValidationError:
    return SCGOValidationError(
        f"Unable to parse composition string: {comp_str}. {detail} "
        "Use chemical capitalization for compact formulas (e.g. Pt3Au, HO2Ru9W2) "
        "or comma-separated symbols (e.g. Pt,Pt,Pt,Au)."
    )


def _parse_lowercase_single_element(comp_str: str) -> list[str] | None:
    """Parse ``pt3``-style single-element formulas, or ``None`` if not applicable."""
    m = re.fullmatch(r"([a-z]{1,2})(\d+)?", comp_str.strip())
    if not m:
        return None

    raw, count_str = m.group(1), m.group(2)
    count = int(count_str) if count_str else 1
    if count == 0:
        raise _compact_formula_error(comp_str, "Zero atom count is not allowed.")

    sym = raw[0].upper() + raw[1:]
    if sym not in atomic_numbers:
        return None

    if len(raw) == 2:
        one_char = raw[0].upper()
        if one_char in atomic_numbers:
            tail = raw[1:] + (count_str or "")
            alt_str = one_char + (tail[0].upper() + tail[1:] if tail else "")
            try:
                alt = list(Formula(alt_str, strict=True))
            except ValueError:
                pass
            else:
                if get_composition_counts(alt) != get_composition_counts([sym] * count):
                    raise _compact_formula_error(
                        comp_str,
                        f"Ambiguous with {alt_str!r} under chemical capitalization rules.",
                    )

    return [sym] * count


def parse_composition_arg(comp_str: str) -> list[str]:
    """Parse composition strings.

    Supported formats:

    - Comma-separated symbols such as ``"Pt,Pt,Au"`` (case-insensitive per
      symbol).
    - Compact formula with chemical capitalization such as ``"Pt3Au"`` or
      ``"HO2Ru9W2"``. Uses :class:`ase.formula.Formula`; ``HO2`` is H + O2,
      not holmium.

    All-lowercase compact formulas are accepted only for unambiguous
    single-element inputs such as ``pt3``. Multi-element lowercase strings
    (``pt3au``) and ambiguous two-letter forms (``ho2`` as H+O2 vs holmium)
    are rejected.
    """
    comp_str = comp_str.strip()
    if not comp_str:
        raise _compact_formula_error(comp_str, "Empty composition string.")

    if "," in comp_str:
        parts = [p.strip() for p in comp_str.split(",") if p.strip()]
        normalized = [p[0].upper() + p[1:].lower() if len(p) > 0 else p for p in parts]
        return normalized

    if re.search(r"([A-Za-z]{1,2})0(?![0-9])", comp_str):
        raise _compact_formula_error(comp_str, "Zero atom count is not allowed.")

    if comp_str == comp_str.lower():
        single = _parse_lowercase_single_element(comp_str)
        if single is not None:
            return single
        m = re.fullmatch(r"([a-z]{1,2})(\d+)?", comp_str)
        if m:
            sym = m.group(1)[0].upper() + m.group(1)[1:]
            if sym not in atomic_numbers:
                raise _compact_formula_error(
                    comp_str, f"Unknown element symbol {sym!r}."
                )
        raise _compact_formula_error(
            comp_str,
            "Lowercase compact formulas with multiple elements are not supported.",
        )

    try:
        composition = [str(symbol) for symbol in Formula(comp_str, strict=True)]
    except ValueError as exc:
        raise _compact_formula_error(comp_str, "Invalid compact formula.") from exc
    if not composition:
        raise _compact_formula_error(comp_str, "Zero atom count is not allowed.")
    return composition


def _as_composition(composition: CompositionInput) -> list[str]:
    if isinstance(composition, Atoms):
        return list(composition.get_chemical_symbols())
    elif isinstance(composition, str):
        return parse_composition_arg(composition)
    elif isinstance(composition, list):
        if not composition:
            raise SCGOValidationError("composition list must not be empty")
        return [str(s) for s in composition]
    else:
        raise SCGOValidationError(
            f"composition must be str, list[str], or Atoms, got {type(composition).__name__}"
        )


def _as_composition_list(items: Iterable[CompositionInput]) -> list[list[str]]:
    out = [_as_composition(x) for x in items]
    if not out:
        raise SCGOValidationError("compositions iterable must not be empty")
    return out


def build_one_element_compositions(
    element: str, min_atoms: int, max_atoms: int
) -> list[list[str]]:
    """Composition list for mono-element size scans (min_atoms..max_atoms)."""
    if not element or not isinstance(element, str):
        raise SCGOValidationError("element must be a non-empty string")
    if min_atoms < 1:
        raise SCGOValidationError("min_atoms must be >= 1")
    if max_atoms < min_atoms:
        raise SCGOValidationError("max_atoms must be >= min_atoms")
    return [[element] * n_atoms for n_atoms in range(min_atoms, max_atoms + 1)]


def build_two_element_compositions(
    element1: str, element2: str, min_atoms: int, max_atoms: int
) -> list[list[str]]:
    """Composition list for bimetallic size scans (min_atoms..max_atoms)."""
    if not element1 or not isinstance(element1, str):
        raise SCGOValidationError("element1 must be a non-empty string")
    if not element2 or not isinstance(element2, str):
        raise SCGOValidationError("element2 must be a non-empty string")
    if min_atoms < 1:
        raise SCGOValidationError("min_atoms must be >= 1")
    if max_atoms < min_atoms:
        raise SCGOValidationError("max_atoms must be >= min_atoms")
    compositions: list[list[str]] = []
    for n_atoms in range(min_atoms, max_atoms + 1):
        for i in range(n_atoms + 1):
            compositions.append([element1] * i + [element2] * (n_atoms - i))
    return compositions
