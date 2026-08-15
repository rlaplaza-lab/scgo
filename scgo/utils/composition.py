"""Composition counting helpers (leaf module; safe for system_types imports)."""

from __future__ import annotations

from collections import Counter


def get_composition_counts(composition: list[str]) -> Counter[str]:
    """Return element counts for a composition."""
    return Counter(composition)
