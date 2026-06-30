"""Small shared helpers for cluster+adsorbate code (no system_types imports)."""

from __future__ import annotations

from collections.abc import Mapping


def resolve_fragment_anchor_and_bond_axis(
    adsorbate_definition: Mapping[str, object],
) -> tuple[int, tuple[int, int] | None]:
    """Return fragment anchor index and optional bond-axis pair from adsorbate metadata."""
    anchor = int(adsorbate_definition.get("fragment_anchor_index", 0))
    fba = adsorbate_definition.get("fragment_bond_axis")
    bond_axis: tuple[int, int] | None = None
    if fba is not None:
        bond_axis = (int(fba[0]), int(fba[1]))
    return anchor, bond_axis


def parse_positive_fragment_lengths(raw: object) -> list[int]:
    """Return positive fragment lengths from an adsorbate_definition field, or ``[]``."""
    if not isinstance(raw, list):
        return []
    return [int(x) for x in raw if isinstance(x, int) and int(x) > 0]
