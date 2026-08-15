"""Small shared helpers for cluster+adsorbate code (no system_types imports)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from scgo.exceptions import SCGOValidationError

if TYPE_CHECKING:
    from scgo.system_types import AdsorbateDefinition


def resolve_fragment_anchor_and_bond_axis(
    adsorbate_definition: AdsorbateDefinition,
) -> tuple[int, tuple[int, int] | None]:
    """Return fragment anchor index and optional bond-axis pair from adsorbate metadata."""
    anchor = adsorbate_definition.fragment_anchor_index
    anchor = int(anchor) if anchor is not None else 0
    bond_axis = adsorbate_definition.fragment_bond_axis
    return anchor, (tuple(bond_axis) if bond_axis is not None else None)


def parse_positive_fragment_lengths(raw: object) -> list[int]:
    """Return positive fragment lengths from an adsorbate_definition field.

    Raises:
        ValueError: If *raw* is not a list or contains no positive integer lengths.
    """
    if not isinstance(raw, list):
        raise SCGOValidationError(
            f"adsorbate_fragment_lengths must be a list of positive ints, got {type(raw).__name__}"
        )
    lengths = [int(x) for x in raw if isinstance(x, int) and int(x) > 0]
    if not lengths:
        raise SCGOValidationError(
            "adsorbate_fragment_lengths must contain positive integers"
        )
    return lengths
