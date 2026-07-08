"""Small shared helpers for cluster+adsorbate code (no system_types imports)."""

from __future__ import annotations

from collections.abc import Mapping

from scgo.exceptions import SCGOValidationError


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
