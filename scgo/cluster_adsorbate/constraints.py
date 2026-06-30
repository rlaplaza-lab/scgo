"""ASE constraints for adsorbates on a cluster."""

from __future__ import annotations

from collections.abc import Sequence

from ase import Atoms
from ase.constraints import FixBondLength


def attach_fix_bond_lengths(
    atoms: Atoms,
    bond_pairs: Sequence[tuple[int, int]],
) -> None:
    """Append one :class:`~ase.constraints.FixBondLength` per pair (global indices).

    Args:
        atoms: Full system (core + adsorbate).
        bond_pairs: Pairs of atom indices in ``atoms`` whose distances to hold fixed.

    Raises:
        ValueError: Invalid indices or duplicate pair.
    """
    n = len(atoms)
    seen: set[tuple[int, int]] = set()
    new_constraints: list = list(atoms.constraints) if atoms.constraints else []
    for a, b in bond_pairs:
        if not (0 <= a < n and 0 <= b < n):
            raise ValueError(
                f"Invalid bond pair ({a}, {b}) for structure with {n} atoms"
            )
        if a == b:
            raise ValueError(f"bond pair ({a}, {b}) must be two distinct atoms")
        key = (min(a, b), max(a, b))
        if key in seen:
            raise ValueError(f"duplicate bond pair {key}")
        seen.add(key)
        new_constraints.append(FixBondLength(int(a), int(b)))
    if new_constraints:
        atoms.set_constraint(new_constraints)


def attach_adsorbate_internal_geometry_constraints(
    atoms: Atoms,
    *,
    n_slab: int,
    adsorbate_definition: dict | None,
) -> None:
    """Freeze pairwise distances inside each adsorbate fragment.

    This enforces rigid internal geometry for adsorbates while still allowing
    collective translation/rotation of each adsorbate fragment.
    """
    if adsorbate_definition is None:
        return
    core_symbols = adsorbate_definition.get("core_symbols", [])
    if not isinstance(core_symbols, list):
        return
    raw_lengths = adsorbate_definition.get("adsorbate_fragment_lengths", [])
    if not isinstance(raw_lengths, list) or not all(
        isinstance(x, int) for x in raw_lengths
    ):
        return
    fragment_lengths = [int(x) for x in raw_lengths if int(x) > 0]
    if not fragment_lengths:
        return

    ads_start = int(n_slab) + len(core_symbols)
    bond_pairs: list[tuple[int, int]] = []
    cursor = ads_start
    for frag_len in fragment_lengths:
        for i in range(cursor, cursor + frag_len):
            for j in range(i + 1, cursor + frag_len):
                bond_pairs.append((i, j))
        cursor += frag_len

    if bond_pairs:
        attach_fix_bond_lengths(atoms, bond_pairs)
