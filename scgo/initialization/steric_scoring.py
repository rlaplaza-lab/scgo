"""Steric deficit scoring and blmin distance lookups for placement and GA."""

from __future__ import annotations

import numpy as np
from ase.geometry import get_distances
from scipy.spatial.distance import cdist, pdist


def get_blmin_distance(
    blmin: dict, atomic_number_a: int, atomic_number_b: int
) -> float:
    """Minimum allowed distance for an element pair from an ASE-style blmin table."""
    key = (int(atomic_number_a), int(atomic_number_b))
    if key in blmin:
        return blmin[key]
    return blmin[(int(atomic_number_b), int(atomic_number_a))]


def _blmin_matrix(atomic_numbers, blmin: dict, other=None) -> np.ndarray:
    """Dense ``(n_a × n_b)`` blmin threshold matrix for element-number arrays.

    ``atomic_numbers`` gives the row elements; ``other`` (defaults to the same
    array) gives the column elements. Each entry is the minimum allowed distance
    between that element pair from ``blmin``. The matrix is symmetric when
    ``other`` is ``None``.
    """
    a = np.asarray(atomic_numbers, dtype=int)
    b = a if other is None else np.asarray(other, dtype=int)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.size, b.size), dtype=float)

    unique = tuple(sorted({int(z) for z in a} | {int(z) for z in b}))
    z_to_i = {z: i for i, z in enumerate(unique)}
    table = np.empty((len(unique), len(unique)), dtype=float)
    for i, zi in enumerate(unique):
        for j, zj in enumerate(unique):
            table[i, j] = float(get_blmin_distance(blmin, zi, zj))
    row_index = np.array([z_to_i[int(z)] for z in a], dtype=int)
    col_index = np.array([z_to_i[int(z)] for z in b], dtype=int)
    return table[row_index[:, None], col_index[None, :]]


def steric_deficit(positions, atomic_numbers, blmin: dict) -> float:
    """Sum of blmin violations within a single structure (lower is better)."""
    n_atoms = len(positions)
    if n_atoms <= 1:
        return 0.0

    distances = pdist(positions)
    numbers = np.asarray(atomic_numbers, dtype=int)
    required = _blmin_matrix(numbers, blmin)
    iu, ju = np.triu_indices(n_atoms, k=1)
    required_u = required[iu, ju]
    return float(np.maximum(required_u - distances, 0.0).sum())


def steric_deficit_two_sets(
    left_positions,
    left_numbers,
    right_positions,
    right_numbers,
    blmin: dict,
    *,
    cell=None,
    pbc=None,
) -> tuple[float, bool]:
    """Return ``(deficit, hard_clash)`` between two disjoint atom sets.

    Distances use the minimum-image convention when any ``pbc`` flag is set.
    """
    if len(left_positions) == 0 or len(right_positions) == 0:
        return 0.0, False

    left = np.asarray(left_positions, dtype=float)
    right = np.asarray(right_positions, dtype=float)
    if cell is not None and pbc is not None and bool(np.any(pbc)):
        _, distances = get_distances(left, right, cell=cell, pbc=pbc)
    else:
        distances = cdist(left, right)
    required = _blmin_matrix(left_numbers, blmin, right_numbers)
    deficit = float(np.maximum(required - distances, 0.0).sum())
    is_hard_clash = bool(np.any(distances < required))
    return deficit, is_hard_clash
