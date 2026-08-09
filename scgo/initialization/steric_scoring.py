"""Steric deficit scoring and blmin distance lookups for placement and GA."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform


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
    vget = np.vectorize(
        lambda zi, zj: get_blmin_distance(blmin, int(zi), int(zj)),
        otypes=[float],
    )
    return vget(a[:, None], b[None, :])


def steric_deficit(positions, atomic_numbers, blmin: dict) -> float:
    """Sum of blmin violations within a single structure (lower is better)."""
    n_atoms = len(positions)
    if n_atoms <= 1:
        return 0.0

    distances = squareform(pdist(positions))
    numbers = np.asarray(atomic_numbers, dtype=int)
    deficit = 0.0
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            gap = get_blmin_distance(blmin, numbers[i], numbers[j]) - distances[i, j]
            if gap > 0.0:
                deficit += gap
    return deficit


def steric_deficit_two_sets(
    left_positions,
    left_numbers,
    right_positions,
    right_numbers,
    blmin: dict,
) -> float:
    """Sum of blmin violations between two disjoint atom sets."""
    if len(left_positions) == 0 or len(right_positions) == 0:
        return 0.0

    distances = cdist(left_positions, right_positions)
    required = _blmin_matrix(left_numbers, blmin, right_numbers)
    return float(np.maximum(required - distances, 0.0).sum())
