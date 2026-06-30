"""Steric deficit scoring and blmin distance lookups for placement and GA."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform


def get_blmin_distance(
    blmin: dict, atomic_number_a: int, atomic_number_b: int
) -> float:
    """Minimum allowed distance for an element pair from an ASE-style blmin table."""
    key = (int(atomic_number_a), int(atomic_number_b))
    if key in blmin:
        return blmin[key]
    return blmin[(int(atomic_number_b), int(atomic_number_a))]


def steric_deficit(positions, atomic_numbers, blmin: dict) -> float:
    """Sum of blmin violations within a single structure (lower is better)."""
    n_atoms = len(positions)
    if n_atoms <= 1:
        return 0.0

    distances = squareform(pdist(positions))
    deficit = 0.0
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            required = get_blmin_distance(blmin, atomic_numbers[i], atomic_numbers[j])
            gap = required - distances[i, j]
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
    deficit = 0.0
    for i, left_pos in enumerate(left_positions):
        for j, right_pos in enumerate(right_positions):
            required = get_blmin_distance(blmin, left_numbers[i], right_numbers[j])
            gap = required - np.linalg.norm(left_pos - right_pos)
            if gap > 0.0:
                deficit += gap
    return deficit
