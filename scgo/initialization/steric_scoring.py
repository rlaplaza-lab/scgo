"""Steric deficit scoring and blmin distance lookups for placement and GA."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist, pdist


def get_blmin_distance(
    blmin: dict, atomic_number_a: int, atomic_number_b: int
) -> float:
    """Minimum allowed distance for an element pair from an ASE-style blmin table."""
    key = (int(atomic_number_a), int(atomic_number_b))
    if key in blmin:
        return blmin[key]
    return blmin[(int(atomic_number_b), int(atomic_number_a))]


def build_blmin_threshold_matrix(
    atomic_numbers,
    blmin: dict,
    *,
    factor: float = 1.0,
    default: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Map atomic numbers onto a dense Z-pair minimum-distance matrix.

    Building the ``(n_unique, n_unique)`` table once turns the per-pair ``blmin``
    dict lookups into a single vectorized gather, which is what makes the
    ``cdist``-based deficit scoring fast for large clusters and slabs.

    Args:
        atomic_numbers: Atomic numbers of the atoms to be scored.
        blmin: ASE-style ``{(Z_i, Z_j): min_distance}`` table.
        factor: Multiplier applied to every threshold (e.g. a permissive
            prefilter factor). Defaults to 1.0.
        default: Value used for element pairs missing from ``blmin``. ``None``
            (the default) reproduces :func:`get_blmin_distance` and raises
            ``KeyError`` for missing pairs.

    Returns:
        Tuple ``(thresholds, index)`` where ``thresholds[i, j]`` is the minimum
        allowed distance between unique element ``i`` and ``j``, and ``index``
        maps each input atom to its row/column in ``thresholds``.
    """
    numbers = np.asarray(atomic_numbers, dtype=int)
    unique_z = np.unique(numbers)
    z_to_i = {int(z): i for i, z in enumerate(unique_z)}

    n_u = len(unique_z)
    thresh = np.zeros((n_u, n_u), dtype=float)
    for zi in unique_z:
        zi_i = int(zi)
        for zj in unique_z:
            zj_i = int(zj)
            if default is None:
                min_allowed = float(get_blmin_distance(blmin, zi_i, zj_i))
            else:
                min_allowed = float(
                    blmin.get((zi_i, zj_i), blmin.get((zj_i, zi_i), default))
                )
            thresh[z_to_i[zi_i], z_to_i[zj_i]] = factor * min_allowed

    index = np.array([z_to_i[int(z)] for z in numbers], dtype=int)
    return thresh, index


def steric_deficit(positions, atomic_numbers, blmin: dict) -> float:
    """Sum of blmin violations within a single structure (lower is better)."""
    n_atoms = len(positions)
    if n_atoms <= 1:
        return 0.0

    thresh, index = build_blmin_threshold_matrix(atomic_numbers, blmin)
    upper = np.triu_indices(n_atoms, k=1)
    # pdist enumerates pairs in the same row-major upper-triangle order.
    required = thresh[np.ix_(index, index)][upper]
    return float(np.maximum(required - pdist(positions), 0.0).sum())


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

    left_z = np.asarray(left_numbers, dtype=int)
    right_z = np.asarray(right_numbers, dtype=int)
    thresh, index = build_blmin_threshold_matrix(
        np.concatenate([left_z, right_z]), blmin
    )
    left_index = index[: len(left_z)]
    right_index = index[len(left_z) :]

    required = thresh[np.ix_(left_index, right_index)]
    distances = cdist(left_positions, right_positions)
    return float(np.maximum(required - distances, 0.0).sum())
