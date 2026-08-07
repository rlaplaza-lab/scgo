"""Vectorized steric deficit scoring matches the reference implementation."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import cdist, squareform

from scgo.initialization.steric_scoring import (
    build_blmin_threshold_matrix,
    get_blmin_distance,
    steric_deficit,
    steric_deficit_two_sets,
)


def _reference_steric_deficit(positions, atomic_numbers, blmin) -> float:
    """Original nested-loop implementation, kept as the numerical reference."""
    n_atoms = len(positions)
    if n_atoms <= 1:
        return 0.0
    from scipy.spatial.distance import pdist

    distances = squareform(pdist(positions))
    numbers = np.asarray(atomic_numbers, dtype=int)
    deficit = 0.0
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            gap = get_blmin_distance(blmin, numbers[i], numbers[j]) - distances[i, j]
            if gap > 0.0:
                deficit += gap
    return deficit


def _reference_steric_deficit_two_sets(
    left_positions, left_numbers, right_positions, right_numbers, blmin
) -> float:
    """Original implementation for the two-set variant."""
    if len(left_positions) == 0 or len(right_positions) == 0:
        return 0.0
    distances = cdist(left_positions, right_positions)
    required = np.array(
        [
            [get_blmin_distance(blmin, int(zi), int(zj)) for zj in right_numbers]
            for zi in left_numbers
        ],
        dtype=float,
    )
    return float(np.maximum(required - distances, 0.0).sum())


def _blmin_table() -> dict:
    return {
        (78, 78): 2.2,
        (78, 8): 1.6,
        (8, 78): 1.6,
        (8, 8): 1.2,
    }


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_steric_deficit_matches_reference(seed):
    rng = np.random.default_rng(seed)
    n_atoms = 40
    positions = rng.uniform(-4.0, 4.0, size=(n_atoms, 3))
    numbers = rng.choice([78, 8], size=n_atoms)
    blmin = _blmin_table()

    new = steric_deficit(positions, numbers, blmin)
    ref = _reference_steric_deficit(positions, numbers, blmin)
    assert new == pytest.approx(ref, abs=1e-9)
    # The random geometry must actually contain violations to be meaningful.
    assert ref > 0.0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_steric_deficit_two_sets_matches_reference(seed):
    rng = np.random.default_rng(seed)
    left = rng.uniform(-3.0, 3.0, size=(15, 3))
    right = rng.uniform(-3.0, 3.0, size=(25, 3))
    left_z = rng.choice([78, 8], size=15)
    right_z = rng.choice([78, 8], size=25)
    blmin = _blmin_table()

    new = steric_deficit_two_sets(left, left_z, right, right_z, blmin)
    ref = _reference_steric_deficit_two_sets(left, left_z, right, right_z, blmin)
    assert new == pytest.approx(ref, abs=1e-9)
    assert ref > 0.0


def test_steric_deficit_zero_for_well_separated_atoms():
    positions = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    assert steric_deficit(positions, [78, 78, 78], _blmin_table()) == 0.0


def test_steric_deficit_edge_cases():
    blmin = _blmin_table()
    assert steric_deficit(np.zeros((0, 3)), [], blmin) == 0.0
    assert steric_deficit(np.zeros((1, 3)), [78], blmin) == 0.0
    assert (
        steric_deficit_two_sets(np.zeros((0, 3)), [], np.zeros((2, 3)), [78, 78], blmin)
        == 0.0
    )


def test_steric_deficit_raises_for_missing_blmin_pair():
    """Missing element pairs still raise, matching ``get_blmin_distance``."""
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    with pytest.raises(KeyError):
        steric_deficit(positions, [78, 47], {(78, 78): 2.2})


def test_build_blmin_threshold_matrix_factor_and_default():
    thresh, index = build_blmin_threshold_matrix(
        [78, 8, 78], _blmin_table(), factor=0.5
    )
    # Unique Zs are sorted ascending: [8, 78]
    assert list(index) == [1, 0, 1]
    assert thresh[1, 1] == pytest.approx(1.1)
    assert thresh[0, 1] == pytest.approx(0.8)

    missing, _ = build_blmin_threshold_matrix(
        [78, 47], {(78, 78): 2.2, (47, 47): 2.0}, default=0.0
    )
    assert missing[0, 1] == 0.0
