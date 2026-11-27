"""Shared distance, rotation, and degenerate-axis helpers."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase_ga.utilities import get_rotation_matrix

from scgo.ase_ga_patches._vector_utils import (
    random_unit_vector as ga_random_unit_vector,
)
from scgo.initialization.geometry_helpers import _unit_vector
from scgo.utils.geometry_math import (
    DEGENERATE_AXIS,
    ROTATION_AXIS_TOLERANCE,
    normalize_or,
    pairwise_distances,
    rotation_matrix_axis_angle,
)
from scgo.utils.geometry_math import random_unit_vector as cluster_random_unit_vector


class _Zeros:
    def normal(self, *_args):
        return np.zeros(3)

    def standard_normal(self, _n):
        return np.zeros(3)


class _Fixed:
    def __init__(self, value):
        self._value = np.asarray(value, dtype=float)

    def normal(self, *_args):
        return self._value.copy()

    def standard_normal(self, _n):
        return self._value.copy()


def test_normalize_or_degenerate_axis_and_unit_vector_band():
    assert np.allclose(normalize_or(np.zeros(3)), DEGENERATE_AXIS)
    # Between the default 1e-12 floor and the initialization 1e-10 floor.
    small = np.array([5e-11, 0.0, 0.0])
    assert np.allclose(normalize_or(small), [1.0, 0.0, 0.0])
    assert np.allclose(_unit_vector(small), DEGENERATE_AXIS)
    assert pytest.approx(1e-10) == ROTATION_AXIS_TOLERANCE


def test_random_unit_vectors_share_degenerate_axis():
    zeros = _Zeros()
    assert np.allclose(ga_random_unit_vector(zeros), DEGENERATE_AXIS)
    assert np.allclose(cluster_random_unit_vector(zeros), DEGENERATE_AXIS)
    direction = np.array([3.0, 0.0, 0.0])
    assert np.allclose(ga_random_unit_vector(_Fixed(direction)), [1.0, 0.0, 0.0])
    assert np.allclose(cluster_random_unit_vector(_Fixed(direction)), [1.0, 0.0, 0.0])


def test_rotation_matches_ase_and_identity_below_tolerance():
    axis = np.array([0.0, 0.0, 2.0])
    angle = 0.4
    got = rotation_matrix_axis_angle(axis, angle)
    expected = get_rotation_matrix(axis / np.linalg.norm(axis), angle)
    assert np.allclose(got, expected)
    assert np.allclose(
        rotation_matrix_axis_angle(np.array([1e-11, 0.0, 0.0]), angle), np.eye(3)
    )


def test_pairwise_distances_mic_policy():
    left = np.array([[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
    cell = np.eye(3) * 10.0
    pbc = np.array([True, True, True])
    mic = pairwise_distances(left, cell=cell, pbc=pbc, use_mic=True)
    atoms = Atoms("H2", positions=left, cell=cell, pbc=pbc)
    assert np.allclose(mic, atoms.get_all_distances(mic=True))

    no_pbc = np.array([False, False, False])
    cartesian = pairwise_distances(
        left, cell=cell, pbc=no_pbc, use_mic=True, require_any_pbc=True
    )
    assert np.allclose(cartesian[0, 1], 9.0)
    # Without the prefilter guard, use_mic still calls ASE even if pbc is false.
    forced = pairwise_distances(left, cell=cell, pbc=no_pbc, use_mic=True)
    assert forced.shape == (2, 2)
