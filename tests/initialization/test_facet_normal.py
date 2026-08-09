"""Regression tests for the planar facet normal (bug 1.3.3).

``get_largest_facets`` falls back to a covariance analysis when the convex hull
is degenerate (linear/planar clusters). ``np.linalg.eigh`` returns eigenvalues in
ASCENDING order, so the plane normal is the *first* eigenvector (smallest
eigenvalue / least positional variance). The previous code used
``eigenvectors[:, -1]``, i.e. the direction of largest variance, which lies
inside the plane instead of perpendicular to it.
"""

import numpy as np
import pytest
from ase import Atoms

from scgo.initialization.geometry_helpers import (
    _classify_seed_geometry,
    get_largest_facets,
)

# Planar Pt5 cluster lying exactly in the z = 0 plane.
PLANAR_POSITIONS = np.array(
    [
        [0.0, 0.0, 0.0],
        [2.7, 0.0, 0.0],
        [1.35, 2.34, 0.0],
        [4.05, 2.34, 0.0],
        [1.35, -2.34, 0.0],
    ]
)


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Return the rotation matrix for ``angle`` radians about ``axis``."""
    axis = axis / np.linalg.norm(axis)
    kx, ky, kz = axis
    K = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _covariance_eigen(atoms: Atoms) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(eigenvalues, eigenvectors)`` of the centered position covariance."""
    positions = atoms.get_positions()
    centered = positions - atoms.get_center_of_mass()
    return np.linalg.eigh(np.cov(centered.T))


class TestPlanarFacetNormal:
    """The degenerate-hull fallback must return the out-of-plane direction."""

    def test_planar_cluster_normal_is_out_of_plane(self):
        """A cluster in the z = 0 plane gets a normal along z."""
        atoms = Atoms("Pt5", positions=PLANAR_POSITIONS)
        assert _classify_seed_geometry(atoms) == "planar"

        facets = get_largest_facets(atoms)
        assert len(facets) == 1
        _center, normal, _area = facets[0]

        assert np.linalg.norm(normal) == pytest.approx(1.0)
        assert abs(float(np.dot(normal, [0.0, 0.0, 1.0]))) == pytest.approx(
            1.0, abs=1e-8
        )

        # Perpendicular to every in-plane displacement.
        positions = atoms.get_positions()
        for other in positions[1:]:
            in_plane = other - positions[0]
            assert float(np.dot(normal, in_plane)) == pytest.approx(0.0, abs=1e-8)

    def test_normal_is_smallest_eigenvalue_eigenvector(self):
        """The normal is the eigenvector of the smallest covariance eigenvalue."""
        atoms = Atoms("Pt5", positions=PLANAR_POSITIONS)
        eigenvalues, eigenvectors = _covariance_eigen(atoms)

        # np.linalg.eigh sorts eigenvalues ascending.
        assert eigenvalues[0] <= eigenvalues[1] <= eigenvalues[2]

        expected = eigenvectors[:, 0] / np.linalg.norm(eigenvectors[:, 0])
        largest = eigenvectors[:, -1] / np.linalg.norm(eigenvectors[:, -1])

        _center, normal, _area = get_largest_facets(atoms)[0]

        assert abs(float(np.dot(normal, expected))) == pytest.approx(1.0, abs=1e-8)
        # Regression guard: the largest-variance eigenvector lies in the plane.
        assert abs(float(np.dot(normal, largest))) == pytest.approx(0.0, abs=1e-8)

    def test_tilted_planar_cluster_normal_follows_plane(self):
        """Rotating the planar cluster rotates the computed normal with it."""
        rotation = _rotation_matrix(np.array([1.0, 1.0, 0.0]), np.pi / 5.0)
        atoms = Atoms("Pt5", positions=PLANAR_POSITIONS @ rotation.T)
        assert _classify_seed_geometry(atoms) == "planar"

        expected_normal = rotation @ np.array([0.0, 0.0, 1.0])

        _center, normal, _area = get_largest_facets(atoms)[0]
        assert abs(float(np.dot(normal, expected_normal))) == pytest.approx(
            1.0, abs=1e-8
        )
