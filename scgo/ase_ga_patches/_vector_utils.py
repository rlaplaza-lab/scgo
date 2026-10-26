"""Vector utilities for ASE GA mutation operators (vendored helpers)."""

from __future__ import annotations

import numpy as np

from scgo.exceptions import SCGOValidationError


def random_unit_vector(rng):
    vector = rng.normal(0.0, 1.0, 3)
    norm = np.linalg.norm(vector)
    if norm <= 1e-12:
        return np.array([1.0, 0.0, 0.0])
    return vector / norm


def kabsch_rotation(reference: np.ndarray, moving: np.ndarray) -> np.ndarray:
    """Rotation R such that ``moving @ R.T`` aligns with centered ``reference``."""
    dim = int(reference.shape[1])
    left, _, right = np.linalg.svd(reference.T @ moving)
    signs = np.ones(dim, dtype=float)
    signs[-1] = float(np.sign(np.linalg.det(left @ right)) or 1.0)
    return left @ np.diag(signs) @ right


def kabsch_rotation_in_plane(
    reference: np.ndarray,
    moving: np.ndarray,
    surface_normal_axis: int = 2,
) -> np.ndarray:
    """3x3 rotation that aligns ``moving`` to ``reference`` in the surface plane."""
    if surface_normal_axis not in (0, 1, 2):
        raise SCGOValidationError("surface_normal_axis must be 0, 1, or 2")
    plane_axes = [i for i in range(3) if i != surface_normal_axis]
    planar = kabsch_rotation(reference[:, plane_axes], moving[:, plane_axes])
    rotation = np.eye(3)
    for row, row_axis in enumerate(plane_axes):
        for col, col_axis in enumerate(plane_axes):
            rotation[row_axis, col_axis] = planar[row, col]
    return rotation


def append_unique_unit_vector(candidates, vector, tol=0.995):
    unit = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(unit)
    if norm <= 1e-12:
        return
    unit /= norm
    for existing in candidates:
        if float(np.dot(unit, existing)) > tol:
            return
    candidates.append(unit)
