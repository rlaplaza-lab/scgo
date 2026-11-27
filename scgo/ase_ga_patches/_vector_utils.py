"""Vector utilities for ASE GA mutation operators (vendored helpers)."""

from __future__ import annotations

import numpy as np

from scgo.utils.geometry_math import kabsch_rotation as kabsch_rotation
from scgo.utils.geometry_math import random_unit_vector as random_unit_vector
from scgo.utils.validation import validate_surface_normal_axis


def kabsch_rotation_in_plane(
    reference: np.ndarray,
    moving: np.ndarray,
    surface_normal_axis: int = 2,
) -> np.ndarray:
    """3x3 rotation that aligns ``moving`` to ``reference`` in the surface plane."""
    validate_surface_normal_axis(surface_normal_axis)
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
