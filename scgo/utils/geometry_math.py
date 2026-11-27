"""Shared distance, rotation, and unit-vector helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from ase.geometry import find_mic, get_distances
from scipy.spatial.distance import cdist

ROTATION_AXIS_TOLERANCE: float = 1e-10

DEGENERATE_AXIS: np.ndarray = np.array([0.0, 0.0, 1.0])


def normalize_or(v: np.ndarray, *, tol: float = 1e-12) -> np.ndarray:
    """Return ``v / ||v||``, or ``[0, 0, 1]`` when ``||v|| < tol``."""
    arr = np.asarray(v, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm < tol:
        return DEGENERATE_AXIS.copy()
    return arr / norm


def random_unit_vector(rng) -> np.ndarray:
    """Unit vector from ``rng.standard_normal``. A near-zero draw becomes ``[0, 0, 1]``."""
    return normalize_or(rng.standard_normal(3))


def pairwise_distances(
    left: np.ndarray,
    right: np.ndarray | None = None,
    *,
    cell: Any = None,
    pbc: Any = None,
    use_mic: bool = False,
    require_any_pbc: bool = False,
) -> np.ndarray:
    """Pairwise distances. MIC requires ``use_mic`` and, if requested, some PBC."""
    apply_mic = bool(use_mic) and cell is not None and pbc is not None
    if require_any_pbc:
        apply_mic = apply_mic and bool(np.any(pbc))
    if apply_mic:
        if right is None:
            _, dist = get_distances(left, cell=cell, pbc=pbc)
        else:
            _, dist = get_distances(left, right, cell=cell, pbc=pbc)
        return dist
    if right is None:
        return cdist(left, left)
    return cdist(left, right)


def rotation_matrix_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Right-handed rotation. Row vectors use ``p @ R.T``. Near-zero axis is ``I``."""
    u = np.asarray(axis, dtype=float).reshape(3)
    norm = float(np.linalg.norm(u))
    if norm < ROTATION_AXIS_TOLERANCE:
        return np.eye(3)
    x, y, z = u / norm
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    one_minus_c = 1.0 - c
    return np.array(
        [
            [
                c + x * x * one_minus_c,
                x * y * one_minus_c - z * s,
                x * z * one_minus_c + y * s,
            ],
            [
                y * x * one_minus_c + z * s,
                c + y * y * one_minus_c,
                y * z * one_minus_c - x * s,
            ],
            [
                z * x * one_minus_c - y * s,
                z * y * one_minus_c + x * s,
                c + z * z * one_minus_c,
            ],
        ]
    )


def kabsch_rotation(reference: np.ndarray, moving: np.ndarray) -> np.ndarray:
    """Proper rotation ``R`` so ``moving @ R.T`` fits centered ``reference`` (no reflection)."""
    dim = int(reference.shape[1])
    left, _, right = np.linalg.svd(reference.T @ moving)
    signs = np.ones(dim, dtype=float)
    signs[-1] = float(np.sign(np.linalg.det(left @ right)) or 1.0)
    return left @ np.diag(signs) @ right


def rotate_positions_about_point(
    positions: np.ndarray,
    center: np.ndarray,
    rotation: np.ndarray,
) -> np.ndarray:
    """Rotate row-vector ``positions`` about ``center`` with ``p @ R.T``."""
    pos = np.asarray(positions, dtype=float)
    c = np.asarray(center, dtype=float)
    r = np.asarray(rotation, dtype=float)
    return c + (pos - c) @ r.T


def mic_displacement(
    origin: np.ndarray,
    target: np.ndarray,
    cell: Any,
    pbc: Any,
    *,
    use_mic: bool = True,
) -> np.ndarray:
    """Displacement from ``origin`` to ``target``, optionally through the MIC."""
    delta = np.asarray(target, dtype=float) - np.asarray(origin, dtype=float)
    if not use_mic or pbc is None or cell is None or not np.any(pbc):
        return delta
    mic, _dist = find_mic(delta.reshape(1, 3), cell, pbc)
    return np.asarray(mic[0], dtype=float)
