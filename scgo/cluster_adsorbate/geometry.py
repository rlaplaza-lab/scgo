"""Geometry helpers for outward adsorbate placement on clusters."""

from __future__ import annotations

import numpy as np
from numpy.random import Generator

from scgo.utils.geometry_math import rotation_matrix_axis_angle


def outermost_point_along_normal(
    pos: np.ndarray, relative_pos: np.ndarray, n: np.ndarray
) -> np.ndarray:
    """Return the core atom position with largest projection along ``n`` from center."""
    dots = relative_pos @ n
    i = int(np.argmax(dots))
    return pos[i].copy()


def rotation_matrix_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return rotation ``R`` with ``R @ a_hat == b_hat`` (3x3, unit vectors)."""
    a = np.asarray(a, dtype=float).reshape(3)
    b = np.asarray(b, dtype=float).reshape(3)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-14 or nb < 1e-14:
        return np.eye(3)
    a = a / na
    b = b / nb
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    c = float(np.dot(a, b))
    if s < 1e-10:
        if c > 0.0:
            return np.eye(3)
        # 180°: rotate around any axis perpendicular to a
        ortho = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(np.dot(ortho, a)) > 0.9:
            ortho = np.array([0.0, 1.0, 0.0], dtype=float)
        ortho = ortho - np.dot(ortho, a) * a
        ortho = ortho / np.linalg.norm(ortho)
        kx = np.array(
            [
                [0.0, -ortho[2], ortho[1]],
                [ortho[2], 0.0, -ortho[0]],
                [-ortho[1], ortho[0], 0.0],
            ]
        )
        return np.eye(3) + 2.0 * (kx @ kx)
    vx = np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


def random_rotation_matrix(rng: Generator) -> np.ndarray:
    """Uniform random rotation (Haar on SO(3)) via random quaternion."""
    u1, u2, u3 = rng.random(3)
    q1 = np.sqrt(1.0 - u1) * np.sin(2 * np.pi * u2)
    q2 = np.sqrt(1.0 - u1) * np.cos(2 * np.pi * u2)
    q3 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    q4 = np.sqrt(u1) * np.cos(2 * np.pi * u3)
    return np.array(
        [
            [
                1.0 - 2.0 * (q3 * q3 + q4 * q4),
                2.0 * (q2 * q3 - q1 * q4),
                2.0 * (q2 * q4 + q1 * q3),
            ],
            [
                2.0 * (q2 * q3 + q1 * q4),
                1.0 - 2.0 * (q2 * q2 + q4 * q4),
                2.0 * (q3 * q4 - q1 * q2),
            ],
            [
                2.0 * (q2 * q4 - q1 * q3),
                2.0 * (q3 * q4 + q1 * q2),
                1.0 - 2.0 * (q2 * q2 + q3 * q3),
            ],
        ]
    )


def random_spin_about_normal(rng: Generator, n: np.ndarray) -> np.ndarray:
    """Random rotation about unit vector ``n`` (surface normal)."""
    nn = np.linalg.norm(n)
    if nn < 1e-14:
        return np.eye(3)
    n = n / nn
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    return rotation_matrix_axis_angle(n, angle)
