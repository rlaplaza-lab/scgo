"""Shared Atoms combine / surface-geometry helpers."""

from __future__ import annotations

import numpy as np
from ase import Atoms

from scgo.cluster_adsorbate.geometry import random_rotation_matrix

__all__ = [
    "concatenate_inherit_cell_pbc",
    "random_rotation_matrix",
    "slab_surface_extreme",
    "top_layer_indices",
]


def concatenate_inherit_cell_pbc(base: Atoms, mobile: Atoms) -> Atoms:
    """Concatenate ``base`` then ``mobile``; inherit cell and PBC from ``base``."""
    if len(mobile) == 0:
        return base.copy()
    out = base.copy() + mobile.copy()
    out.set_cell(base.get_cell())
    out.set_pbc(base.get_pbc())
    return out


def slab_surface_extreme(
    positions_or_atoms: Atoms | np.ndarray,
    axis: int,
    *,
    upper: bool = True,
) -> float:
    """Return max (or min) Cartesian coordinate along ``axis``."""
    if isinstance(positions_or_atoms, Atoms):
        pos = positions_or_atoms.get_positions()
    else:
        pos = np.asarray(positions_or_atoms)
    if len(pos) == 0:
        return 0.0
    return float(np.max(pos[:, axis]) if upper else np.min(pos[:, axis]))


def top_layer_indices(
    positions: np.ndarray,
    axis: int,
    *,
    thickness: float = 2.5,
) -> list[int]:
    """Indices of atoms within ``thickness`` Å of the upper surface extreme."""
    pos = np.asarray(positions)
    if len(pos) == 0:
        return []
    top = slab_surface_extreme(pos, axis, upper=True)
    mask = pos[:, axis] >= top - thickness
    indices = [i for i, m in enumerate(mask) if m]
    return indices if indices else list(range(len(pos)))
