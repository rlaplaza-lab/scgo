"""Periodic boundary helpers for slab systems."""

from __future__ import annotations

import numpy as np
from ase import Atoms

from scgo.exceptions import SCGOValidationError
from scgo.utils.logging import get_logger

logger = get_logger(__name__)


def normalize_slab_pbc(
    slab: Atoms,
    *,
    surface_normal_axis: int = 2,
) -> Atoms:
    """Ensure ``slab`` uses slab-like PBC: periodic in-plane, open along the vacuum axis.

    Mutates ``slab`` in place. Typical ASE builders (``fcc111``, ``graphene``, etc.)
    already set ``pbc`` to ``(True, True, False)`` for vacuum along ``z``; this
    helper downgrades accidental full 3D periodicity (e.g. ``slab.pbc = True``) so
    calculators see a true slab during relaxation and global optimization.

    In-plane periodicity follows the cell: an axis with a zero-length (degenerate)
    cell vector is reported as non-periodic even when the caller set ``pbc=True``.

    Args:
        slab: Substrate ``Atoms`` object.
        surface_normal_axis: Index of the Cartesian axis with vacuum (default 2 = z).

    Returns:
        The same ``slab`` instance (for chaining).

    Raises:
        SCGOValidationError: If ``surface_normal_axis`` is not 0, 1, or 2.
    """
    if surface_normal_axis not in (0, 1, 2):
        raise SCGOValidationError("surface_normal_axis must be 0, 1, or 2")

    pbc = np.asarray(slab.pbc, dtype=bool).copy()
    in_plane = [i for i in range(3) if i != surface_normal_axis]

    for axis in in_plane:
        # Assign (not just set True): a degenerate/zero-length in-plane vector
        # must also turn periodicity *off* when the caller requested pbc=True.
        pbc[axis] = float(np.linalg.norm(slab.cell[axis])) > 1e-8

    old = tuple(bool(x) for x in slab.pbc)
    pbc[surface_normal_axis] = False
    slab.pbc = pbc
    if old != tuple(bool(x) for x in slab.pbc):
        logger.debug(
            "Normalized slab PBC from %s to %s (vacuum axis %d)",
            old,
            tuple(bool(x) for x in slab.pbc),
            surface_normal_axis,
        )

    return slab
