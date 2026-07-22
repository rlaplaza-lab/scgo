"""Concatenate core and adsorbate fragment atoms."""

from __future__ import annotations

import numpy as np
from ase import Atoms

from scgo.utils.combine_atoms import concatenate_inherit_cell_pbc


def combine_core_adsorbate(core: Atoms, adsorbate: Atoms) -> Atoms:
    """Concatenate core then adsorbate; inherit cell and PBC from core."""
    return concatenate_inherit_cell_pbc(core, adsorbate)


def expand_cubic_cell_to_fit(atoms: Atoms, margin: float) -> None:
    """Set a cubic cell and center so all atoms lie inside with ``margin`` padding.

    Mutates ``atoms`` in place. For ``pbc=False`` isolated clusters.
    """
    pos = atoms.get_positions()
    if len(pos) == 0:
        return
    span = float(np.max(pos.max(axis=0) - pos.min(axis=0)))
    side = max(span + margin, margin)
    atoms.set_cell([side, side, side])
    atoms.center()
