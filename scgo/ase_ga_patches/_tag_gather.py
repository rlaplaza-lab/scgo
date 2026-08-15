"""MIC-unwrap same-tag atom groups without running constraint projectors."""

from __future__ import annotations

import numpy as np
from ase import Atoms

from scgo.system_types import SystemType, slab_is_search_target

__all__ = ["gather_atoms_by_tag", "periodic_sheet_tag_to_skip"]


def periodic_sheet_tag_to_skip(system_type: SystemType) -> int | None:
    """Return the tag that must not be MIC-gathered, or ``None``.

    Tag 0 in the GA ``n_top`` slice is a periodic sheet only when the slab
    itself is the search target. For ``surface_cluster`` the same tag is the
    finite nanoparticle core and *must* be gathered so fragments unwrap.
    """
    return 0 if slab_is_search_target(system_type) else None


def gather_atoms_by_tag(atoms: Atoms, *, skip_tag: int | None = None) -> None:
    """Translate same-tag atoms onto the minimum-image convention.

    This is a kinematic unwrap, not a constraint projection: positions are
    written with ``apply_constraint=False``. ASE ``FixBondLengths`` SHAKE is a
    local-relax projector and must not rewrite operator-built coordinates.

    ``skip_tag`` leaves that tag group in place so a periodic lattice is not
    folded onto one image. Pass :func:`periodic_sheet_tag_to_skip` from GA
    operators.
    """
    tags = np.asarray(atoms.get_tags())
    pos = atoms.get_positions()
    for tag in np.unique(tags):
        if skip_tag is not None and int(tag) == int(skip_tag):
            continue
        indices = np.flatnonzero(tags == tag)
        if len(indices) <= 1:
            continue
        vectors = atoms.get_distances(
            int(indices[0]), indices[1:], mic=True, vector=True
        )
        pos[indices[1:]] = pos[indices[0]] + vectors
    atoms.set_positions(pos, apply_constraint=False)
