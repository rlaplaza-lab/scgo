"""Cached slab–mobile and mobile–mobile clash helpers for CutAndSplicePairing.

These replace ASE-GA's ``atoms_too_close`` and ``atoms_too_close_two_sets`` in
the pairing inner loop.  The two key savings are:

1. **Slab-side caching** – slab positions are constant for the lifetime of the
   pairing operator.  The full PBC-expanded slab position array and a vectorised
   type-pair threshold table are built once (``SlabClashChecker.__init__``) and
   reused on every ``is_too_close`` call.

2. **No per-call ``Atoms.copy``** – ``atoms_too_close`` copies the child on
   every call for tag-gathering; the mobile-mobile helper below operates
   directly on the position / number arrays that ``_get_pairing`` already
   produces.

Semantic equivalence with ASE-GA:

- ``SlabClashChecker.is_too_close`` reproduces
  ``atoms_too_close_two_sets(slab, child, blmin)`` exactly: same PBC image set
  (``[-1,0,1]`` per periodic axis), same ``cdist``, same type-pair min.
- ``mobile_too_close_no_copy`` reproduces the ``use_tags=False`` branch of
  ``atoms_too_close(child, blmin, use_tags=False)`` without calling
  ``atoms.copy()`` on every pairing candidate.
- ``mobile_too_close_tagged`` reproduces the ``use_tags=True`` branch of
  ``atoms_too_close``.  It still copies the child before ``gather_atoms_by_tag``
  (required — the original child must remain unmodified for ``slab + child``),
  but is otherwise self-contained and does not call ASE-GA's helper.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy.spatial.distance import cdist

from scgo.ase_ga_patches._tag_gather import (
    gather_atoms_by_tag,
    periodic_sheet_tag_to_skip,
)


class SlabClashChecker:
    """Pre-computes and caches slab positions (with PBC images) for fast clash screening.

    Parameters
    ----------
    slab_numbers:
        Integer atomic numbers of the slab atoms (1-D int array, length N_slab).
    slab_positions:
        Cartesian positions of the slab atoms (N_slab × 3).
    cell:
        3×3 unit-cell matrix (same convention as ``Atoms.get_cell()``).
    pbc:
        Boolean PBC flags along each axis (length 3).
    blmin:
        Minimum-distance dict ``{(Z1, Z2): distance_Angstrom}``.
    """

    def __init__(
        self,
        slab_numbers: np.ndarray,
        slab_positions: np.ndarray,
        cell: np.ndarray,
        pbc: np.ndarray,
        blmin: dict,
    ) -> None:
        self._num_slab = np.asarray(slab_numbers, dtype=int)
        self._unique_slab_z = sorted({int(z) for z in self._num_slab})

        # Build all PBC image displacements once – identical to ASE-GA logic.
        neighbours: list[list[int]] = []
        for i in range(3):
            neighbours.append([-1, 0, 1] if bool(pbc[i]) else [0])
        images: list[np.ndarray] = []
        slab_pos = np.asarray(slab_positions, dtype=float)
        cell_arr = np.asarray(cell, dtype=float)
        for nx, ny, nz in itertools.product(*neighbours):
            displacement = cell_arr.T @ np.array([nx, ny, nz], dtype=float)
            images.append(slab_pos + displacement)
        # Shape: (n_images * N_slab, 3) – pre-expanded for a single cdist call.
        self._slab_expanded = np.vstack(images)  # (n_images * N_slab, 3)
        n_images = len(images)
        # Repeat slab numbers along image dimension for type-pair lookup.
        self._num_slab_expanded = np.tile(
            self._num_slab, n_images
        )  # (n_images * N_slab,)

        self._blmin = blmin

    def is_too_close(
        self, mobile_numbers: np.ndarray, mobile_positions: np.ndarray
    ) -> bool:
        """Return True when any mobile atom is within blmin distance of a slab atom.

        Semantically equivalent to
        ``atoms_too_close_two_sets(slab, child, blmin)`` where ``child``
        contains the mobile atoms and shares cell/pbc with the slab.

        Parameters
        ----------
        mobile_numbers:
            Atomic numbers of the mobile (child) atoms.
        mobile_positions:
            Cartesian positions of the mobile atoms.
        """
        mob_num = np.asarray(mobile_numbers, dtype=int)
        mob_pos = np.asarray(mobile_positions, dtype=float)

        if len(mob_pos) == 0 or len(self._slab_expanded) == 0:
            return False

        # One cdist over all PBC images of the slab – vectorised, no Python loop
        # over images.  Shape: (N_mobile, n_images * N_slab).
        dists = cdist(mob_pos, self._slab_expanded)

        unique_types = sorted({int(z) for z in mob_num} | set(self._unique_slab_z))
        for z1 in unique_types:
            x1 = np.where(mob_num == z1)[0]
            if len(x1) == 0:
                continue
            for z2 in unique_types:
                x2 = np.where(self._num_slab_expanded == z2)[0]
                if len(x2) == 0:
                    continue
                threshold = float(
                    self._blmin.get((z1, z2), self._blmin.get((z2, z1), 0.0))
                )
                if threshold <= 0.0:
                    continue
                if float(np.min(dists[np.ix_(x1, x2)])) < threshold:
                    return True
        return False


def mobile_too_close_no_copy(
    numbers: np.ndarray,
    positions: np.ndarray,
    pbc: np.ndarray,
    cell: np.ndarray,
    blmin: dict,
) -> bool:
    """Mobile-only self-clash check without copying the Atoms object.

    Reproduces the ``use_tags=False`` branch of
    ``atoms_too_close(child, blmin, use_tags=False)`` but operates on raw
    arrays and avoids the ``atoms.copy()`` inside that function.
    """
    num = np.asarray(numbers, dtype=int)
    pos = np.asarray(positions, dtype=float)
    n = len(pos)
    if n < 2:
        return False

    cell_arr = np.asarray(cell, dtype=float)
    unique_types = sorted({int(z) for z in num})

    neighbours: list[list[int]] = []
    for i in range(3):
        neighbours.append([-1, 0, 1] if bool(pbc[i]) else [0])

    for nx, ny, nz in itertools.product(*neighbours):
        displacement = cell_arr.T @ np.array([nx, ny, nz], dtype=float)
        pos_new = pos + displacement
        dists = cdist(pos, pos_new)
        if nx == 0 and ny == 0 and nz == 0:
            # Exclude self-pairs.
            dists += 1e2 * np.eye(n)
        for z1, z2 in itertools.combinations_with_replacement(unique_types, 2):
            x1 = np.where(num == z1)[0]
            x2 = np.where(num == z2)[0]
            threshold = float(blmin.get((z1, z2), blmin.get((z2, z1), 0.0)))
            if threshold <= 0.0:
                continue
            if float(np.min(dists[np.ix_(x1, x2)])) < threshold:
                return True
    return False


def mobile_too_close_tagged(
    child_atoms,
    blmin: dict,
    *,
    system_type: str,
) -> bool:
    """Mobile self-clash when use_tags=True.

    Makes a shallow positions-only copy so that ``gather_atoms_by_tag`` does
    not mutate the child that will later be used in ``self.slab + child``.
    This preserves the semantics of ``atoms_too_close(child, blmin,
    use_tags=True)`` which also copies before gathering.
    """
    gathered = child_atoms.copy()
    gather_atoms_by_tag(
        gathered,
        skip_tag=periodic_sheet_tag_to_skip(system_type),
    )
    pbc = gathered.get_pbc()
    cell = gathered.get_cell()
    numbers = gathered.get_atomic_numbers()
    positions = gathered.get_positions()
    tags = gathered.get_tags()
    n = len(positions)
    if n < 2:
        return False

    cell_arr = np.asarray(cell, dtype=float)
    unique_types = sorted({int(z) for z in numbers})

    neighbours: list[list[int]] = []
    for i in range(3):
        neighbours.append([-1, 0, 1] if bool(pbc[i]) else [0])

    for nx, ny, nz in itertools.product(*neighbours):
        displacement = cell_arr.T @ np.array([nx, ny, nz], dtype=float)
        pos_new = positions + displacement
        dists = cdist(positions, pos_new)
        if nx == 0 and ny == 0 and nz == 0:
            x = tags.reshape(-1, 1)
            dists += 1e2 * (cdist(x.astype(float), x.astype(float)) == 0)
        for z1, z2 in itertools.combinations_with_replacement(unique_types, 2):
            x1 = np.where(numbers == z1)[0]
            x2 = np.where(numbers == z2)[0]
            threshold = float(blmin.get((z1, z2), blmin.get((z2, z1), 0.0)))
            if threshold <= 0.0:
                continue
            if float(np.min(dists[np.ix_(x1, x2)])) < threshold:
                return True
    return False
