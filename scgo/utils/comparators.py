"""Structural comparison tools for atomic clusters.

This module provides comparators for determining if two cluster structures are
geometrically equivalent, based on sorted interatomic distance analysis as
described in Vilhelmsen and Hammer, PRL 108, 126101 (2012).
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from scipy.spatial.distance import pdist

from scgo.constants import (
    DEFAULT_COMPARATOR_TOL,
    DEFAULT_ENERGY_TOLERANCE,
    DEFAULT_PAIR_COR_MAX,
)
from scgo.exceptions import (
    SCGOValidationError,
)
from scgo.metadata.atoms import get_tag

_SORTED_DIST_FP_INFO_KEY = "_scgo_sorted_dist_fp"


def _sorted_dist_content_key(atoms: Atoms, *, mic: bool) -> tuple:
    """Build a content key that invalidates when geometry/composition changes."""
    positions = np.ascontiguousarray(atoms.get_positions(), dtype=np.float64)
    numbers = np.ascontiguousarray(atoms.get_atomic_numbers(), dtype=np.int32)
    key: tuple = (hash(positions.tobytes()), hash(numbers.tobytes()), bool(mic))
    if mic or np.any(atoms.get_pbc()):
        cell = np.ascontiguousarray(atoms.get_cell().array, dtype=np.float64)
        pbc = tuple(bool(x) for x in atoms.get_pbc())
        key = (*key, hash(cell.tobytes()), pbc)
    return key


def _compute_sorted_dist_list(atoms: Atoms, mic: bool) -> dict[int, np.ndarray]:
    """Compute per-element sorted distance fingerprints without using the cache."""
    numbers = atoms.numbers
    unique_types = set(numbers)
    pair_cor: dict[int, np.ndarray] = {}
    # Honor ``mic`` literally so GA (ASE comparator) and Pure stay coherent when
    # callers pass mic=False on periodic cells (e.g. comparator_use_mic=False).
    use_mic_path = bool(mic)

    all_d: np.ndarray | None = None
    if use_mic_path:
        all_d = atoms.get_all_distances(mic=True)

    for n in unique_types:
        i_un = np.flatnonzero(numbers == n)
        if i_un.size == 0:
            continue

        if not use_mic_path:
            positions = atoms.get_positions()[i_un]
            d = pdist(positions).tolist()
        else:
            if all_d is None:
                raise TypeError(
                    "all_d distance matrix is required when use_mic_path=True"
                )
            sub = all_d[np.ix_(i_un, i_un)]
            # Upper triangle excluding diagonal (same order as nested get_distance).
            d = sub[np.triu_indices(len(i_un), k=1)].tolist()

        d.sort()
        pair_cor[n] = np.array(d)
    return pair_cor


def get_sorted_dist_list(atoms: Atoms, mic: bool = False) -> dict[int, np.ndarray]:
    """Calculates a dictionary of sorted interatomic distances for an Atoms object.

    This utility method is used to generate a structural fingerprint of a cluster
    by calculating all interatomic distances for each element type and sorting them.

    Results are cached on ``atoms.info`` under ``_scgo_sorted_dist_fp`` and
    invalidated when positions, numbers, or (for MIC) cell/PBC change.

    Args:
        atoms: The Atoms object for which to calculate the distances.
        mic: Whether to use the minimum image convention for periodic systems.
            Defaults to False.

    Returns:
        A dictionary where keys are atomic numbers (integers) and values are
        sorted 1D numpy arrays of interatomic distances for that element type.
    """
    content_key = _sorted_dist_content_key(atoms, mic=mic)
    cached = atoms.info.get(_SORTED_DIST_FP_INFO_KEY)
    if (
        isinstance(cached, dict)
        and cached.get("content_key") == content_key
        and cached.get("mic") == bool(mic)
        and isinstance(cached.get("pair_cor"), dict)
    ):
        return cached["pair_cor"]

    pair_cor = _compute_sorted_dist_list(atoms, mic=mic)
    atoms.info[_SORTED_DIST_FP_INFO_KEY] = {
        "content_key": content_key,
        "mic": bool(mic),
        "pair_cor": pair_cor,
    }
    return pair_cor


def get_mobile_atom_indices(atoms: Atoms) -> np.ndarray:
    """Return indices for atoms not constrained by ``FixAtoms``.

    If no fixed atoms are present (or all atoms are fixed), this falls back to
    all atom indices to preserve historical comparison behavior.
    """
    n_atoms = len(atoms)
    fixed_mask = np.zeros(n_atoms, dtype=bool)
    for constraint in getattr(atoms, "constraints", ()):
        if isinstance(constraint, FixAtoms):
            idx = np.asarray(constraint.get_indices(), dtype=int)
            fixed_mask[idx] = True

    if not np.any(fixed_mask):
        return np.arange(n_atoms, dtype=int)

    mobile = np.flatnonzero(~fixed_mask).astype(int, copy=False)
    if mobile.size == 0:
        return np.arange(n_atoms, dtype=int)
    return mobile


def _resolve_n_slab_metadata(a1: Atoms, a2: Atoms) -> int | None:
    """Read a shared ``n_slab_atoms`` partition from structure tags, if present.

    Returns the integer slab count when both structures agree on a positive
    ``n_slab_atoms`` tag, otherwise ``None`` (caller falls back to constraints).
    """
    n1 = get_tag(a1, "n_slab_atoms", None)
    n2 = get_tag(a2, "n_slab_atoms", None)
    if n1 is None and n2 is None:
        return None
    try:
        n1_i = int(n1) if n1 is not None else None
        n2_i = int(n2) if n2 is not None else None
    except (TypeError, ValueError):
        return None
    if n1_i is not None and n2_i is not None and n1_i != n2_i:
        return None
    resolved = n1_i if n1_i is not None else n2_i
    if resolved is None or resolved <= 0:
        return None
    return resolved


def get_shared_mobile_atom_indices(
    a1: Atoms,
    a2: Atoms,
    *,
    n_slab: int | None = None,
) -> np.ndarray:
    """Return index set suitable for comparing two structures.

    When ``n_slab`` is set (e.g. from :class:`~scgo.surface.config.SurfaceSystemConfig`
    at TS time), indices ``n_slab:`` are used on both structures. This is the
    authoritative partition for surface workflows and does not require
    ``FixAtoms`` or stored ``n_slab_atoms`` metadata on loaded minima.

    When ``n_slab`` is ``None``, a stored ``n_slab_atoms`` tag (from
    ``key_value_pairs`` metadata on either structure) provides the same
    surface/adsorbate partition when ``FixAtoms`` constraints are absent.
    Otherwise the intersection of mobile (non-``FixAtoms``) indices is used.
    Raises if the chosen set is empty.
    """
    if len(a1) != len(a2):
        raise SCGOValidationError(
            f"The two configurations must have the same number of atoms: {len(a1)} vs {len(a2)}",
        )

    if n_slab is None:
        n_slab = _resolve_n_slab_metadata(a1, a2)

    if n_slab is not None:
        n_slab_i = int(n_slab)
        if n_slab_i < 0 or n_slab_i >= len(a1):
            raise SCGOValidationError(
                f"n_slab={n_slab_i} invalid for structure comparison (len={len(a1)})."
            )
        mobile = np.arange(n_slab_i, len(a1), dtype=int)
        if mobile.size == 0:
            raise SCGOValidationError(
                "No mobile atoms after applying surface n_slab partition."
            )
        return mobile

    idx1 = get_mobile_atom_indices(a1)
    idx2 = get_mobile_atom_indices(a2)
    shared = np.intersect1d(idx1, idx2, assume_unique=False)
    if shared.size == 0:
        raise SCGOValidationError("No shared mobile atoms across endpoints.")
    return shared.astype(int, copy=False)


class PureInteratomicDistanceComparator:
    """A structural comparator based on sorted interatomic distances.

    This class implements the comparison criteria described in
    L.B. Vilhelmsen and B. Hammer, PRL, 108, 126101 (2012),
    but without considering energy differences. It is used to determine if two
    cluster geometries are structurally equivalent.

    Args:
        n_top: The number of atoms from the top of the Atoms object to include
            in the comparison. If None or 0, all atoms are used. Defaults to None.
        tol: The tolerance for the cumulative structural difference (eq. 2 in
            the reference paper). Defaults to `DEFAULT_COMPARATOR_TOL`.
        pair_cor_max: The tolerance for the maximum single interatomic distance
            difference (eq. 3 in the reference paper). Defaults to `DEFAULT_PAIR_COR_MAX`.
        dE: A placeholder for API consistency with other ASE comparators; it is
            not used in this implementation. Defaults to `DEFAULT_ENERGY_TOLERANCE`.
        mic: Whether to use the minimum image convention when calculating
            distances. Defaults to False. Honored literally even when the cell
            has PBC (does not auto-enable MIC from ``atoms.pbc``). Set True for
            adsorbates on periodic slabs via
            :func:`scgo.system_types.resolve_structure_mic`.
    """

    def __init__(
        self,
        n_top: int | None = None,
        tol: float = DEFAULT_COMPARATOR_TOL,
        pair_cor_max: float = DEFAULT_PAIR_COR_MAX,
        dE: float = DEFAULT_ENERGY_TOLERANCE,
        mic: bool = False,
    ):
        self.tol = tol
        self.pair_cor_max = pair_cor_max
        self.dE = dE  # Not used, but kept for API consistency
        self.n_top = n_top or 0
        self.mic = mic

    def looks_like(self, a1: Atoms, a2: Atoms) -> bool:
        """Determines if two structures are structurally similar.

        This method calculates the structural differences using `get_differences`
        and returns True if both the cumulative and maximum differences are
        below their respective tolerances.

        Args:
            a1: The first Atoms object.
            a2: The second Atoms object.

        Returns:
            True if the structures are considered similar, False otherwise.
            Structures with different compositions are never similar (False).
        """
        cum_diff, max_diff = self.get_differences(a1, a2)

        return cum_diff < self.tol and max_diff < self.pair_cor_max

    def get_differences(self, a1: Atoms, a2: Atoms) -> tuple[float, float]:
        """Compute cumulative and maximum structural differences between two Atoms.

        Differences are based on their sorted interatomic distances.

        Args:
            a1: The first Atoms object.
            a2: The second Atoms object.

        Returns:
            A tuple containing (cumulative_difference, max_difference).

        Raises:
            SCGOValidationError: If the two Atoms objects do not have the same
                number of atoms.
        """
        if len(a1) != len(a2):
            raise SCGOValidationError(
                "The two configurations must have the same number of atoms",
            )

        # If n_top is defined, only compare the specified number of atoms
        a1top = a1[-self.n_top :] if self.n_top > 0 else a1
        a2top = a2[-self.n_top :] if self.n_top > 0 else a2
        return self.__compare_structure__(a1top, a2top)

    def __compare_structure__(self, a1: Atoms, a2: Atoms) -> tuple[float, float]:
        """Private method to perform the core structural comparison.

        Args:
            a1: The first Atoms object (or subset).
            a2: The second Atoms object (or subset).

        Returns:
            A tuple containing the cumulative difference and the maximum difference.
            Structures with different compositions (element *counts* included)
            are reported as a maximal non-match ``(inf, inf)`` rather than
            raising, so callers such as :meth:`looks_like` simply return False.
        """
        if Counter(a1.numbers) != Counter(a2.numbers):
            # Different compositions can never "look like" each other; report a
            # maximal difference instead of raising so population dedup and
            # diversity scoring keep working on mixed-composition pools.
            return (float("inf"), float("inf"))

        p1 = get_sorted_dist_list(a1, mic=self.mic)
        p2 = get_sorted_dist_list(a2, mic=self.mic)
        numbers = a1.numbers
        total_cum_diff = 0.0
        max_diff = 0.0

        for n in p1:
            c1 = p1[n]
            c2 = p2[n]

            if len(c1) != len(c2):
                # This should not happen if compositions are the same
                raise SCGOValidationError(
                    "Mismatch in number of distances being compared."
                )

            if len(c1) == 0:
                continue

            total_dist_sum = np.sum(c1)
            if total_dist_sum <= 1e-10:  # Use epsilon for floating-point comparison
                continue

            d = np.abs(c1 - c2)
            cum_diff_for_type = np.sum(d)
            max_diff_for_type = np.max(d)

            max_diff = max(max_diff, max_diff_for_type)

            num_atoms_of_type = float(np.sum(numbers == n))  # Vectorized operation
            total_cum_diff += (
                cum_diff_for_type
                / total_dist_sum
                * num_atoms_of_type
                / float(len(numbers))
            )

        return (total_cum_diff, max_diff)
