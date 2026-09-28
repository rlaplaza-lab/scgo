"""Fast geometric clash prefilter used before expensive local relaxations."""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.geometry import get_distances
from scipy.spatial.distance import cdist

PREFILTER_BLMIN_FACTOR = 0.55

# Cache by (unique Z, id(blmin)). Cleared by GA each generation so recycled ids
# and per-generation empty ``{}`` (prefilter off) cannot accumulate stale entries.
_BLMIN_THRESH_CACHE: dict[
    tuple[tuple[int, ...], int], tuple[np.ndarray, dict[int, int]]
] = {}

# Cache of upper-triangle index pairs for the mobile–mobile clash prefilter,
# avoiding a fresh O(n²) boolean mask allocation on every offspring.
_TRIU_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def clear_blmin_threshold_cache() -> None:
    """Drop cached Z-pair threshold tables (call once per GA generation)."""
    _BLMIN_THRESH_CACHE.clear()


def _triu_cache(n: int) -> tuple[np.ndarray, np.ndarray]:
    return _TRIU_CACHE.setdefault(n, np.triu_indices(n, k=1))


def _blmin_threshold_matrix(
    atomic_numbers: np.ndarray, blmin: dict
) -> tuple[np.ndarray, dict[int, int]]:
    """Map atomic numbers to a dense Z-pair clash-threshold matrix and Z→index."""
    unique_z = tuple(sorted(int(z) for z in np.unique(atomic_numbers)))
    cache_key = (unique_z, id(blmin))
    cached = _BLMIN_THRESH_CACHE.get(cache_key)
    if cached is None:
        z_to_i = {z: i for i, z in enumerate(unique_z)}
        n_u = len(unique_z)
        min_allowed = np.zeros((n_u, n_u), dtype=float)
        for i, zi in enumerate(unique_z):
            for j, zj in enumerate(unique_z):
                min_allowed[i, j] = float(blmin.get((zi, zj), blmin.get((zj, zi), 0.0)))
        mask = min_allowed > 0.0
        thresh = np.zeros((n_u, n_u), dtype=float)
        thresh[mask] = PREFILTER_BLMIN_FACTOR * min_allowed[mask]
        _BLMIN_THRESH_CACHE[cache_key] = (thresh, z_to_i)
    else:
        thresh, z_to_i = cached
    return thresh, z_to_i


def _distance_matrix(
    left: np.ndarray,
    right: np.ndarray | None = None,
    *,
    cell=None,
    pbc=None,
    use_mic: bool = False,
) -> np.ndarray:
    """Pairwise distances; MIC when ``use_mic`` and any PBC flag is set."""
    if use_mic and cell is not None and pbc is not None and bool(np.any(pbc)):
        if right is None:
            _, dist = get_distances(left, cell=cell, pbc=pbc)
        else:
            _, dist = get_distances(left, right, cell=cell, pbc=pbc)
        return dist
    if right is None:
        return cdist(left, left)
    return cdist(left, right)


def fails_fast_geometric_prefilter(
    atoms: Atoms,
    blmin: dict,
    *,
    n_slab: int = 0,
    use_mic: bool = False,
) -> bool:
    """Return True when a severe clash is detected quickly.

    Only mobile atoms (indices ``n_slab:``) participate: mobile–mobile and
    mobile–slab pairs are checked; slab–slab pairs are skipped. Thresholds are
    ``PREFILTER_BLMIN_FACTOR`` times the blmin table entry. When ``use_mic`` is
    True and the structure has any periodic flag, distances use the
    minimum-image convention.
    """
    n_atoms = len(atoms)
    if n_atoms < 2:
        return False
    n_slab_i = max(0, min(int(n_slab), n_atoms))
    n_mobile = n_atoms - n_slab_i
    if n_mobile < 1:
        return False
    if not blmin:
        return False

    numbers = atoms.get_atomic_numbers()
    positions = atoms.get_positions()
    thresh, z_to_i = _blmin_threshold_matrix(numbers, blmin)
    z_index = np.array([z_to_i[int(z)] for z in numbers], dtype=int)
    mobile_pos = positions[n_slab_i:]
    mobile_idx = z_index[n_slab_i:]
    cell = atoms.cell.array if use_mic else None
    pbc = atoms.pbc if use_mic else None

    # Mobile–mobile pairs (upper triangle, cached index pass).
    if n_mobile >= 2:
        mm = _distance_matrix(mobile_pos, cell=cell, pbc=pbc, use_mic=use_mic)
        pair_thresh = thresh[np.ix_(mobile_idx, mobile_idx)]
        iu, ju = _triu_cache(n_mobile)
        mm_u = mm[iu, ju]
        pt_u = pair_thresh[iu, ju]
        if np.any((pt_u > 0.0) & (mm_u < pt_u)):
            return True

    # Mobile–slab pairs.
    if n_slab_i > 0:
        slab_pos = positions[:n_slab_i]
        slab_idx = z_index[:n_slab_i]
        ms = _distance_matrix(mobile_pos, slab_pos, cell=cell, pbc=pbc, use_mic=use_mic)
        pair_thresh = thresh[np.ix_(mobile_idx, slab_idx)]
        if np.any((pair_thresh > 0.0) & (ms < pair_thresh)):
            return True

    return False
