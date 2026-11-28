"""Shared helpers used by more than one mutation operator."""

from __future__ import annotations


import numpy as np

from ase import Atoms
from ase.geometry import find_mic
from ase_ga.offspring_creator import OffspringCreator
from ase_ga.utilities import atoms_too_close, atoms_too_close_two_sets

from scgo.ase_ga_patches._vector_utils import (
    append_unique_unit_vector as _append_unique_unit_vector,
)
from scgo.ase_ga_patches._vector_utils import random_unit_vector as _random_unit_vector
from scgo.initialization.atomic_radii import get_covalent_radius
from scgo.initialization.geometry_helpers import (
    _find_connected_components,
    validate_cluster_structure,
)
from scgo.initialization.initialization_config import (
    BOND_REPAIR_INSET,
    CONNECTIVITY_FACTOR,
    MIN_DISTANCE_FACTOR_DEFAULT,
)
from scgo.initialization.steric_scoring import steric_deficit as _steric_deficit
from scgo.initialization.steric_scoring import (
    steric_deficit_two_sets as _steric_deficit_two_sets,
)
from scgo.system_types.connectivity_factor import (
    ConnectivityFactorInput,
    NormalizedConnectivityFactor,
    normalize_connectivity_factor,
    pair_bond_threshold,
)
from scgo.system_types.policy import SystemType, get_system_policy
from scgo.utils.geometry_math import mic_displacement
from scgo.utils.rng_helpers import ensure_rng_or_create as _ensure_rng

__all__ = [
    "_append_unique_unit_vector",
    "_ensure_rng",
    "_filter_unique_tags",
    "_geometry_candidate_directions",
    "_pca_sorted_eigenvectors",
    "_IDENTITY_ATOL",
    "_accept_connected_candidate",
    "_accept_connected_kwargs",
    "_disconnection_gap",
    "_fails_only_slab_clash",
    "_mobile_clashes",
    "_mobile_is_connected",
    "_mutation_steric_score",
    "MutationOperator",
    "_pin_subset_contact_atom",
    "_preserves_mobile_connectivity",
    "_reconnect_mobile_components",
    "_random_unit_vector",
    "_reanchor_mobile_to_slab",
    "_resolve_op_connectivity_factor",
]


_IDENTITY_ATOL = 1e-8


def _mutation_description(descriptor: str) -> str:
    """``RattleMutation`` -> ``mutation: rattle``."""
    name = descriptor.removesuffix("Mutation")
    parts: list[str] = []
    for char in name:
        if char.isupper() and parts:
            parts.append("_")
        parts.append(char.lower())
    return "mutation: " + "".join(parts)


class MutationOperator(OffspringCreator):
    """Shared creator setup and ``get_new_individual`` for SCGO mutations.

    Subclasses set ``descriptor`` to the class name. The GA description is
    ``mutation:`` plus that name in snake_case with the ``Mutation`` suffix
    removed. A subclass that does not override ``descriptor`` keeps its
    parent's label (``CustomPermutationMutation`` stays ``permutation``).
    """

    descriptor: str

    def _init_operator(
        self,
        verbose: bool,
        rng,
        system_type: SystemType,
        *,
        n_top,
        blmin,
        test_dist_to_slab: bool = True,
        use_tags: bool = False,
        target_tags=None,
        surface_normal_axis: int | None = None,
    ) -> None:
        rng = _ensure_rng(rng)
        OffspringCreator.__init__(self, verbose, rng=rng)
        self.blmin = blmin
        self.n_top = n_top
        self.system_type = system_type
        self._policy = get_system_policy(system_type)
        self.test_dist_to_slab = test_dist_to_slab
        self.use_tags = use_tags
        self.target_tags = target_tags
        if surface_normal_axis is not None:
            self.surface_normal_axis = surface_normal_axis
        self.descriptor = type(self).descriptor
        self.min_inputs = 1

    def get_new_individual(self, parents):
        # _finalize imports helpers from this module, so the import stays local.
        from scgo.ase_ga_patches.mutations._finalize import _finalize_mutant

        parent = parents[0]
        mutant = self.mutate(parent)
        return _finalize_mutant(
            self,
            parent,
            mutant,
            _mutation_description(self.descriptor),
        )


def split_slab_and_mobile(
    atoms: Atoms,
    n_top: int | None,
) -> tuple[Atoms, Atoms, int]:
    """Split combined slab+mobile ``Atoms`` into fixed slab and mobile region."""
    n_mobile = len(atoms) if n_top is None else n_top
    split = len(atoms) - n_mobile
    return atoms[:split], atoms[split:], n_mobile


def swap_tag_group_centroids(
    pos: np.ndarray,
    tags: np.ndarray,
    unique_tags: np.ndarray,
    i: int,
    j: int,
) -> tuple:
    """Exchange tag-group centroids in ``pos``.

    Returns ``(ind1, ind2, shift1, shift2)`` so callers can revert on clash.
    """
    ind1 = np.where(tags == unique_tags[i])
    ind2 = np.where(tags == unique_tags[j])
    cop1 = np.mean(pos[ind1], axis=0)
    cop2 = np.mean(pos[ind2], axis=0)
    shift1 = cop2 - cop1
    shift2 = cop1 - cop2
    pos[ind1] += shift1
    pos[ind2] += shift2
    return ind1, ind2, shift1, shift2


def _filter_unique_tags(
    unique_tags: np.ndarray,
    target_tags: list[int] | tuple[int, ...] | set[int] | None,
) -> np.ndarray | None:
    """Return tags to mutate, or ``None`` when none remain."""
    if target_tags is not None:
        target_tags_set = set(target_tags)
        unique_tags = np.array([t for t in unique_tags if t in target_tags_set])
    if len(unique_tags) == 0:
        return None
    return unique_tags


def _resolve_op_connectivity_factor(
    creator,
) -> ConnectivityFactorInput | NormalizedConnectivityFactor:
    """Connectivity factor stamped on a mutation operator, else the module default."""
    return getattr(creator, "connectivity_factor", CONNECTIVITY_FACTOR)


def _pca_sorted_eigenvectors(centered, *, descending: bool = False):
    """Return PCA eigenvectors of *centered* coords, sorted by eigenvalue."""
    covariance = np.dot(centered.T, centered)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)
    if descending:
        order = order[::-1]
    return [eigenvectors[:, index] for index in order]


def _geometry_candidate_directions(
    positions, center_of_mass, slab, rng, max_candidates
):
    """Ranked unit directions from slab normal, PCA axes, and random fill."""
    centered = positions - center_of_mass
    candidates = []
    outward = None

    if len(slab) > 0:
        outward = center_of_mass - np.mean(slab.get_positions(), axis=0)
        _append_unique_unit_vector(candidates, outward)

    if len(centered) > 1:
        axes = _pca_sorted_eigenvectors(centered)

        for axis in axes:
            oriented_axis = axis
            if outward is not None and np.dot(oriented_axis, outward) < 0.0:
                oriented_axis = -oriented_axis
            _append_unique_unit_vector(candidates, oriented_axis)

        if len(axes) >= 2:
            blends = [axes[0] + axes[-1]]
            if len(axes) >= 3:
                blends.append(axes[1] + axes[-1])
            else:
                blends.append(axes[0] + axes[1])
            for axis in blends:
                oriented_axis = axis
                if outward is not None and np.dot(oriented_axis, outward) < 0.0:
                    oriented_axis = -oriented_axis
                _append_unique_unit_vector(candidates, oriented_axis)

        radial_norms = np.linalg.norm(centered, axis=1)
        if len(radial_norms) > 0:
            radial_axis = centered[int(np.argmax(radial_norms))]
            if outward is not None and np.dot(radial_axis, outward) < 0.0:
                radial_axis = -radial_axis
            _append_unique_unit_vector(candidates, radial_axis)
    else:
        _append_unique_unit_vector(candidates, np.array([0.0, 0.0, 1.0]))

    attempts = 0
    while len(candidates) < max_candidates and attempts < 100:
        axis = _random_unit_vector(rng)
        if outward is not None and np.dot(axis, outward) < 0.0:
            axis = -axis
        _append_unique_unit_vector(candidates, axis)
        attempts += 1

    return candidates[:max_candidates]


def _pin_subset_contact_atom(parent_positions, new_positions, subset_mask):
    """Translate the mutated subset so its parent contact atom stays put.

    The contact atom is the subset atom closest to any leftover atom. Tagged
    flattening around the subset COM otherwise pulls that binding atom off the
    remainder (core or adsorbate). No leftover atoms → returned unchanged.
    """
    subset_mask = np.asarray(subset_mask, dtype=bool)
    leftover_mask = ~subset_mask
    if not np.any(subset_mask) or not np.any(leftover_mask):
        return new_positions
    parent_positions = np.asarray(parent_positions, dtype=float)
    new_positions = np.asarray(new_positions, dtype=float).copy()
    parent_sub = parent_positions[subset_mask]
    leftover = parent_positions[leftover_mask]
    delta = parent_sub[:, None, :] - leftover[None, :, :]
    dist2 = np.einsum("ijk,ijk->ij", delta, delta)
    bind_local = int(np.argmin(np.min(dist2, axis=1)))
    bind_idx = int(np.flatnonzero(subset_mask)[bind_local])
    new_positions[subset_mask] += parent_positions[bind_idx] - new_positions[bind_idx]
    return new_positions


def _reanchor_mobile_to_slab(
    parent_mobile: Atoms,
    mutant_mobile: Atoms,
    slab: Atoms,
    surface_normal_axis: int,
) -> Atoms:
    """Snap ``mutant_mobile`` back to the parent's adsorption height along the normal.

    Translates ``mutant_mobile`` only along ``surface_normal_axis`` so that the
    lowest-atom coordinate of the mobile region matches that of ``parent_mobile``.
    In-plane drift is irrelevant because the slab is periodic along the other axes,
    so this keeps slab contact without wrapping through PBC.

    Returns a new ``Atoms`` with the same numbers/cell/pbc/tags as
    ``mutant_mobile``. When ``len(slab) == 0`` (gas phase) the helper is a no-op
    and returns ``mutant_mobile`` unchanged, so gas behaviour is untouched.
    """
    if len(slab) == 0:
        return mutant_mobile

    axis = int(surface_normal_axis)
    positions = mutant_mobile.get_positions().copy()
    parent_low = float(np.min(parent_mobile.get_positions()[:, axis]))
    mutant_low = float(np.min(positions[:, axis]))
    shift = parent_low - mutant_low
    positions[:, axis] += shift

    return Atoms(
        numbers=mutant_mobile.get_atomic_numbers(),
        positions=positions,
        cell=mutant_mobile.get_cell(),
        pbc=mutant_mobile.get_pbc(),
        tags=mutant_mobile.get_tags(),
    )


def _mobile_is_connected(
    mobile: Atoms,
    *,
    use_mic: bool,
    connectivity_factor: ConnectivityFactorInput
    | NormalizedConnectivityFactor = CONNECTIVITY_FACTOR,
) -> bool:
    """Return True when ``mobile`` is a single connected component (or has <2 atoms)."""
    if len(mobile) < 2:
        return True
    ok, _msg = validate_cluster_structure(
        mobile,
        MIN_DISTANCE_FACTOR_DEFAULT,
        connectivity_factor,
        check_clashes=False,
        check_connectivity=True,
        use_mic=use_mic,
    )
    return bool(ok)


def _preserves_mobile_connectivity(
    parent_mobile: Atoms,
    mutant_mobile: Atoms,
    *,
    use_mic: bool,
    connectivity_factor: ConnectivityFactorInput
    | NormalizedConnectivityFactor = CONNECTIVITY_FACTOR,
) -> bool:
    """True when the mutant is connected, or the parent was already disconnected."""
    if _mobile_is_connected(
        mutant_mobile, use_mic=use_mic, connectivity_factor=connectivity_factor
    ):
        return True
    return not _mobile_is_connected(
        parent_mobile, use_mic=use_mic, connectivity_factor=connectivity_factor
    )


def _prepare_mobile_target(operator, atoms):
    """Return slab, mobile arrays, and tag mask, or ``None`` if nothing matches."""
    n_mobile = len(atoms) if operator.n_top is None else operator.n_top
    slab = atoms[: len(atoms) - n_mobile]
    top = atoms[len(atoms) - n_mobile :]
    tags = top.get_tags()
    unique_tags = _filter_unique_tags(np.unique(tags), operator.target_tags)
    if unique_tags is None:
        return None
    mask = np.isin(tags, unique_tags)
    if not np.any(mask):
        return None
    return (
        slab,
        top,
        top.get_positions(),
        top.get_atomic_numbers(),
        top.get_cell(),
        top.get_pbc(),
        tags,
        mask,
    )


def _mobile_clashes(
    mobile: Atoms,
    *,
    blmin,
    slab: Atoms,
    test_dist_to_slab: bool,
    use_tags: bool,
) -> bool:
    """True when ``mobile`` violates blmin with itself or with ``slab``."""
    if atoms_too_close(mobile, blmin, use_tags=use_tags):
        return True
    return bool(
        test_dist_to_slab
        and len(slab) > 0
        and atoms_too_close_two_sets(mobile, slab, blmin)
    )


def _component_groups(mobile, connectivity_factor, use_mic):
    components = _find_connected_components(mobile, connectivity_factor, use_mic)
    return sorted(
        (np.asarray(idxs, dtype=int) for idxs in components.values() if idxs),
        key=len,
        reverse=True,
    )


def _closest_pair_gap(positions, group_a, group_b, atoms, use_mic):
    """Shortest gap between two index groups, and the atom indices of that pair."""
    delta = positions[group_a][:, None, :] - positions[group_b][None, :, :]
    flat = np.ascontiguousarray(delta.reshape(-1, 3))
    if use_mic:
        _mic, dist = find_mic(flat, atoms.cell, atoms.pbc)
        distances = np.asarray(dist, dtype=float)
    else:
        distances = np.linalg.norm(flat, axis=1)
    flat_index = int(np.argmin(distances))
    index_a, index_b = np.unravel_index(flat_index, (len(group_a), len(group_b)))
    return (
        float(distances[flat_index]),
        int(group_a[index_a]),
        int(group_b[index_b]),
    )


def _clamp_repair_step(atoms, step, use_mic):
    """Keep one repair step from crossing the periodic cell."""
    step = float(step)
    if step <= 0.0 or not use_mic or not np.any(atoms.pbc):
        return max(0.0, step)
    lengths = np.asarray(atoms.cell.lengths(), dtype=float)
    periodic = lengths[np.asarray(atoms.pbc, dtype=bool)]
    periodic = periodic[periodic > 1e-8]
    if len(periodic) == 0:
        return step
    return min(step, 0.45 * float(np.min(periodic)))


def _disconnection_gap(
    mobile: Atoms,
    *,
    use_mic: bool,
    connectivity_factor: ConnectivityFactorInput
    | NormalizedConnectivityFactor = CONNECTIVITY_FACTOR,
) -> float:
    """Shortest gap between the largest component and any other fragment.

    ``0`` when ``mobile`` is already a single component.
    """
    if len(mobile) < 2:
        return 0.0
    groups = _component_groups(mobile, connectivity_factor, use_mic)
    if len(groups) <= 1:
        return 0.0
    positions = mobile.get_positions()
    largest = groups[0]
    gap = float("inf")
    for smaller in groups[1:]:
        pair_gap, _small, _large = _closest_pair_gap(
            positions, smaller, largest, mobile, use_mic
        )
        gap = min(gap, pair_gap)
    return gap


def _reconnect_mobile_components(
    mobile: Atoms,
    *,
    use_mic: bool,
    connectivity_factor: ConnectivityFactorInput
    | NormalizedConnectivityFactor = CONNECTIVITY_FACTOR,
) -> Atoms:
    """Pull the nearest fragment onto the largest component in one step.

    The step follows the minimum-image gap and stops just inside the bond
    cutoff. It is clamped so the fragment cannot tunnel across the cell.
    """
    work = mobile.copy()
    groups = _component_groups(work, connectivity_factor, use_mic)
    if len(groups) < 2:
        return work
    positions = work.get_positions()
    symbols = work.get_chemical_symbols()
    spec = normalize_connectivity_factor(connectivity_factor)
    largest = groups[0]
    best: tuple[float, int, int, np.ndarray] | None = None
    for smaller in groups[1:]:
        gap, small_idx, large_idx = _closest_pair_gap(
            positions, smaller, largest, work, use_mic
        )
        if best is None or gap < best[0]:
            best = (gap, small_idx, large_idx, smaller)
    if best is None:
        return work
    gap, small_idx, large_idx, smaller = best
    if gap <= 1e-8:
        return work
    threshold = pair_bond_threshold(
        get_covalent_radius(symbols[small_idx]),
        get_covalent_radius(symbols[large_idx]),
        symbols[small_idx],
        symbols[large_idx],
        spec,
    )
    needed = _clamp_repair_step(work, gap - BOND_REPAIR_INSET * threshold, use_mic)
    if needed <= 1e-8:
        return work
    direction = mic_displacement(
        positions[small_idx],
        positions[large_idx],
        work.cell,
        work.pbc,
        use_mic=bool(use_mic),
    )
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-8:
        return work
    updated = positions.copy()
    updated[smaller] += (needed / norm) * direction
    work.set_positions(updated)
    return work


def _fails_only_slab_clash(
    parent_mobile: Atoms,
    candidate_mobile: Atoms,
    *,
    blmin,
    slab: Atoms,
    test_dist_to_slab: bool,
    use_tags: bool,
    use_mic: bool,
    connectivity_factor: ConnectivityFactorInput
    | NormalizedConnectivityFactor = CONNECTIVITY_FACTOR,
) -> bool:
    """True when the mobile region is legal and only the slab contact clashes."""
    if atoms_too_close(candidate_mobile, blmin, use_tags=use_tags):
        return False
    if not _preserves_mobile_connectivity(
        parent_mobile,
        candidate_mobile,
        use_mic=use_mic,
        connectivity_factor=connectivity_factor,
    ):
        return False
    return bool(
        test_dist_to_slab
        and len(slab) > 0
        and atoms_too_close_two_sets(candidate_mobile, slab, blmin)
    )


def _mutation_steric_score(
    positions: np.ndarray,
    numbers: np.ndarray,
    slab: Atoms,
    blmin: dict,
) -> float:
    """Self steric deficit plus optional slab–mobile deficit for ranking trials."""
    score = float(_steric_deficit(positions, numbers, blmin))
    if len(slab) > 0:
        score += float(
            _steric_deficit_two_sets(
                positions,
                numbers,
                slab.get_positions(),
                slab.numbers,
                blmin,
            )[0]
        )
    return score


def _accept_connected_kwargs(operator, slab):
    """Clash and connectivity arguments shared by the local operators."""
    return {
        "blmin": operator.blmin,
        "slab": slab,
        "test_dist_to_slab": operator.test_dist_to_slab,
        "use_tags": operator.use_tags,
        "use_mic": bool(operator._policy.uses_surface),
        "connectivity_factor": _resolve_op_connectivity_factor(operator),
    }


def _accept_connected_candidate(
    parent_mobile: Atoms,
    candidate_mobile: Atoms,
    *,
    blmin,
    slab: Atoms,
    test_dist_to_slab: bool,
    use_tags: bool,
    use_mic: bool,
    connectivity_factor: ConnectivityFactorInput
    | NormalizedConnectivityFactor = CONNECTIVITY_FACTOR,
) -> Atoms | None:
    """Accept ``candidate_mobile`` when it is clash-free and stays connected.

    A disconnected candidate is declined. Callers that want a repair do that
    once, outside this check, so a fragment pull cannot silently undo the move.
    """
    if _mobile_clashes(
        candidate_mobile,
        blmin=blmin,
        slab=slab,
        test_dist_to_slab=test_dist_to_slab,
        use_tags=use_tags,
    ):
        return None
    if _preserves_mobile_connectivity(
        parent_mobile,
        candidate_mobile,
        use_mic=use_mic,
        connectivity_factor=connectivity_factor,
    ):
        return candidate_mobile
    return None
