"""Shared helpers used by more than one mutation operator."""

# fmt: off

from __future__ import annotations


import numpy as np

from ase import Atoms

from scgo.ase_ga_patches._vector_utils import (
    append_unique_unit_vector as _append_unique_unit_vector,
)
from scgo.ase_ga_patches._vector_utils import random_unit_vector as _random_unit_vector
from scgo.initialization.geometry_helpers import validate_cluster_structure
from scgo.initialization.initialization_config import (
    CONNECTIVITY_FACTOR,
    MIN_DISTANCE_FACTOR_DEFAULT,
)
from scgo.system_types.connectivity_factor import (
    ConnectivityFactorInput,
    NormalizedConnectivityFactor,
)
from scgo.utils.rng_helpers import ensure_rng_or_create as _ensure_rng

__all__ = [
    "_append_unique_unit_vector",
    "_ensure_rng",
    "_geometry_candidate_directions",
    "_IDENTITY_ATOL",
    "_mobile_is_connected",
    "_pin_subset_contact_atom",
    "_preserves_mobile_connectivity",
    "_random_unit_vector",
    "_reanchor_mobile_to_slab",
    "_resolve_op_connectivity_factor",
]


_IDENTITY_ATOL = 1e-8


def _resolve_op_connectivity_factor(
    creator,
) -> ConnectivityFactorInput | NormalizedConnectivityFactor:
    """Connectivity factor stamped on a mutation operator, else the module default."""
    return getattr(creator, "connectivity_factor", CONNECTIVITY_FACTOR)


def _geometry_candidate_directions(positions, center_of_mass, slab, rng, max_candidates):
    """Ranked unit directions from slab normal, PCA axes, and random fill."""
    centered = positions - center_of_mass
    candidates = []
    outward = None

    if len(slab) > 0:
        outward = center_of_mass - np.mean(slab.get_positions(), axis=0)
        _append_unique_unit_vector(candidates, outward)

    if len(centered) > 1:
        covariance = np.dot(centered.T, centered)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)
        axes = [eigenvectors[:, index] for index in order]

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


# fmt: on
