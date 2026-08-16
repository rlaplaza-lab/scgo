"""Structure validation for core + adsorbate (connectivity and clashes)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from ase import Atoms

from scgo.initialization.atomic_radii import get_covalent_radius
from scgo.initialization.geometry_helpers import (
    _find_connected_components,
    pairwise_distances,
    validate_cluster_structure,
)
from scgo.initialization.initialization_config import (
    CONNECTIVITY_FACTOR,
    MIN_DISTANCE_FACTOR_DEFAULT,
)
from scgo.system_types.connectivity_factor import (
    ConnectivityFactorInput,
    NormalizedConnectivityFactor,
    bond_threshold_cross_matrix,
    normalize_connectivity_factor,
)

# Covalent H-X contact distance (Å) above which an inter-fragment H contact is
# treated as a hydrogen bond rather than a newly formed covalent bond.
_H_CONTACT_THRESHOLD_A = 1.15


def validate_combined_cluster_structure(
    atoms: Atoms,
    *,
    min_distance_factor: float = MIN_DISTANCE_FACTOR_DEFAULT,
    connectivity_factor: ConnectivityFactorInput
    | NormalizedConnectivityFactor = CONNECTIVITY_FACTOR,
    check_clashes: bool = True,
    check_connectivity: bool = True,
    use_mic: bool = False,
) -> tuple[bool, str]:
    """Validate core + adsorbate structure (clashes and connectivity). Delegates to cluster init rules."""
    return validate_cluster_structure(
        atoms,
        min_distance_factor,
        connectivity_factor,
        check_clashes=check_clashes,
        check_connectivity=check_connectivity,
        use_mic=use_mic,
    )


def validate_adsorbate_fragment_integrity(
    atoms: Atoms,
    *,
    n_slab: int,
    n_core_mobile: int,
    adsorbate_fragment_lengths: Sequence[int],
    connectivity_factor: ConnectivityFactorInput
    | NormalizedConnectivityFactor = CONNECTIVITY_FACTOR,
    use_mic: bool = False,
) -> tuple[bool, str]:
    """Validate that adsorbate fragments preserve their chemical identities.

    Enforces two-way integrity:
    1. each fragment remains internally connected (no dissociation)
    2. no new bonds form between atoms that belong to different fragments
    """
    if not adsorbate_fragment_lengths:
        return True, ""

    n_atoms = len(atoms)
    if n_slab < 0 or n_slab > n_atoms:
        return False, f"Invalid n_slab={n_slab} for len(atoms)={n_atoms}"

    mobile_len = n_atoms - n_slab
    if n_core_mobile < 0 or n_core_mobile > mobile_len:
        return (
            False,
            f"Invalid n_core_mobile={n_core_mobile} for mobile length={mobile_len}",
        )

    ads_mobile_len = mobile_len - n_core_mobile
    if sum(int(x) for x in adsorbate_fragment_lengths) != ads_mobile_len:
        return (
            False,
            "adsorbate_fragment_lengths must sum to mobile adsorbate length "
            f"(sum={sum(int(x) for x in adsorbate_fragment_lengths)}, "
            f"expected={ads_mobile_len})",
        )

    mobile_start = n_slab
    ads_start = mobile_start + n_core_mobile
    fragment_index_ranges: list[list[int]] = []
    offset = 0
    for frag_idx, frag_len_raw in enumerate(adsorbate_fragment_lengths):
        frag_len = int(frag_len_raw)
        frag_global_indices = list(
            range(ads_start + offset, ads_start + offset + frag_len)
        )
        fragment_index_ranges.append(frag_global_indices)
        if frag_len <= 1:
            offset += frag_len
            continue
        fragment = atoms[frag_global_indices]
        components, _ = _find_connected_components(
            fragment,
            connectivity_factor=connectivity_factor,
            use_mic=use_mic,
        )
        n_components = len(components)
        if n_components > 1:
            return (
                False,
                "Adsorbate fragment integrity check failed: "
                f"fragment {frag_idx} (size={frag_len}) split into {n_components} "
                "components.",
            )
        offset += frag_len

    if len(fragment_index_ranges) < 2:
        return True, ""

    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    cf = normalize_connectivity_factor(connectivity_factor)
    for i, fragment_i in enumerate(fragment_index_ranges):
        for j in range(i + 1, len(fragment_index_ranges)):
            fragment_j = fragment_index_ranges[j]
            dist = pairwise_distances(
                positions[fragment_i], positions[fragment_j], atoms, use_mic=use_mic
            )
            radii_i = np.array(
                [get_covalent_radius(symbols[k]) for k in fragment_i], dtype=float
            )
            radii_j = np.array(
                [get_covalent_radius(symbols[k]) for k in fragment_j], dtype=float
            )
            sym_i = [symbols[k] for k in fragment_i]
            sym_j = [symbols[k] for k in fragment_j]
            is_h_i = np.array([s == "H" for s in sym_i])
            is_h_j = np.array([s == "H" for s in sym_j])
            h_pair = is_h_i[:, None] | is_h_j[None, :]
            # Hydrogen bonds between fragments are allowed; only covalent H-X
            # contacts (e.g. unwanted HOH formation) are rejected.
            covalent = bond_threshold_cross_matrix(radii_i, sym_i, radii_j, sym_j, cf)
            thresholds = np.where(h_pair, _H_CONTACT_THRESHOLD_A, covalent)
            violations = np.argwhere(dist <= thresholds)
            if violations.size == 0:
                continue
            local_i, local_j = (int(x) for x in violations[0])
            idx_i, idx_j = fragment_i[local_i], fragment_j[local_j]
            threshold = float(thresholds[local_i, local_j])
            detail = (
                "covalent H-contact threshold="
                if bool(h_pair[local_i, local_j])
                else "threshold="
            )
            return (
                False,
                "Adsorbate fragment integrity check failed: "
                f"fragment {i} bonded to fragment {j} "
                f"(atoms {idx_i}-{idx_j}, "
                f"distance={float(dist[local_i, local_j]):.3f} Å, "
                f"{detail}{threshold:.3f} Å).",
            )

    return True, ""
