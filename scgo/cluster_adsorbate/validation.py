"""Structure validation for core + adsorbate (connectivity and clashes)."""

from __future__ import annotations

from collections.abc import Sequence

from ase import Atoms

from scgo.initialization.geometry_helpers import (
    _find_connected_components,
    validate_cluster_structure,
)
from scgo.initialization.initialization_config import (
    CONNECTIVITY_FACTOR,
    MIN_DISTANCE_FACTOR_DEFAULT,
)


def validate_combined_cluster_structure(
    atoms: Atoms,
    *,
    min_distance_factor: float = MIN_DISTANCE_FACTOR_DEFAULT,
    connectivity_factor: float = CONNECTIVITY_FACTOR,
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
    connectivity_factor: float = CONNECTIVITY_FACTOR,
    use_mic: bool = False,
) -> tuple[bool, str]:
    """Validate that each adsorbate fragment remains internally connected.

    This check enforces non-dissociative adsorbate integrity for each fragment
    independently. External bonds (to core or other fragments) are allowed.
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
    offset = 0
    for frag_idx, frag_len_raw in enumerate(adsorbate_fragment_lengths):
        frag_len = int(frag_len_raw)
        if frag_len <= 1:
            offset += frag_len
            continue
        frag_global_indices = list(
            range(ads_start + offset, ads_start + offset + frag_len)
        )
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

    return True, ""
