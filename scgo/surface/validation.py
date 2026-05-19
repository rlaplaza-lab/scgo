"""Geometric validation for slab + adsorbate (supported cluster) deposits."""

from __future__ import annotations

import json

import numpy as np
from ase import Atoms

from scgo.database.metadata import get_metadata
from scgo.initialization.geometry_helpers import (
    _find_connected_components,
    get_covalent_radius,
    validate_cluster_structure,
)
from scgo.initialization.initialization_config import (
    CONNECTIVITY_FACTOR,
    MIN_DISTANCE_FACTOR_DEFAULT,
)
from scgo.surface.config import SurfaceSystemConfig

# Small slack below nominal slab top (numerical / structural roughness).
_BINDING_PENETRATION_TOLERANCE_A = 0.1


def validate_surface_config_slab_prefix(
    atoms: Atoms, config: SurfaceSystemConfig
) -> None:
    """Ensure ``atoms`` satisfies the slab-first ordering contract for ``config``.

    Production workflows assume indices ``0 .. len(config.slab)-1`` are exactly the
    reference slab (same chemical symbols in the same order as ``config.slab``).
    :func:`attach_slab_constraints_from_surface_config` and surface GA rely on this.

    Raises:
        ValueError: If the structure is too short or the prefix does not match.
    """
    n = len(config.slab)
    if len(atoms) < n:
        raise ValueError(
            "Slab-first ordering: combined system must have at least "
            f"{n} atoms (slab size from surface_config.slab); got len(atoms)={len(atoms)}"
        )
    ref = config.slab.get_chemical_symbols()
    got = atoms.get_chemical_symbols()[:n]
    if got != ref:
        ref_head, got_head = ref[:12], got[:12]
        raise ValueError(
            "Slab-first ordering contract violated: the first len(slab) atoms must "
            "match surface_config.slab chemical symbols in order (same count and "
            "sequence as the template slab). "
            f"Expected prefix (len {len(ref)}): {ref_head}{'...' if len(ref) > 12 else ''}; "
            f"got (len {len(got)}): {got_head}{'...' if len(got) > 12 else ''}."
        )


def validate_stored_slab_adsorbate_metadata(atoms: Atoms) -> None:
    """If GA slab metadata is present, verify the atom list still matches it.

    Older databases may only have ``n_slab_atoms`` / ``system_type`` without
    ``slab_chemical_symbols_json``; in that case only ``len(atoms) >= n_slab`` is checked.
    """
    if get_metadata(atoms, "system_type") not in {
        "surface_cluster",
        "surface_cluster_adsorbate",
    }:
        return
    n_meta = int(get_metadata(atoms, "n_slab_atoms", 0) or 0)
    if n_meta <= 0:
        return
    if len(atoms) < n_meta:
        raise ValueError(
            "Slab metadata expects at least "
            f"{n_meta} atoms (n_slab_atoms), got len(atoms)={len(atoms)}"
        )
    js = get_metadata(atoms, "slab_chemical_symbols_json", None)
    if js is None:
        return
    expected = json.loads(js)
    got = atoms.get_chemical_symbols()[:n_meta]
    if list(expected) != got:
        raise ValueError(
            "Loaded structure disagrees with stored slab_chemical_symbols_json prefix; "
            "atom ordering may have been scrambled when reading/writing the file."
        )


def validate_stored_mobile_partition_metadata(atoms: Atoms) -> None:
    """If GA core/adsorbate metadata is present, verify the mobile region matches it.

    For ``surface_cluster_adsorbate``, the mobile region follows the slab prefix.
    For ``gas_cluster_adsorbate``, the full structure is mobile.
    """
    st = get_metadata(atoms, "system_type")
    if st not in {"gas_cluster_adsorbate", "surface_cluster_adsorbate"}:
        return
    n_core = int(get_metadata(atoms, "n_core_atoms", 0) or 0)
    n_ads = int(get_metadata(atoms, "n_adsorbate_fragment_atoms", 0) or 0)
    if n_core == 0 and n_ads == 0:
        return
    n_slab = (
        int(get_metadata(atoms, "n_slab_atoms", 0) or 0)
        if st == "surface_cluster_adsorbate"
        else 0
    )
    mobile = atoms.get_chemical_symbols()[n_slab:]
    if len(mobile) < n_core + n_ads:
        raise ValueError(
            "Mobile region shorter than n_core_atoms + n_adsorbate_fragment_atoms: "
            f"len(mobile)={len(mobile)}, n_core={n_core}, n_ads={n_ads}"
        )
    core_js = get_metadata(atoms, "core_chemical_symbols_json", None)
    ads_js = get_metadata(atoms, "adsorbate_fragment_chemical_symbols_json", None)
    if core_js is None or ads_js is None:
        return
    core_exp = json.loads(core_js)
    ads_exp = json.loads(ads_js)
    if mobile[:n_core] != list(core_exp):
        raise ValueError(
            "Loaded structure disagrees with stored core_chemical_symbols_json for the "
            f"mobile region (after slab). Expected core prefix (len {n_core}): "
            f"{core_exp[:12]}{'...' if len(core_exp) > 12 else ''}; "
            f"got: {mobile[: min(12, n_core)]!r}."
        )
    if mobile[n_core : n_core + n_ads] != list(ads_exp):
        raise ValueError(
            "Loaded structure disagrees with stored "
            "adsorbate_fragment_chemical_symbols_json for the mobile region."
        )


def _slab_top_coordinate(slab: Atoms, axis: int) -> float:
    """Max Cartesian coordinate of slab atoms along ``axis`` (vacuum side)."""
    pos = slab.get_positions()
    if len(pos) == 0:
        return 0.0
    return float(np.max(pos[:, axis]))


def _mobile_atom_touches_slab(
    combined: Atoms,
    mobile_global_idx: int,
    n_slab: int,
    *,
    connectivity_factor: float,
    use_mic: bool,
) -> bool:
    """True when ``mobile_global_idx`` has a slab neighbor within the bonding threshold."""
    symbols = combined.get_chemical_symbols()
    r_i = get_covalent_radius(symbols[mobile_global_idx])
    for j in range(n_slab):
        r_j = get_covalent_radius(symbols[j])
        threshold = (r_i + r_j) * connectivity_factor
        d = float(combined.get_distance(mobile_global_idx, j, mic=use_mic))
        if d <= threshold:
            return True
    return False


def _adsorbate_subgroup_touches_slab(
    combined: Atoms,
    n_slab: int,
    subgroup_local_indices: list[int],
    *,
    connectivity_factor: float,
    use_mic: bool,
) -> bool:
    """True when any atom in a mobile subgroup is slab-connected."""
    return any(
        _mobile_atom_touches_slab(
            combined,
            n_slab + int(local_i),
            n_slab,
            connectivity_factor=connectivity_factor,
            use_mic=use_mic,
        )
        for local_i in subgroup_local_indices
    )


def validate_supported_cluster_deposit(
    combined: Atoms,
    n_slab: int,
    *,
    surface_normal_axis: int,
    use_mic: bool = False,
    min_distance_factor: float = MIN_DISTANCE_FACTOR_DEFAULT,
    connectivity_factor: float = CONNECTIVITY_FACTOR,
    penetration_tolerance: float = _BINDING_PENETRATION_TOLERANCE_A,
    allow_dissociative_adsorption: bool = False,
) -> tuple[bool, str]:
    """Validate a combined slab + supported mobile cluster (full cluster, not the fragment only).

    The slice ``combined[n_slab:]`` is the **entire** supported mobile region: nanoparticle
    core plus any chemisorbed species. (This is not the same as
    ``adsorbate_definition['adsorbate_symbols']`` alone.)

    Default (``allow_dissociative_adsorption=False``): the mobile region must form one
    connected component (when ``len(ads) > 2``) and touch the slab.

    Dissociative (``allow_dissociative_adsorption=True``): the mobile region may split into
    multiple connected subgroups, but every subgroup must touch the slab.

    Clash screening always uses
    :func:`~scgo.initialization.geometry_helpers.validate_cluster_structure` on the mobile
    slice. Optionally uses MIC for distances when ``use_mic`` is True (match
    :attr:`SurfaceSystemConfig.comparator_use_mic`).

    Args:
        combined: Full system with slab atoms first, then the supported mobile cluster.
        n_slab: Number of slab atoms (prefix length).
        surface_normal_axis: Cartesian axis index for the surface normal.
        use_mic: Pass through to distance and connectivity checks when True.
        min_distance_factor: Mobile cluster self clash scale (initialization default).
        connectivity_factor: Bonding connectivity scale (initialization default).
        penetration_tolerance: Allow mobile cluster atoms this far (Å) below the
            nominal slab top along ``surface_normal_axis``.
        allow_dissociative_adsorption: If True, allow multiple mobile subgroups that are
            each slab-connected; if False, require a single connected mobile cluster.

    Returns:
        ``(True, "")`` if valid, else ``(False, message)``.
    """
    n = len(combined)
    if n_slab < 0 or n_slab > n:
        return False, f"Invalid n_slab={n_slab} for len(combined)={n}"
    if n_slab == n:
        return False, "No adsorbate atoms in combined structure"

    ads = combined[n_slab:]
    n_ads = len(ads)
    # Internal connectivity is only enforced for 3+ mobile atoms (same as gas-phase init).
    require_single_mobile_component = n_ads > 2 and not allow_dissociative_adsorption
    ok, err = validate_cluster_structure(
        ads,
        min_distance_factor,
        connectivity_factor,
        check_clashes=True,
        check_connectivity=require_single_mobile_component,
        use_mic=use_mic,
    )
    if not ok:
        return False, f"Adsorbate validation failed: {err}"

    slab = combined[:n_slab]
    slab_top = _slab_top_coordinate(slab, surface_normal_axis)
    positions = combined.get_positions()
    ads_coords = positions[n_slab:]
    axis_coord = ads_coords[:, surface_normal_axis]
    if bool(np.any(axis_coord < slab_top - penetration_tolerance)):
        min_c = float(np.min(axis_coord))
        return (
            False,
            "Adsorbate penetrates below nominal slab top along surface normal "
            f"(min coord={min_c:.3f} Å, slab_top={slab_top:.3f} Å)",
        )

    if allow_dissociative_adsorption and n_ads >= 2:
        components, _ = _find_connected_components(
            ads, connectivity_factor, use_mic=use_mic
        )
        subgroups = list(components.values())
        for subgroup in subgroups:
            if not _adsorbate_subgroup_touches_slab(
                combined,
                n_slab,
                subgroup,
                connectivity_factor=connectivity_factor,
                use_mic=use_mic,
            ):
                return (
                    False,
                    "Dissociative adsorption requires every mobile subgroup to touch "
                    f"the slab (subgroup size {len(subgroup)} has no slab contact within "
                    f"connectivity_factor={connectivity_factor})",
                )
        return True, ""

    symbols = combined.get_chemical_symbols()
    touches = False
    min_cross = float("inf")
    for i in range(n_slab, n):
        r_i = get_covalent_radius(symbols[i])
        for j in range(n_slab):
            r_j = get_covalent_radius(symbols[j])
            threshold = (r_i + r_j) * connectivity_factor
            d = float(combined.get_distance(i, j, mic=use_mic))
            min_cross = min(min_cross, d)
            if d <= threshold:
                touches = True
                break
        if touches:
            break

    if not touches:
        return (
            False,
            "No adsorbate–slab pair within connectivity distance "
            f"(min cross-set distance={min_cross:.3f} Å, "
            f"connectivity_factor={connectivity_factor})",
        )

    return True, ""
