"""Geometric validation for slab + adsorbate (supported cluster) deposits."""

from __future__ import annotations

import json

import numpy as np
from ase import Atoms

from scgo.cluster_adsorbate.validation import validate_adsorbate_fragment_integrity
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
        raise ValueError(
            "surface_cluster structures require n_slab_atoms > 0 in metadata"
        )
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


def _classify_mobile_component(
    local_indices: list[int],
    n_core_mobile: int,
) -> str:
    """Classify a mobile connected component as ``core``, ``ads_only``, or ``mixed``."""
    has_core = any(int(i) < n_core_mobile for i in local_indices)
    has_ads = any(int(i) >= n_core_mobile for i in local_indices)
    if has_core and has_ads:
        return "mixed"
    if has_core:
        return "core"
    return "ads_only"


def _validate_mobile_connectivity_policy(
    combined: Atoms,
    n_slab: int,
    mobile: Atoms,
    *,
    n_core_mobile: int,
    connectivity_factor: float,
    use_mic: bool,
    allow_cluster_fragmentation: bool,
    allow_adsorbate_surface_detachment: bool,
    surface_normal_axis: int = 2,
) -> tuple[bool, str]:
    """Enforce mobile-region connectivity rules and per-subgroup slab contact."""
    n_ads = len(mobile)
    if n_ads < 2:
        return _check_mobile_touches_slab(
            combined,
            n_slab,
            connectivity_factor=connectivity_factor,
            use_mic=use_mic,
            surface_normal_axis=surface_normal_axis,
        )

    components, _ = _find_connected_components(
        mobile, connectivity_factor, use_mic=use_mic
    )
    subgroups = list(components.values())
    allow_split = allow_cluster_fragmentation or allow_adsorbate_surface_detachment

    if not allow_split:
        if len(subgroups) != 1:
            return (
                False,
                "Mobile region must form a single connected component "
                f"(found {len(subgroups)} components; enable "
                "allow_cluster_fragmentation and/or allow_adsorbate_surface_detachment "
                "to permit splits)",
            )
        return _check_mobile_touches_slab(
            combined,
            n_slab,
            connectivity_factor=connectivity_factor,
            use_mic=use_mic,
            surface_normal_axis=surface_normal_axis,
        )

    core_like: list[list[int]] = []
    ads_only: list[list[int]] = []
    for subgroup in subgroups:
        kind = _classify_mobile_component(subgroup, n_core_mobile)
        if kind == "ads_only":
            ads_only.append(subgroup)
        else:
            core_like.append(subgroup)

    if allow_cluster_fragmentation and not allow_adsorbate_surface_detachment:
        if ads_only:
            return (
                False,
                "Detached adsorbate-only mobile subgroups are not allowed "
                "(set allow_adsorbate_surface_detachment=True to permit adsorbates "
                "on the slab without cluster contact)",
            )
    elif (
        allow_adsorbate_surface_detachment
        and not allow_cluster_fragmentation
        and len(core_like) != 1
    ):
        return (
            False,
            "Exactly one core-connected mobile component is required when "
            f"allow_cluster_fragmentation=False (found {len(core_like)} "
            "core/mixed components)",
        )
    # both flags True: any split is allowed

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
                "Every mobile subgroup must touch the slab "
                f"(subgroup size {len(subgroup)} has no slab contact within "
                f"connectivity_factor={connectivity_factor})",
            )
    return True, ""


def _slab_surface_layer_indices(
    combined: Atoms,
    n_slab: int,
    *,
    surface_normal_axis: int,
    thickness: float = 2.5,
) -> list[int]:
    """Indices of slab atoms within ``thickness`` Å of the top surface."""
    pos = combined.get_positions()
    if n_slab <= 0 or len(pos) < n_slab:
        return list(range(n_slab))
    slab_pos = pos[:n_slab]
    top = float(np.max(slab_pos[:, surface_normal_axis]))
    mask = slab_pos[:, surface_normal_axis] >= top - thickness
    indices = [i for i in range(n_slab) if mask[i]]
    return indices if indices else list(range(n_slab))


def _check_mobile_touches_slab(
    combined: Atoms,
    n_slab: int,
    *,
    connectivity_factor: float,
    use_mic: bool,
    surface_normal_axis: int = 2,
) -> tuple[bool, str]:
    """True when at least one mobile atom is within bonding distance of the slab."""
    n = len(combined)
    symbols = combined.get_chemical_symbols()
    slab_indices = _slab_surface_layer_indices(
        combined,
        n_slab,
        surface_normal_axis=surface_normal_axis,
    )
    slab_radii = [get_covalent_radius(symbols[j]) for j in slab_indices]
    touches = False
    min_cross = float("inf")
    for i in range(n_slab, n):
        r_i = get_covalent_radius(symbols[i])
        for j_idx, j in enumerate(slab_indices):
            r_j = slab_radii[j_idx]
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


def validate_supported_cluster_deposit(
    combined: Atoms,
    n_slab: int,
    *,
    surface_normal_axis: int,
    use_mic: bool = False,
    min_distance_factor: float = MIN_DISTANCE_FACTOR_DEFAULT,
    connectivity_factor: float = CONNECTIVITY_FACTOR,
    penetration_tolerance: float = _BINDING_PENETRATION_TOLERANCE_A,
    n_core_mobile: int | None = None,
    adsorbate_fragment_lengths: list[int] | None = None,
    allow_cluster_fragmentation: bool = False,
    allow_adsorbate_surface_detachment: bool = False,
    enforce_adsorbate_subgraph_integrity: bool = True,
) -> tuple[bool, str]:
    """Validate a combined slab + supported mobile cluster (full cluster, not the fragment only).

    The slice ``combined[n_slab:]`` is the **entire** supported mobile region: nanoparticle
    core plus any chemisorbed species. (This is not the same as
    ``adsorbate_definition['adsorbate_symbols']`` alone.)

    **Default** (both relaxation flags False): the mobile region must form one
    connected component (when ``len(mobile) > 2``) and touch the slab.

    **``allow_cluster_fragmentation``**: multiple core/mixed mobile subgroups are allowed;
    detached adsorbate-only subgroups are still rejected unless
    ``allow_adsorbate_surface_detachment`` is also True.

    **``allow_adsorbate_surface_detachment``**: exactly one core/mixed subgroup is required
    when cluster fragmentation is disabled; additional adsorbate-only subgroups on the slab
    are allowed.

    **Both flags True**: any mobile split is allowed if every subgroup touches the slab.

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
        n_core_mobile: Atoms in the mobile slice belonging to the cluster core (prefix).
            When ``None``, all mobile atoms are treated as core (``surface_cluster``).
        adsorbate_fragment_lengths: Optional ordered lengths for adsorbate fragments
            within the mobile adsorbate suffix.
        allow_cluster_fragmentation: Allow multiple disconnected core-bearing subgroups.
        allow_adsorbate_surface_detachment: Allow adsorbate-only subgroups on the slab.
        enforce_adsorbate_subgraph_integrity: Require each adsorbate fragment to
            remain internally connected.

    Returns:
        ``(True, "")`` if valid, else ``(False, message)``.
    """
    n = len(combined)
    if n_slab < 0 or n_slab > n:
        return False, f"Invalid n_slab={n_slab} for len(combined)={n}"
    if n_slab == n:
        return False, "No adsorbate atoms in combined structure"

    mobile = combined[n_slab:]
    n_ads = len(mobile)
    n_core_eff = int(n_ads if n_core_mobile is None else n_core_mobile)
    if n_core_eff < 0 or n_core_eff > n_ads:
        return False, f"Invalid n_core_mobile={n_core_mobile} for mobile len={n_ads}"

    allow_split = allow_cluster_fragmentation or allow_adsorbate_surface_detachment
    require_single_mobile_component = n_ads > 2 and not allow_split
    ok, err = validate_cluster_structure(
        mobile,
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

    if enforce_adsorbate_subgraph_integrity and adsorbate_fragment_lengths:
        ok, msg = validate_adsorbate_fragment_integrity(
            combined,
            n_slab=n_slab,
            n_core_mobile=n_core_eff,
            adsorbate_fragment_lengths=adsorbate_fragment_lengths,
            connectivity_factor=connectivity_factor,
            use_mic=use_mic,
        )
        if not ok:
            return False, msg

    return _validate_mobile_connectivity_policy(
        combined,
        n_slab,
        mobile,
        n_core_mobile=n_core_eff,
        connectivity_factor=connectivity_factor,
        use_mic=use_mic,
        allow_cluster_fragmentation=allow_cluster_fragmentation,
        allow_adsorbate_surface_detachment=allow_adsorbate_surface_detachment,
        surface_normal_axis=surface_normal_axis,
    )
