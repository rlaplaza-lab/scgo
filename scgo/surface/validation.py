"""Geometric validation for slab + adsorbate (supported cluster) deposits."""

from __future__ import annotations

import itertools
import json
from collections.abc import Iterable

import numpy as np
from ase import Atoms
from scipy.spatial import cKDTree

from scgo.cluster_adsorbate.validation import validate_adsorbate_fragment_integrity
from scgo.exceptions import (
    SCGOValidationError,
)
from scgo.initialization.atomic_radii import get_covalent_radius
from scgo.initialization.geometry_helpers import validate_cluster_structure
from scgo.initialization.initialization_config import (
    CONNECTIVITY_FACTOR,
    MIN_DISTANCE_FACTOR_DEFAULT,
)
from scgo.metadata.atoms import get_tag
from scgo.surface.config import SurfaceSystemConfig
from scgo.system_types import validate_connectivity_policy
from scgo.system_types.connectivity_factor import (
    ConnectivityFactorInput,
    NormalizedConnectivityFactor,
    bond_thresholds_for_cross_pairs,
    format_connectivity_factor,
    normalize_connectivity_factor,
)
from scgo.utils.combine_atoms import (
    slab_surface_extreme,
    top_layer_indices,
)

# Small slack below nominal slab top (numerical / structural roughness).
_BINDING_PENETRATION_TOLERANCE_A = 0.1
# When the template's mobile layers are not covalently bound to the frozen
# prefix (layered crystals), candidates may expand that template gap by this
# factor before the sheet counts as desorbed. Metallic slabs never take this
# path: their template layers already satisfy covalent contact.
_STACKING_GAP_RUMPLE_FACTOR = 1.5


def _symbols_tag_list(value: object) -> list[str]:
    """Normalize a chemical-symbols tag (JSON string or already-decoded list)."""
    if isinstance(value, str):
        value = json.loads(value)
    return [str(x) for x in value]  # type: ignore[arg-type]


def validate_surface_config_slab_prefix(
    atoms: Atoms, config: SurfaceSystemConfig
) -> None:
    """Ensure ``atoms`` satisfies the slab-first ordering contract for ``config``.

    Production workflows assume indices ``0 .. len(config.slab)-1`` are exactly the
    reference slab (same chemical symbols in the same order as ``config.slab``).
    :func:`~scgo.surface.constraints.attach_slab_constraints_from_surface_config` and surface GA rely on this.

    Raises:
        SCGOValidationError: If the structure is too short or the prefix does not
            match.
    """
    n = len(config.slab)
    if len(atoms) < n:
        raise SCGOValidationError(
            "Slab-first ordering: combined system must have at least "
            f"{n} atoms (slab size from surface_config.slab); got len(atoms)={len(atoms)}"
        )
    ref = config.slab.get_chemical_symbols()
    got = atoms.get_chemical_symbols()[:n]
    if got != ref:
        ref_head, got_head = ref[:12], got[:12]
        raise SCGOValidationError(
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
    if get_tag(atoms, "system_type") not in {
        "surface_cluster",
        "surface_cluster_adsorbate",
        "surface",
        "surface_adsorbate",
    }:
        return
    n_meta = int(get_tag(atoms, "n_slab_atoms", 0) or 0)
    if n_meta <= 0:
        raise SCGOValidationError(
            "surface_* structures require n_slab_atoms > 0 in metadata"
        )
    if len(atoms) < n_meta:
        raise SCGOValidationError(
            "Slab metadata expects at least "
            f"{n_meta} atoms (n_slab_atoms), got len(atoms)={len(atoms)}"
        )
    js = get_tag(atoms, "slab_chemical_symbols_json", None)
    if js is None:
        return
    expected = _symbols_tag_list(js)
    got = atoms.get_chemical_symbols()[:n_meta]
    if list(expected) != got:
        raise SCGOValidationError(
            "Loaded structure disagrees with stored slab_chemical_symbols_json prefix; "
            "atom ordering may have been scrambled when reading/writing the file."
        )


def validate_stored_mobile_partition_metadata(atoms: Atoms) -> None:
    """If GA core/adsorbate metadata is present, verify the mobile region matches it.

    For ``surface_cluster_adsorbate``, the mobile region follows the slab prefix.
    For ``gas_cluster_adsorbate``, the full structure is mobile.
    """
    st = get_tag(atoms, "system_type")
    if st not in {
        "gas_cluster_adsorbate",
        "surface_cluster_adsorbate",
        "surface_adsorbate",
    }:
        return
    n_core = int(get_tag(atoms, "n_core_atoms", 0) or 0)
    n_ads = int(get_tag(atoms, "n_adsorbate_fragment_atoms", 0) or 0)
    if n_core == 0 and n_ads == 0:
        return
    n_slab = (
        int(get_tag(atoms, "n_slab_atoms", 0) or 0)
        if st in {"surface_cluster_adsorbate", "surface_adsorbate"}
        else 0
    )
    mobile = atoms.get_chemical_symbols()[n_slab:]
    if len(mobile) < n_core + n_ads:
        raise SCGOValidationError(
            "Mobile region shorter than n_core_atoms + n_adsorbate_fragment_atoms: "
            f"len(mobile)={len(mobile)}, n_core={n_core}, n_ads={n_ads}"
        )
    core_js = get_tag(atoms, "core_chemical_symbols_json", None)
    ads_js = get_tag(atoms, "adsorbate_fragment_chemical_symbols_json", None)
    if core_js is None or ads_js is None:
        return
    core_exp = _symbols_tag_list(core_js)
    ads_exp = _symbols_tag_list(ads_js)
    if mobile[:n_core] != list(core_exp):
        raise SCGOValidationError(
            "Loaded structure disagrees with stored core_chemical_symbols_json for the "
            f"mobile region (after slab). Expected core prefix (len {n_core}): "
            f"{core_exp[:12]}{'...' if len(core_exp) > 12 else ''}; "
            f"got: {mobile[: min(12, n_core)]!r}."
        )
    if mobile[n_core : n_core + n_ads] != list(ads_exp):
        raise SCGOValidationError(
            "Loaded structure disagrees with stored "
            "adsorbate_fragment_chemical_symbols_json for the mobile region."
        )


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
    return top_layer_indices(pos[:n_slab], surface_normal_axis, thickness=thickness)


def _mobile_indices_touch_slab(
    combined: Atoms,
    n_slab: int,
    mobile_global_indices: Iterable[int],
    *,
    connectivity_factor: ConnectivityFactorInput | NormalizedConnectivityFactor,
    use_mic: bool,
    surface_normal_axis: int,
    stacking_cutoff_a: float | None = None,
) -> tuple[bool, float]:
    """Whether any mobile atom contacts the slab top layer.

    Default contact is covalent (sum of radii × ``connectivity_factor``).
    When ``stacking_cutoff_a`` is set, a nearest-neighbor separation within
    that cutoff also counts — used for layered crystals whose template
    interlayer is itself non-covalent.

    Only slab atoms within 2.5 Å of the top surface count: a mobile atom cannot
    bond a buried slab atom without also bonding a top-layer one.

    Returns:
        ``(touches, min_cross_distance)``. The distance is the smallest
        mobile/slab-surface-layer separation, or ``inf`` when there is no pair.
    """
    symbols = combined.get_chemical_symbols()
    slab_indices = _slab_surface_layer_indices(
        combined,
        n_slab,
        surface_normal_axis=surface_normal_axis,
    )
    mobile_indices = [int(i) for i in mobile_global_indices]
    if not slab_indices or not mobile_indices:
        return False, float("inf")

    positions = combined.get_positions()
    mob_pos = positions[mobile_indices]
    slab_pos = positions[slab_indices]
    r_mobile = np.array(
        [get_covalent_radius(symbols[i]) for i in mobile_indices], dtype=float
    )
    r_slab = np.array(
        [get_covalent_radius(symbols[j]) for j in slab_indices], dtype=float
    )
    sym_mobile = [symbols[i] for i in mobile_indices]
    sym_slab = [symbols[j] for j in slab_indices]
    cf = normalize_connectivity_factor(connectivity_factor)
    stack_cut = (
        float(stacking_cutoff_a)
        if stacking_cutoff_a is not None and float(stacking_cutoff_a) > 0.0
        else None
    )

    # Use a cKDTree over the (restricted) top surface layer instead of the full
    # O(M*N) pairwise-distance matrix. When MIC is requested we expand the slab
    # layer by its periodic images so the nearest-neighbor query is exact (it
    # reproduces the ASE minimum-image distance used by pairwise_distances).
    cell = np.asarray(combined.cell)
    pbc = np.asarray(combined.pbc)
    if use_mic and bool(pbc.any()):
        shifts = itertools.product(
            *(range(-1, 2) if pbc[k] else range(1) for k in range(3))
        )
        exp_pos: list[np.ndarray] = []
        exp_j: list[int] = []
        for shift in shifts:
            disp = cell @ np.asarray(shift, dtype=float)
            for jj, p in enumerate(slab_pos):
                exp_pos.append(p + disp)
                exp_j.append(int(jj))
        slab_tree_pos = np.asarray(exp_pos, dtype=float)
        slab_tree_j = np.asarray(exp_j, dtype=int)
    else:
        slab_tree_pos = slab_pos
        slab_tree_j = np.arange(len(slab_pos))

    tree = cKDTree(slab_tree_pos)
    dists, nn = tree.query(mob_pos)
    dists = np.atleast_1d(np.asarray(dists, dtype=float))
    nn = np.atleast_1d(np.asarray(nn, dtype=int))
    min_dist = float(np.min(dists)) if len(dists) else float("inf")
    nn_j = slab_tree_j[nn.astype(int)]
    i_local = np.arange(len(mobile_indices), dtype=int)
    thresholds = bond_thresholds_for_cross_pairs(
        r_mobile,
        sym_mobile,
        i_local,
        r_slab,
        sym_slab,
        nn_j.astype(int),
        cf,
    )
    touched = bool(np.any(dists <= thresholds)) if thresholds.size else False
    if not touched and stack_cut is not None and min_dist <= stack_cut:
        touched = True
    return touched, min_dist


def layer_stacking_cutoff_from_template(
    slab: Atoms,
    n_fixed: int,
    *,
    surface_normal_axis: int,
    connectivity_factor: ConnectivityFactorInput | NormalizedConnectivityFactor,
    use_mic: bool,
) -> float | None:
    """Stacking cutoff for slab-as-search-target contact, or ``None``.

    If the template mobile layers already covalently touch the frozen prefix
    (metals), candidates keep the covalent gate and this returns ``None``.
    If they do not (layered crystals), the cutoff is the template's own
    interlayer times ``_STACKING_GAP_RUMPLE_FACTOR``.
    """
    if n_fixed <= 0 or n_fixed >= len(slab):
        return None
    touches, min_d = _mobile_indices_touch_slab(
        slab,
        n_fixed,
        range(n_fixed, len(slab)),
        connectivity_factor=connectivity_factor,
        use_mic=use_mic,
        surface_normal_axis=surface_normal_axis,
    )
    if touches or not np.isfinite(min_d) or min_d <= 0.0:
        return None
    return float(min_d) * _STACKING_GAP_RUMPLE_FACTOR


def _adsorbate_subgroup_touches_slab(
    combined: Atoms,
    n_slab: int,
    subgroup_local_indices: list[int],
    *,
    connectivity_factor: ConnectivityFactorInput | NormalizedConnectivityFactor,
    use_mic: bool,
    surface_normal_axis: int,
    stacking_cutoff_a: float | None = None,
) -> bool:
    """True when any atom in a mobile subgroup touches the slab surface layer."""
    touches, _min_cross = _mobile_indices_touch_slab(
        combined,
        n_slab,
        (n_slab + int(local_i) for local_i in subgroup_local_indices),
        connectivity_factor=connectivity_factor,
        use_mic=use_mic,
        surface_normal_axis=surface_normal_axis,
        stacking_cutoff_a=stacking_cutoff_a,
    )
    return touches


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


def _check_mobile_touches_slab(
    combined: Atoms,
    n_slab: int,
    *,
    connectivity_factor: ConnectivityFactorInput | NormalizedConnectivityFactor,
    use_mic: bool,
    surface_normal_axis: int = 2,
    stacking_cutoff_a: float | None = None,
) -> tuple[bool, str]:
    """True when a mobile atom is within bonding distance of the slab surface layer."""
    cf = normalize_connectivity_factor(connectivity_factor)
    touches, min_cross = _mobile_indices_touch_slab(
        combined,
        n_slab,
        range(n_slab, len(combined)),
        connectivity_factor=cf,
        use_mic=use_mic,
        surface_normal_axis=surface_normal_axis,
        stacking_cutoff_a=stacking_cutoff_a,
    )
    if not touches:
        extra = (
            f", stacking_cutoff={float(stacking_cutoff_a):.3f} Å"
            if stacking_cutoff_a is not None
            else ""
        )
        return (
            False,
            "No mobile-slab contact within connectivity distance "
            f"(min mobile-to-surface-layer distance={min_cross:.3f} Å, "
            f"connectivity_factor={format_connectivity_factor(cf)}{extra})",
        )
    return True, ""


def validate_supported_cluster_deposit(
    combined: Atoms,
    n_slab: int,
    *,
    surface_normal_axis: int,
    use_mic: bool = False,
    min_distance_factor: float = MIN_DISTANCE_FACTOR_DEFAULT,
    connectivity_factor: ConnectivityFactorInput
    | NormalizedConnectivityFactor = CONNECTIVITY_FACTOR,
    binding_penetration_tolerance_a: float = _BINDING_PENETRATION_TOLERANCE_A,
    n_core_mobile: int | None = None,
    adsorbate_fragment_lengths: list[int] | None = None,
    allow_cluster_fragmentation: bool = False,
    allow_adsorbate_surface_detachment: bool = False,
    enforce_adsorbate_subgraph_integrity: bool = True,
    slab_stacking_cutoff_a: float | None = None,
) -> tuple[bool, str]:
    """Validate a combined slab + supported mobile cluster (full cluster, not the fragment only).

    The slice ``combined[n_slab:]`` is the **entire** supported mobile region: nanoparticle
    core plus any chemisorbed species. (This is not the same as
    ``adsorbate_definition['adsorbate_symbols']`` alone.)

    **Default** (both relaxation flags False): the mobile region must form one
    connected component (when ``len(mobile) >= 2``) and touch the slab.

    ``allow_cluster_fragmentation``: multiple core/mixed mobile subgroups are allowed;
    detached adsorbate-only subgroups are still rejected unless
    ``allow_adsorbate_surface_detachment`` is also True.

    ``allow_adsorbate_surface_detachment``: exactly one core/mixed subgroup is required
    when cluster fragmentation is disabled; additional adsorbate-only subgroups on the slab
    are allowed.

    **Both flags True**: any mobile split is allowed if every subgroup touches the slab.

    Clash screening always uses
    :func:`~scgo.initialization.validate_cluster_structure` on the
    **mobile slice only** — it does not screen mobile-atom vs slab-atom clashes
    (slab<->mobile contact is gated earlier, at placement, by
    ``atoms_too_close_two_sets`` with the blmin table). Optionally uses MIC for
    distances when ``use_mic`` is True (match
    :attr:`~scgo.surface.config.SurfaceSystemConfig.comparator_use_mic`).

    ``binding_penetration_tolerance_a`` guards against mobile atoms straying below the
    **nominal** slab top along the surface normal (a bookkeeping check on the
    slab-extreme plane), NOT a contact-distance check between the mobile region and the
    slab surface.

    Args:
        combined: Full system with slab atoms first, then the supported mobile cluster.
        n_slab: Number of slab atoms (prefix length).
        surface_normal_axis: Cartesian axis index for the surface normal.
        use_mic: Pass through to distance and connectivity checks when True.
        min_distance_factor: Mobile cluster self clash scale (initialization default).
        connectivity_factor: Bonding connectivity scale (initialization default).
        binding_penetration_tolerance_a: Allow mobile cluster atoms this far (Å) below
            the nominal slab top along ``surface_normal_axis``.
        n_core_mobile: Atoms in the mobile slice belonging to the cluster core (prefix).
            When ``None``, all mobile atoms are treated as core (``surface_cluster``).
        adsorbate_fragment_lengths: Optional ordered lengths for adsorbate fragments
            within the mobile adsorbate suffix.
        allow_cluster_fragmentation: Allow multiple disconnected core-bearing subgroups.
        allow_adsorbate_surface_detachment: Allow adsorbate-only subgroups on the slab.
        enforce_adsorbate_subgraph_integrity: Require each adsorbate fragment to
            remain internally connected.
        slab_stacking_cutoff_a: Optional nearest-neighbor cutoff (Å) that also
            counts as slab contact. Set from
            :func:`layer_stacking_cutoff_from_template` when the template
            itself is a van der Waals stack rather than a covalent multilayer.

    Returns:
        ``(True, "")`` if valid, else ``(False, message)``.
    """
    n = len(combined)
    if n_slab < 0 or n_slab > n:
        return False, f"Invalid n_slab={n_slab} for len(combined)={n}"
    if n_slab == n:
        return False, "No mobile atoms in combined structure"

    mobile = combined[n_slab:]
    n_ads = len(mobile)
    n_core_eff = int(n_ads if n_core_mobile is None else n_core_mobile)
    if n_core_eff < 0 or n_core_eff > n_ads:
        return False, f"Invalid n_core_mobile={n_core_mobile} for mobile len={n_ads}"

    ok, err = validate_cluster_structure(
        mobile,
        min_distance_factor,
        connectivity_factor,
        check_clashes=True,
        check_connectivity=False,
        use_mic=use_mic,
    )
    if not ok:
        return False, f"Mobile-region validation failed: {err}"

    slab = combined[:n_slab]
    slab_top = slab_surface_extreme(slab, surface_normal_axis)
    positions = combined.get_positions()
    ads_coords = positions[n_slab:]
    axis_coord = ads_coords[:, surface_normal_axis]
    if np.any(axis_coord < slab_top - binding_penetration_tolerance_a):
        min_c = float(np.min(axis_coord))
        return (
            False,
            "Mobile region penetrates below nominal slab top along surface normal "
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

    ok, msg = validate_connectivity_policy(
        combined,
        uses_surface=True,
        n_slab=n_slab,
        n_core_mobile=n_core_eff,
        connectivity_factor=connectivity_factor,
        use_mic=use_mic,
        allow_cluster_fragmentation=allow_cluster_fragmentation,
        allow_adsorbate_surface_detachment=allow_adsorbate_surface_detachment,
        surface_normal_axis=surface_normal_axis,
        slab_stacking_cutoff_a=slab_stacking_cutoff_a,
    )
    if not ok:
        return False, f"Mobile-region validation failed: {msg}"
    return True, ""
