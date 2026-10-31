"""Move one intact, core-bound adsorbate fragment onto a nearby slab site."""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from ase_ga.offspring_creator import OffspringCreator
from ase_ga.utilities import atoms_too_close_two_sets
from numpy.random import Generator
from scipy.spatial import ConvexHull, QhullError

from scgo.ase_ga_patches.mutations import _ensure_rng
from scgo.ase_ga_patches.mutations._common import _resolve_op_connectivity_factor
from scgo.ase_ga_patches.mutations._finalize import _finalize_mutant
from scgo.cluster_adsorbate.config import (
    ClusterAdsorbateConfig,
    resolve_cluster_adsorbate_config,
)
from scgo.cluster_adsorbate.helpers import (
    parse_positive_fragment_lengths,
    resolve_fragment_anchor_and_bond_axis,
)
from scgo.cluster_adsorbate.placement import place_fragment_on_cluster
from scgo.cluster_adsorbate.sites import (
    SiteType,
    SurfaceSiteCandidate,
    count_site_candidates,
)
from scgo.cluster_adsorbate.validation import validate_adsorbate_fragment_integrity
from scgo.initialization.atomic_radii import get_covalent_radius
from scgo.initialization.geometry_helpers import _find_connected_components
from scgo.surface.deposition import (
    _outward_slab_site_candidates,
    _slab_surface_layer,
)
from scgo.surface.validation import _adsorbate_subgroup_touches_slab
from scgo.system_types import (
    AdsorbateDefinition,
    AdsorbateFragmentInput,
    SystemType,
    get_system_policy,
    resolve_adsorbate_fragments,
)
from scgo.system_types.connectivity_factor import (
    ConnectivityFactorInput,
    NormalizedConnectivityFactor,
    normalize_connectivity_factor,
    pair_bond_threshold,
)

# In-plane distance to the nearest core atom, in bond-threshold units.
# The lower edge clears the fragment so it is no longer bonded to the core.
_NEIGHBORHOOD_LENGTHS = 3.0
_DETACH_CLEARANCE_A = 0.05
_PLACEMENT_TRIES = 8


def _blank_sites() -> dict[SiteType, list[SurfaceSiteCandidate]]:
    return {"vertex": [], "edge": [], "facet": []}


def _mic_delta(origin: np.ndarray, target: np.ndarray, atoms: Atoms) -> np.ndarray:
    delta = np.asarray(target, dtype=float) - np.asarray(origin, dtype=float)
    if not np.any(atoms.pbc):
        return delta
    mic, _dist = find_mic(delta.reshape(1, 3), atoms.cell, atoms.pbc)
    return np.asarray(mic[0], dtype=float)


def _in_plane_distance(delta: np.ndarray, axis: int) -> float:
    copied = np.array(delta, dtype=float, copy=True)
    copied[int(axis)] = 0.0
    return float(np.linalg.norm(copied))


def _drop_normal(points: np.ndarray, axis: int) -> np.ndarray:
    keep = [index for index in range(3) if index != int(axis)]
    return np.asarray(points, dtype=float)[..., keep]


def _max_pair_threshold(
    fragment_symbols: list[str],
    core_symbols: list[str],
    connectivity_factor: ConnectivityFactorInput | NormalizedConnectivityFactor,
) -> float:
    spec = normalize_connectivity_factor(connectivity_factor)
    longest = 0.0
    for frag_symbol in set(fragment_symbols):
        frag_radius = get_covalent_radius(frag_symbol)
        for core_symbol in set(core_symbols):
            longest = max(
                longest,
                pair_bond_threshold(
                    frag_radius,
                    get_covalent_radius(core_symbol),
                    frag_symbol,
                    core_symbol,
                    spec,
                ),
            )
    return longest


def _fragment_extent(fragment: Atoms, anchor_index: int) -> float:
    positions = fragment.get_positions()
    if len(positions) <= 1:
        return 0.0
    anchor = positions[int(anchor_index)]
    return float(np.max(np.linalg.norm(positions - anchor, axis=1)))


def _hull_equations(core_xy: np.ndarray) -> np.ndarray | None:
    """Unit-normal hull equations, or ``None`` when the core is not a polygon."""
    if len(core_xy) < 3:
        return None
    try:
        hull = ConvexHull(core_xy)
    except QhullError:
        return None
    return np.asarray(hull.equations, dtype=float)


def _inside_expanded_hull(
    point_xy: np.ndarray,
    equations: np.ndarray,
    margin: float,
) -> bool:
    signed = equations[:, :2] @ np.asarray(point_xy, dtype=float) + equations[:, 2]
    return bool(np.all(signed <= margin + 1e-8))


def _filter_spillover_sites(
    sites: dict[SiteType, list[SurfaceSiteCandidate]],
    *,
    core: Atoms,
    slab: Atoms,
    fragment: Atoms,
    anchor_index: int,
    axis: int,
    connectivity_factor: ConnectivityFactorInput | NormalizedConnectivityFactor,
) -> dict[SiteType, list[SurfaceSiteCandidate]]:
    """Keep slab sites beside the cluster, not under it and not across the cell."""
    kept = _blank_sites()
    if len(core) == 0 or len(slab) == 0:
        return kept
    length = _max_pair_threshold(
        fragment.get_chemical_symbols(),
        core.get_chemical_symbols(),
        connectivity_factor,
    )
    if length <= 0.0:
        return kept
    minimum = length + _fragment_extent(fragment, anchor_index) + _DETACH_CLEARANCE_A
    maximum = _NEIGHBORHOOD_LENGTHS * length
    if minimum >= maximum:
        return kept

    core_positions = core.get_positions()
    core_radius = max(get_covalent_radius(sym) for sym in core.get_chemical_symbols())
    com = np.mean(core_positions, axis=0)
    equations = _hull_equations(_drop_normal(core_positions, axis))

    for site_type, candidates in sites.items():
        accepted: list[SurfaceSiteCandidate] = []
        for candidate in candidates:
            nearest = min(
                _in_plane_distance(
                    _mic_delta(core_position, candidate.anchor, slab), axis
                )
                for core_position in core_positions
            )
            if not (minimum < nearest <= maximum):
                continue
            if equations is not None:
                site_xy = _drop_normal(
                    com + _mic_delta(com, candidate.anchor, slab), axis
                )
                if _inside_expanded_hull(site_xy, equations, core_radius):
                    continue
            accepted.append(candidate)
        kept[site_type] = accepted
    return kept


def _component_roots(
    mobile: Atoms,
    connectivity_factor: ConnectivityFactorInput | NormalizedConnectivityFactor,
    use_mic: bool,
) -> dict[int, int]:
    components = _find_connected_components(mobile, connectivity_factor, use_mic)
    roots: dict[int, int] = {}
    for root, members in components.items():
        for index in members:
            roots[int(index)] = int(root)
    return roots


def _placement_config(
    base: ClusterAdsorbateConfig,
    *,
    relaxed: bool,
) -> ClusterAdsorbateConfig:
    """Slab-contact heights. Connectivity to the clash target stays off."""
    height_min = max(base.height_min, 1.35)
    height_max = max(base.height_max, height_min)
    attempts = base.max_placement_attempts
    blmin_ratio = base.blmin_ratio
    min_distance = base.structure_min_distance_factor
    if relaxed:
        height_min *= 0.9
        height_max *= 1.15
        attempts = min(max(attempts * 3, 80), 400)
        blmin_ratio = min(blmin_ratio, 0.6)
        min_distance = min(min_distance, 0.35)
    return ClusterAdsorbateConfig(
        max_placement_attempts=attempts,
        random_spin_about_normal=True,
        height_min=height_min,
        height_max=height_max,
        blmin_ratio=blmin_ratio,
        structure_min_distance_factor=min_distance,
        validate_combined_structure=base.validate_combined_structure,
        structure_connectivity_factor=base.structure_connectivity_factor,
        structure_check_clashes=base.structure_check_clashes,
        structure_check_connectivity=False,
    )


class FragmentDetachMutation(OffspringCreator):
    """Move one intact, core-bound adsorbate onto a nearby slab site.

    The child is an adsorbate-only subgroup. ``allow_disconnected_mobile``
    keeps the shared finalizer from dropping that child.
    """

    def __init__(
        self,
        blmin: dict,
        n_top: int,
        system_type: SystemType,
        adsorbate_definition: AdsorbateDefinition,
        fragment_templates: AdsorbateFragmentInput | None = None,
        cluster_adsorbate_config: ClusterAdsorbateConfig | None = None,
        *,
        rng: Generator | None = None,
        verbose: bool = False,
        surface_normal_axis: int = 2,
    ) -> None:
        rng = _ensure_rng(rng)
        OffspringCreator.__init__(self, verbose, rng=rng)
        self.blmin = blmin
        self.n_top = n_top
        self.system_type = system_type
        self._policy = get_system_policy(system_type)
        self.adsorbate_definition = adsorbate_definition
        self.fragment_templates = fragment_templates
        self.cluster_adsorbate_config = cluster_adsorbate_config
        self.surface_normal_axis = surface_normal_axis
        self.descriptor = "FragmentDetachMutation"
        self.min_inputs = 1
        self.allow_disconnected_mobile = True

    def get_new_individual(self, parents):
        parent = parents[0]
        mutant = self.mutate(parent)
        return _finalize_mutant(self, parent, mutant, "mutation: fragment_detach")

    def _fragment_template_for_tag(
        self, mobile: Atoms, tag: int, frag_index: int
    ) -> Atoms:
        if self.fragment_templates is not None:
            fragments = resolve_adsorbate_fragments(
                self.fragment_templates,
                self.adsorbate_definition,
                context="FragmentDetachMutation",
            )
            if 0 <= frag_index < len(fragments):
                return fragments[frag_index].copy()
        mask = mobile.get_tags() == tag
        return mobile[mask].copy()

    def _bonded_tags(
        self,
        mobile: Atoms,
        tags: np.ndarray,
        ads_tags: list[int],
        connectivity_factor: ConnectivityFactorInput | NormalizedConnectivityFactor,
        use_mic: bool,
    ) -> list[int]:
        roots = _component_roots(mobile, connectivity_factor, use_mic)
        core_roots = {roots[int(index)] for index in np.flatnonzero(tags == 0)}
        bonded: list[int] = []
        for tag in ads_tags:
            fragment_roots = {
                roots[int(index)] for index in np.flatnonzero(tags == tag)
            }
            if fragment_roots & core_roots:
                bonded.append(tag)
        return bonded

    def _accepts(
        self,
        slab: Atoms,
        new_mobile: Atoms,
        target_tag: int,
        connectivity_factor: ConnectivityFactorInput | NormalizedConnectivityFactor,
        use_mic: bool,
    ) -> bool:
        if len(slab) > 0 and atoms_too_close_two_sets(new_mobile, slab, self.blmin):
            return False
        n_slab = len(slab)
        combined = slab + new_mobile
        n_core = int(self.adsorbate_definition.n_core)
        ok, _msg = validate_adsorbate_fragment_integrity(
            combined,
            n_slab=n_slab,
            n_core_mobile=n_core,
            adsorbate_fragment_lengths=list(
                self.adsorbate_definition.adsorbate_fragment_lengths
            ),
            connectivity_factor=connectivity_factor,
            use_mic=use_mic,
        )
        if not ok:
            return False
        tags = new_mobile.get_tags()
        roots = _component_roots(new_mobile, connectivity_factor, use_mic)
        core_roots = {roots[int(index)] for index in np.flatnonzero(tags == 0)}
        fragment_indices = [int(index) for index in np.flatnonzero(tags == target_tag)]
        fragment_roots = {roots[index] for index in fragment_indices}
        if len(core_roots) != 1 or len(fragment_roots) != 1:
            return False
        if fragment_roots & core_roots:
            return False
        return _adsorbate_subgroup_touches_slab(
            combined,
            n_slab,
            fragment_indices,
            connectivity_factor=connectivity_factor,
            use_mic=use_mic,
            surface_normal_axis=self.surface_normal_axis,
        )

    def mutate(self, atoms: Atoms) -> Atoms | None:
        n_top = int(self.n_top)
        if n_top <= 0 or len(atoms) <= n_top:
            return None
        if int(self.adsorbate_definition.n_core) <= 0:
            return None
        slab = atoms[: len(atoms) - n_top]
        mobile = atoms[-n_top:].copy()
        tags = mobile.get_tags()
        ads_tags = sorted({int(tag) for tag in tags if int(tag) > 0})
        if not ads_tags or not np.any(tags == 0):
            return None

        lengths = parse_positive_fragment_lengths(
            self.adsorbate_definition.adsorbate_fragment_lengths
        )
        connectivity_factor = _resolve_op_connectivity_factor(self)
        use_mic = bool(self._policy.uses_surface)
        bonded = self._bonded_tags(mobile, tags, ads_tags, connectivity_factor, use_mic)
        if not bonded:
            return None
        target_tag = int(self.rng.choice(bonded))
        frag_index = target_tag - 1
        if not (0 <= frag_index < len(lengths)):
            return None
        ads_mask = tags == target_tag

        fragment_tmpl = self._fragment_template_for_tag(mobile, target_tag, frag_index)
        anchor, bond_axis = resolve_fragment_anchor_and_bond_axis(
            self.adsorbate_definition
        )
        axis = int(self.surface_normal_axis)
        site_layer = _slab_surface_layer(slab, axis)
        if len(site_layer) == 0:
            return None
        sites = _filter_spillover_sites(
            _outward_slab_site_candidates(site_layer, axis),
            core=mobile[tags == 0],
            slab=slab,
            fragment=fragment_tmpl,
            anchor_index=anchor,
            axis=axis,
            connectivity_factor=connectivity_factor,
        )
        if count_site_candidates(sites) == 0:
            return None

        ca = resolve_cluster_adsorbate_config(self.cluster_adsorbate_config)
        clash_mobile = mobile[~ads_mask]
        for relaxed in (False, True):
            cfg = _placement_config(ca, relaxed=relaxed)
            for _ in range(_PLACEMENT_TRIES):
                placed = place_fragment_on_cluster(
                    site_layer,
                    fragment_tmpl,
                    self.rng,
                    cfg,
                    anchor_index=anchor,
                    bond_axis=bond_axis,
                    site_core=site_layer,
                    clash_atoms=clash_mobile,
                    site_candidates=sites,
                    uses_surface=True,
                )
                if placed is None:
                    break
                new_mobile = mobile.copy()
                new_mobile.positions[ads_mask] = placed.get_positions()
                if self._accepts(
                    slab,
                    new_mobile,
                    target_tag,
                    connectivity_factor,
                    use_mic,
                ):
                    return slab + new_mobile
        return None
