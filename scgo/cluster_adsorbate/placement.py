"""Place molecular fragments near the surface of a gas-phase cluster."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from ase import Atoms
from ase_ga.utilities import atoms_too_close_two_sets, closest_distances_generator
from numpy.random import Generator
from scipy.spatial import ConvexHull, QhullError

from scgo.cluster_adsorbate.combine import combine_core_adsorbate
from scgo.cluster_adsorbate.config import ClusterAdsorbateConfig, ClusterOHConfig
from scgo.cluster_adsorbate.geometry import (
    outermost_point_along_normal,
    random_rotation_matrix,
    random_spin_about_normal,
    random_unit_vector,
    rotation_matrix_a_to_b,
)
from scgo.cluster_adsorbate.validation import validate_combined_cluster_structure
from scgo.utils.logging import get_logger

logger = get_logger(__name__)


SiteType = Literal["vertex", "edge", "facet"]


@dataclass(frozen=True)
class SurfaceSiteCandidate:
    site_type: SiteType
    anchor: np.ndarray
    normal: np.ndarray


def _safe_normalize(v: np.ndarray) -> np.ndarray:
    vn = float(np.linalg.norm(v))
    if vn < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return v / vn


def _compute_surface_site_candidates(
    core: Atoms,
) -> dict[SiteType, list[SurfaceSiteCandidate]]:
    """Build explicit vertex/edge/facet adsorption sites from a convex hull."""
    out: dict[SiteType, list[SurfaceSiteCandidate]] = {
        "vertex": [],
        "edge": [],
        "facet": [],
    }
    if len(core) < 4:
        return out
    pos = core.get_positions()
    com = np.mean(pos, axis=0)
    try:
        hull = ConvexHull(pos)
    except (QhullError, ValueError):
        return out

    vertices = np.asarray(hull.vertices, dtype=np.intp)
    for vidx in vertices:
        anchor = pos[int(vidx)]
        normal = _safe_normalize(anchor - com)
        out["vertex"].append(
            SurfaceSiteCandidate(site_type="vertex", anchor=anchor, normal=normal)
        )

    edge_pairs: set[tuple[int, int]] = set()
    for simplex in hull.simplices:
        i, j, k = int(simplex[0]), int(simplex[1]), int(simplex[2])
        edge_pairs.add(tuple(sorted((i, j))))
        edge_pairs.add(tuple(sorted((j, k))))
        edge_pairs.add(tuple(sorted((i, k))))
    for i, j in sorted(edge_pairs):
        midpoint = 0.5 * (pos[i] + pos[j])
        normal = _safe_normalize(midpoint - com)
        out["edge"].append(
            SurfaceSiteCandidate(site_type="edge", anchor=midpoint, normal=normal)
        )

    for simplex in hull.simplices:
        tri = pos[np.asarray(simplex, dtype=np.intp)]
        v1 = tri[1] - tri[0]
        v2 = tri[2] - tri[0]
        centroid = np.mean(tri, axis=0)
        normal = _safe_normalize(np.cross(v1, v2))
        if float(np.dot(normal, centroid - com)) < 0.0:
            normal = -normal
        out["facet"].append(
            SurfaceSiteCandidate(site_type="facet", anchor=centroid, normal=normal)
        )
    return out


def _select_site_type(
    available_types: list[SiteType],
    rng: Generator,
    within_structure_site_counts: dict[str, int] | None,
    batch_site_counts: dict[str, int] | None,
) -> SiteType:
    """Select site class with anti-repetition weighting."""
    if len(available_types) == 1:
        return available_types[0]
    weights: list[float] = []
    for st in available_types:
        local = (
            0
            if within_structure_site_counts is None
            else int(within_structure_site_counts.get(st, 0))
        )
        batch = 0 if batch_site_counts is None else int(batch_site_counts.get(st, 0))
        weights.append(1.0 / (1.0 + local + batch))
    probs = np.asarray(weights, dtype=float)
    probs /= float(np.sum(probs))
    idx = int(rng.choice(len(available_types), p=probs))
    return available_types[idx]


def blmin_for_core_and_fragment(
    core: Atoms, fragment: Atoms, blmin_ratio: float
) -> dict:
    """Minimum interatomic distances for all element pairs in core ∪ fragment."""
    zs = {int(z) for z in core.numbers}
    zs.update(int(z) for z in fragment.numbers)
    return closest_distances_generator(list(zs), ratio_of_covalent_radii=blmin_ratio)


def place_fragment_on_cluster(
    core: Atoms,
    fragment_template: Atoms,
    rng: Generator,
    config: ClusterAdsorbateConfig | None = None,
    *,
    anchor_index: int = 0,
    bond_axis: tuple[int, int] | None = None,
    within_structure_site_counts: dict[str, int] | None = None,
    batch_site_counts: dict[str, int] | None = None,
    placement_metadata: dict[str, str] | None = None,
) -> Atoms | None:
    """Rigidly place a gas-phase fragment with random orientation near the cluster.

    The fragment geometry is copied from ``fragment_template``. Positions are
    expressed relative to ``anchor_index``, optionally rotated, then the anchor
    atom is placed at ``surface_point + height * n`` with ``n`` a random outward
    direction and ``height`` uniform in ``[height_min, height_max]``.

    Args:
        core: Bare metal cluster.
        fragment_template: Equilibrium adsorbate (any size ≥ 1). Symbols and
            geometry define the rigid body; only positions are transformed.
        rng: Random number generator.
        config: Placement parameters.
        anchor_index: Fragment atom placed along the height offset from the
            outermost core atom in direction ``n``.
        bond_axis: If ``(i, j)``, rotate the fragment so the vector from atom
            ``i`` to ``j`` aligns with the outward normal before optional spin.
            Use for diatomics (e.g. OH, CO). If ``None``, apply a uniform random
            rotation (for multi-atom molecules such as H2O).

    Returns:
        The positioned fragment only, or ``None`` if placement fails.

    When ``config.validate_combined_structure`` is True (default), candidates
    that fail the same connectivity / clash checks as cluster initialization
    are discarded and retried.
    """
    if config is None:
        config = ClusterAdsorbateConfig()
    if len(core) == 0:
        raise ValueError("core must contain at least one atom")
    n_frag = len(fragment_template)
    if n_frag == 0:
        raise ValueError("fragment_template must contain at least one atom")
    if not (0 <= anchor_index < n_frag):
        raise ValueError(
            f"anchor_index={anchor_index} invalid for fragment with {n_frag} atoms"
        )
    if bond_axis is not None:
        i, j = bond_axis
        if not (0 <= i < n_frag and 0 <= j < n_frag) or i == j:
            raise ValueError(f"bond_axis={bond_axis} invalid for this fragment")

    core_pos = core.get_positions()
    com = np.mean(core_pos, axis=0)
    relative_core_pos = core_pos - com
    blmin = blmin_for_core_and_fragment(core, fragment_template, config.blmin_ratio)
    symbols = fragment_template.get_chemical_symbols()
    site_candidates = _compute_surface_site_candidates(core)
    flat_candidates = [
        candidate for entries in site_candidates.values() for candidate in entries
    ]

    # Precompute base geometry relative to anchor
    base_frag_pos = fragment_template.get_positions().astype(float).copy()
    base_frag_pos -= base_frag_pos[anchor_index]

    for _ in range(config.max_placement_attempts):
        chosen_site_type: str = "directional_fallback"
        selected_type: SiteType | None = None
        if flat_candidates:
            available_types: list[SiteType] = [
                st for st in ("vertex", "edge", "facet") if site_candidates[st]
            ]
            selected_type = _select_site_type(
                available_types=available_types,
                rng=rng,
                within_structure_site_counts=within_structure_site_counts,
                batch_site_counts=batch_site_counts,
            )
            chosen = site_candidates[selected_type][
                int(rng.integers(0, len(site_candidates[selected_type])))
            ]
            n_dir = chosen.normal
            anchor_surf = chosen.anchor
            chosen_site_type = selected_type
        else:
            n_dir = random_unit_vector(rng)
            anchor_surf = outermost_point_along_normal(
                core_pos, relative_core_pos, n_dir
            )
        h_off = float(rng.uniform(config.height_min, config.height_max))
        target = anchor_surf + h_off * n_dir

        pos = base_frag_pos.copy()

        if n_frag > 1:
            if bond_axis is not None:
                ia, ja = bond_axis
                v = pos[ja] - pos[ia]
                vn = float(np.linalg.norm(v))
                if vn < 1e-10:
                    continue
                v = v / vn
                r_align = rotation_matrix_a_to_b(v, n_dir)
                pos = (r_align @ pos.T).T
                if config.random_spin_about_normal:
                    r_spin = random_spin_about_normal(rng, n_dir)
                    pos = (r_spin @ pos.T).T
            else:
                r_rand = random_rotation_matrix(rng)
                pos = (r_rand @ pos.T).T

        pos += target

        frag = Atoms(
            symbols=symbols,
            positions=pos,
            cell=core.get_cell(),
            pbc=core.get_pbc(),
        )

        if atoms_too_close_two_sets(frag, core, blmin):
            continue

        if config.validate_combined_structure:
            trial = combine_core_adsorbate(core, frag)
            ok, _msg = validate_combined_cluster_structure(
                trial,
                min_distance_factor=config.structure_min_distance_factor,
                connectivity_factor=config.structure_connectivity_factor,
                check_clashes=config.structure_check_clashes,
                check_connectivity=config.structure_check_connectivity,
            )
            if not ok:
                continue

        if selected_type is not None and within_structure_site_counts is not None:
            within_structure_site_counts[selected_type] = (
                int(within_structure_site_counts.get(selected_type, 0)) + 1
            )
        if placement_metadata is not None:
            placement_metadata["site_type"] = chosen_site_type
        return frag

    logger.warning(
        "place_fragment_on_cluster: exceeded max_placement_attempts=%s",
        config.max_placement_attempts,
    )
    return None


def place_oh_on_cluster(
    core: Atoms,
    rng: Generator,
    config: ClusterOHConfig | None = None,
) -> Atoms | None:
    """Build an OH fragment positioned near the cluster (O = anchor, O–H ∥ outward)."""
    if config is None:
        config = ClusterOHConfig()
    d = config.oh_bond_length
    tmpl = Atoms(
        symbols=["O", "H"],
        positions=np.array([[0.0, 0.0, 0.0], [d, 0.0, 0.0]], dtype=float),
        cell=core.get_cell(),
        pbc=core.get_pbc(),
    )
    return place_fragment_on_cluster(
        core,
        tmpl,
        rng,
        config,
        anchor_index=0,
        bond_axis=(0, 1),
    )
