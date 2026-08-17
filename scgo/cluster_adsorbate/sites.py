"""Convex-hull adsorption site discovery for cluster and slab placement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from ase import Atoms
from scipy.spatial import Delaunay, QhullError, Voronoi, cKDTree

from scgo.exceptions import SCGOValidationError
from scgo.initialization.geometry_helpers import resolve_cluster_extent

SiteType = Literal["vertex", "edge", "facet"]


@dataclass(frozen=True)
class SurfaceSiteCandidate:
    site_type: SiteType
    anchor: np.ndarray
    normal: np.ndarray


# Call-stack / placement-session cache: identical site-core geometries reuse Qhull.
_SITE_CANDIDATE_CACHE: dict[int, dict[SiteType, list[SurfaceSiteCandidate]]] = {}
# Planar-layer (graphene/graphite) hollow-site cache, keyed by positions hash,
# surface normal axis, in-plane cell and PBC so the periodic Voronoi branch is
# reproducible and reused across the up-to-1000 placement attempts per structure.
_PLANAR_SITE_CACHE: dict[
    tuple[int, int, int, tuple[bool, ...]],
    dict[SiteType, list[SurfaceSiteCandidate]],
] = {}
_SITE_CACHE_MAX = 64


def _safe_normalize(v: np.ndarray) -> np.ndarray:
    vn = float(np.linalg.norm(v))
    if vn < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return v / vn


def _empty_sites() -> dict[SiteType, list[SurfaceSiteCandidate]]:
    return {"vertex": [], "edge": [], "facet": []}


def _site_core_positions_key(core: Atoms) -> int:
    pos = np.ascontiguousarray(core.get_positions(), dtype=np.float64)
    return hash(pos.tobytes())


def _planar_cache_key(
    layer: Atoms, axis: int
) -> tuple[int, int, int, tuple[bool, ...]]:
    """Cache key for a planar layer: positions hash, axis, cell hash and PBC."""
    pos_key = _site_core_positions_key(layer)
    cell = np.ascontiguousarray(layer.cell.array, dtype=np.float64)
    cell_key = hash(cell.tobytes())
    pbc = tuple(bool(p) for p in layer.get_pbc())
    return (pos_key, int(axis), cell_key, pbc)


def clear_surface_site_cache() -> None:
    """Drop cached hull and planar-layer site candidates (e.g. between sessions)."""
    _SITE_CANDIDATE_CACHE.clear()
    _PLANAR_SITE_CACHE.clear()


def get_or_compute_surface_site_candidates(
    core: Atoms,
) -> dict[SiteType, list[SurfaceSiteCandidate]]:
    """Return surface sites for ``core``, caching by positions hash."""
    key = _site_core_positions_key(core)
    cached = _SITE_CANDIDATE_CACHE.get(key)
    if cached is not None:
        return cached
    result = compute_surface_site_candidates(core)
    if len(_SITE_CANDIDATE_CACHE) >= _SITE_CACHE_MAX:
        # Drop the oldest inserted entry (dicts preserve insertion order).
        _SITE_CANDIDATE_CACHE.pop(next(iter(_SITE_CANDIDATE_CACHE)))
    _SITE_CANDIDATE_CACHE[key] = result
    return result


def compute_surface_site_candidates(
    core: Atoms,
) -> dict[SiteType, list[SurfaceSiteCandidate]]:
    """Build vertex/edge/facet sites from 3D hull or PCA extent.

    Empty for ``n < 4``. Periodic planar slabs stay empty here so callers use
    :func:`planar_layer_site_candidates` (Voronoi hollows).
    """
    out = _empty_sites()
    if len(core) < 4:
        return out

    pos = core.get_positions()
    com = np.mean(pos, axis=0)
    extent = resolve_cluster_extent(pos)

    if extent.kind == "linear" and extent.axis is not None:
        axis_u = _safe_normalize(extent.axis)
        for vidx in extent.vertices:
            anchor = pos[int(vidx)]
            outward = _safe_normalize(anchor - com)
            normal = -axis_u if float(np.dot(outward, axis_u)) < 0.0 else axis_u
            out["vertex"].append(
                SurfaceSiteCandidate(site_type="vertex", anchor=anchor, normal=normal)
            )
        if len(extent.vertices) >= 2:
            i0, i1 = int(extent.vertices[0]), int(extent.vertices[1])
            mid = 0.5 * (pos[i0] + pos[i1])
            out["edge"].append(
                SurfaceSiteCandidate(
                    site_type="edge",
                    anchor=mid,
                    normal=_safe_normalize(mid - com),
                )
            )
        return out

    if extent.kind == "planar":
        if any(bool(p) for p in core.get_pbc()):
            return out
        verts = [int(v) for v in extent.vertices]
        plane_n = _safe_normalize(extent.normal)
        if float(np.dot(plane_n, np.mean(pos[verts], axis=0) - com)) < 0.0:
            plane_n = -plane_n
        for vidx in verts:
            anchor = pos[vidx]
            out["vertex"].append(
                SurfaceSiteCandidate(
                    site_type="vertex",
                    anchor=anchor,
                    normal=_safe_normalize(anchor - com),
                )
            )
        if len(verts) >= 2:
            ref = np.array([1.0, 0.0, 0.0])
            if abs(float(np.dot(ref, plane_n))) > 0.9:
                ref = np.array([0.0, 1.0, 0.0])
            e1 = _safe_normalize(np.cross(plane_n, ref))
            e2 = _safe_normalize(np.cross(plane_n, e1))
            centered = pos[verts] - com
            order = [
                verts[i] for i in np.argsort(np.arctan2(centered @ e2, centered @ e1))
            ]
            for a, b in zip(order, order[1:] + order[:1], strict=True):
                mid = 0.5 * (pos[a] + pos[b])
                out["edge"].append(
                    SurfaceSiteCandidate(
                        site_type="edge",
                        anchor=mid,
                        normal=_safe_normalize(mid - com),
                    )
                )
            out["facet"].append(
                SurfaceSiteCandidate(
                    site_type="facet",
                    anchor=np.mean(pos[order], axis=0),
                    normal=plane_n,
                )
            )
        return out

    hull = extent.hull
    if hull is None:
        return out

    for vidx in np.asarray(hull.vertices, dtype=np.intp):
        anchor = pos[int(vidx)]
        out["vertex"].append(
            SurfaceSiteCandidate(
                site_type="vertex",
                anchor=anchor,
                normal=_safe_normalize(anchor - com),
            )
        )

    edge_pairs: set[tuple[int, int]] = set()
    for simplex in hull.simplices:
        i, j, k = int(simplex[0]), int(simplex[1]), int(simplex[2])
        edge_pairs.add(tuple(sorted((i, j))))
        edge_pairs.add(tuple(sorted((j, k))))
        edge_pairs.add(tuple(sorted((i, k))))
    for i, j in sorted(edge_pairs):
        mid = 0.5 * (pos[i] + pos[j])
        out["edge"].append(
            SurfaceSiteCandidate(
                site_type="edge", anchor=mid, normal=_safe_normalize(mid - com)
            )
        )

    for simplex in hull.simplices:
        tri = pos[np.asarray(simplex, dtype=np.intp)]
        centroid = np.mean(tri, axis=0)
        normal = _safe_normalize(np.cross(tri[1] - tri[0], tri[2] - tri[0]))
        if float(np.dot(normal, centroid - com)) < 0.0:
            normal = -normal
        out["facet"].append(
            SurfaceSiteCandidate(site_type="facet", anchor=centroid, normal=normal)
        )
    return out


def _planar_hollow_sites(
    layer: Atoms,
    *,
    surface_normal_axis: int,
) -> list[SurfaceSiteCandidate]:
    """Hollow (3-fold) adsorption sites for a flat slab layer via Voronoi.

    A planar layer (graphene/graphite top) has no 3D convex hull, but its hollow
    sites are the vertices of the in-plane Voronoi diagram. For fully-periodic
    in-plane directions with a non-degenerate in-plane cell, the points are tiled
    over a 3×3 image block and only vertices whose in-plane fractional
    coordinates fall in ``[0, 1)`` are kept (the periodic Voronoi). Otherwise the
    Voronoi is built on the raw points and only vertices inside the in-plane
    convex hull (``Delaunay.find_simplex >= 0``) are kept.

    Candidate vertices closer than half the median in-plane nearest-neighbor
    spacing to any atom are dropped (they overlap an on-top / bridge site), and
    the 3 nearest in-plane atoms set the hollow anchor's out-of-plane coordinate
    (mean of their ``surface_normal_axis`` coordinates). Degenerate / collinear
    inputs (``< 4`` points) raise ``QhullError`` and yield no hollow sites.

    Args:
        layer: Planar slab-layer Atoms (positions used for the in-plane diagram).
        surface_normal_axis: Cartesian axis index for the outward surface normal.

    Returns:
        A list of ``facet`` (hollow) ``SurfaceSiteCandidate`` entries, each with a
        ``+axis`` outward normal.
    """
    axis = int(surface_normal_axis)
    if axis not in (0, 1, 2):
        raise SCGOValidationError(f"surface_normal_axis must be 0, 1, or 2, got {axis}")
    out: list[SurfaceSiteCandidate] = []
    pos = np.asarray(layer.get_positions(), dtype=float)
    if len(pos) < 4:
        return out
    in_plane = [i for i in (0, 1, 2) if i != axis]
    proj = pos[:, in_plane]  # (n, 2) in-plane projections
    cell = np.asarray(layer.cell.array, dtype=float)
    in_plane_cell = cell[np.ix_(in_plane, in_plane)]
    det = float(np.linalg.det(in_plane_cell))
    pbc = np.asarray(layer.get_pbc(), dtype=bool)
    periodic = bool(pbc[in_plane[0]] and pbc[in_plane[1]]) and abs(det) > 1e-8

    try:
        if periodic:
            shifts = np.array(
                [
                    [-1, -1],
                    [-1, 0],
                    [-1, 1],
                    [0, -1],
                    [0, 0],
                    [0, 1],
                    [1, -1],
                    [1, 0],
                    [1, 1],
                ],
                dtype=float,
            )
            tiled = np.vstack([proj + s @ in_plane_cell for s in shifts])
            vor = Voronoi(tiled)
            inv = np.linalg.inv(in_plane_cell)
            frac = vor.vertices @ inv
            keep = np.all((frac >= 0.0) & (frac < 1.0), axis=1)
            vertices = vor.vertices[keep]
        else:
            vor = Voronoi(proj)
            hull = Delaunay(proj)
            inside = hull.find_simplex(vor.vertices) >= 0
            vertices = vor.vertices[inside]
    except QhullError:
        # Degenerate / collinear point set: no hollow sites.
        return out

    if len(vertices) == 0:
        return out

    # Median in-plane nearest-neighbor spacing sets the on-top/bridge overlap floor.
    tree = cKDTree(proj)
    nn_dists, _ = tree.query(proj, k=2)
    nn = nn_dists[:, 1]
    median_nn = float(np.median(nn))
    min_dist = 0.5 * median_nn
    if min_dist <= 0.0:
        return out

    axis_coords = pos[:, axis]
    normal = np.zeros(3, dtype=float)
    normal[axis] = 1.0
    seen: set[tuple[float, float]] = set()
    for v in vertices:
        d = np.linalg.norm(proj - v, axis=1)
        if float(np.min(d)) < min_dist:
            continue
        order = np.argsort(d)
        nearest3 = order[:3]
        mean_axis = float(np.mean(axis_coords[nearest3]))
        key = (round(float(v[0]), 6), round(float(v[1]), 6))
        if key in seen:
            continue
        seen.add(key)
        anchor = np.zeros(3, dtype=float)
        anchor[in_plane] = v
        anchor[axis] = mean_axis
        out.append(
            SurfaceSiteCandidate(site_type="facet", anchor=anchor, normal=normal.copy())
        )
    return out


def planar_layer_site_candidates(
    layer: Atoms,
    *,
    surface_normal_axis: int = 2,
) -> dict[SiteType, list[SurfaceSiteCandidate]]:
    """Build atom-, bond-midpoint and hollow sites for a flat slab layer (no 3D hull).

    Graphene/graphite top layers are planar, so ``try_convex_hull`` fails. Use
    each atom as a vertex (on-top) site, nearest-neighbor midpoints as edge
    (bridge) sites, and the in-plane Voronoi vertices as facet (hollow) sites,
    all with the outward surface normal along ``surface_normal_axis``.

    Raises:
        SCGOValidationError: If ``layer`` is non-empty and
            ``surface_normal_axis`` is not 0, 1, or 2.
    """
    out: dict[SiteType, list[SurfaceSiteCandidate]] = {
        "vertex": [],
        "edge": [],
        "facet": [],
    }
    if len(layer) == 0:
        return out
    axis = int(surface_normal_axis)
    if axis not in (0, 1, 2):
        raise SCGOValidationError(f"surface_normal_axis must be 0, 1, or 2, got {axis}")
    normal = np.zeros(3, dtype=float)
    normal[axis] = 1.0
    pos = np.asarray(layer.get_positions(), dtype=float)
    for anchor in pos:
        out["vertex"].append(
            SurfaceSiteCandidate(
                site_type="vertex", anchor=anchor.copy(), normal=normal.copy()
            )
        )
    if len(pos) < 2:
        return out
    # In-plane nearest-neighbor midpoints as bridge sites. The nearest-neighbor
    # relation is asymmetric (``argmin`` picks a single partner per atom), so
    # collect unordered pairs in a set and emit one midpoint per unique pair.
    in_plane = [i for i in (0, 1, 2) if i != axis]
    nn_pairs: set[tuple[int, int]] = set()
    for i, pi in enumerate(pos):
        deltas = pos - pi
        d2 = deltas[:, in_plane[0]] ** 2 + deltas[:, in_plane[1]] ** 2
        d2[i] = np.inf
        j = int(np.argmin(d2))
        if j == i:
            continue
        nn_pairs.add((min(i, j), max(i, j)))
    for i, j in sorted(nn_pairs):
        midpoint = 0.5 * (pos[i] + pos[j])
        out["edge"].append(
            SurfaceSiteCandidate(
                site_type="edge", anchor=midpoint, normal=normal.copy()
            )
        )
    # Hollow (facet) sites from the in-plane Voronoi vertices.
    out["facet"] = _planar_hollow_sites(layer, surface_normal_axis=axis)
    return out


def get_or_compute_planar_layer_site_candidates(
    layer: Atoms,
    *,
    surface_normal_axis: int = 2,
) -> dict[SiteType, list[SurfaceSiteCandidate]]:
    """Return planar-layer sites for ``layer``, caching by geometry + axis + PBC.

    Hollow-site computation builds a (possibly tiled) Voronoi diagram, which is
    expensive when called once per placement attempt (up to ~1000/structure). The
    cache reuses the result for identical layer geometry, axis, in-plane cell and
    PBC, with the same ``_SITE_CACHE_MAX`` eviction policy as the hull cache.
    """
    key = _planar_cache_key(layer, surface_normal_axis)
    cached = _PLANAR_SITE_CACHE.get(key)
    if cached is not None:
        return cached
    result = planar_layer_site_candidates(
        layer, surface_normal_axis=surface_normal_axis
    )
    if len(_PLANAR_SITE_CACHE) >= _SITE_CACHE_MAX:
        # Drop the oldest inserted entry (dicts preserve insertion order).
        _PLANAR_SITE_CACHE.pop(next(iter(_PLANAR_SITE_CACHE)))
    _PLANAR_SITE_CACHE[key] = result
    return result


def filter_sites_to_outward(
    sites: dict[SiteType, list[SurfaceSiteCandidate]],
    *,
    axis: int,
    top_layer_z_min: float,
    tol: float = 1e-6,
) -> dict[SiteType, list[SurfaceSiteCandidate]]:
    """Keep only candidates whose normal points away from the slab bulk.

    A full 3D convex hull of a slab top-layer slice yields normals in every
    direction; roughly half aim under or beside the slab and are unusable for
    deposition. A candidate is kept when its normal has a positive component
    along ``axis`` and its anchor sits at or above ``top_layer_z_min``.

    Returns a **new** dict; ``sites`` is never mutated (the hull cache in
    :func:`scgo.cluster_adsorbate.sites.get_or_compute_surface_site_candidates` shares its value).

    Raises:
        SCGOValidationError: If ``axis`` is not 0, 1, or 2.
    """
    ax = int(axis)
    if ax not in (0, 1, 2):
        raise SCGOValidationError(f"axis must be 0, 1, or 2, got {ax}")
    out: dict[SiteType, list[SurfaceSiteCandidate]] = {
        "vertex": [],
        "edge": [],
        "facet": [],
    }
    for site_type, entries in sites.items():
        kept = [
            candidate
            for candidate in entries
            if float(candidate.normal[ax]) > tol
            and float(candidate.anchor[ax]) >= top_layer_z_min
        ]
        out[site_type] = kept
    return out


def count_site_candidates(
    sites: dict[SiteType, list[SurfaceSiteCandidate]],
) -> int:
    """Return total number of vertex/edge/facet candidates."""
    return sum(len(entries) for entries in sites.values())
