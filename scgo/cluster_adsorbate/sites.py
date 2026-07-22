"""Convex-hull adsorption site discovery for cluster and slab placement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from ase import Atoms

from scgo.initialization.geometry_helpers import try_convex_hull
from scgo.utils.logging import get_logger

logger = get_logger(__name__)

SiteType = Literal["vertex", "edge", "facet"]


@dataclass(frozen=True)
class SurfaceSiteCandidate:
    site_type: SiteType
    anchor: np.ndarray
    normal: np.ndarray


# Call-stack / placement-session cache: identical site-core geometries reuse Qhull.
_SITE_CANDIDATE_CACHE: dict[int, dict[SiteType, list[SurfaceSiteCandidate]]] = {}
_SITE_CACHE_MAX = 64


def _safe_normalize(v: np.ndarray) -> np.ndarray:
    vn = float(np.linalg.norm(v))
    if vn < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return v / vn


def _site_core_positions_key(core: Atoms) -> int:
    pos = np.ascontiguousarray(core.get_positions(), dtype=np.float64)
    return hash(pos.tobytes())


def clear_surface_site_cache() -> None:
    """Drop cached hull site candidates (e.g. between independent placement stacks)."""
    _SITE_CANDIDATE_CACHE.clear()


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
        # Drop an arbitrary old entry (insertion order in CPython 3.7+).
        _SITE_CANDIDATE_CACHE.pop(next(iter(_SITE_CANDIDATE_CACHE)))
    _SITE_CANDIDATE_CACHE[key] = result
    return result


def compute_surface_site_candidates(
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
    hull = try_convex_hull(pos)
    if hull is None:
        logger.debug(
            "Convex hull site discovery unavailable for %d core atoms", len(core)
        )
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
