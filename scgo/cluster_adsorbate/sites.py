"""Convex-hull adsorption site discovery for cluster and slab placement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from ase import Atoms
from scipy.spatial import QhullError

from scgo.initialization.geometry_helpers import _get_cached_hull

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
    try:
        hull = _get_cached_hull(pos)
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
