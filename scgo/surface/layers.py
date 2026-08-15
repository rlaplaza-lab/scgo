"""Layer clustering helpers for slab systems (leaf module; no package cycles)."""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fclusterdata

from scgo.utils.logging import get_logger

logger = get_logger(__name__)

_LAYER_CLUSTER_THRESHOLD_ANG = 0.4


def _layer_indices_by_clustering(
    positions: np.ndarray,
    axis: int,
    *,
    n_layers: int,
    from_top: bool,
    threshold: float = _LAYER_CLUSTER_THRESHOLD_ANG,
) -> set[int]:
    """Return atom indices in ``n_layers`` distinct coordinate layers along ``axis``."""
    if n_layers < 1 or len(positions) == 0:
        return set()

    coord = positions[:, axis].reshape(-1, 1)
    try:
        clusters = fclusterdata(
            coord,
            threshold,
            criterion="distance",
            method="single",
        )
        unique_clusters = np.unique(clusters)
        cluster_means = np.array([coord[clusters == c].mean() for c in unique_clusters])
        sorted_cluster_ids = unique_clusters[np.argsort(cluster_means)]

        if from_top:
            if len(sorted_cluster_ids) <= n_layers:
                return set(range(len(positions)))
            selected = sorted_cluster_ids[-n_layers:]
        else:
            if len(sorted_cluster_ids) <= n_layers:
                return set(range(len(positions)))
            selected = sorted_cluster_ids[:n_layers]

        return {i for i in range(len(positions)) if clusters[i] in selected}
    except (ValueError, TypeError, np.linalg.LinAlgError):
        logger.debug(
            "Layer selection via fclusterdata failed; using coordinate rounding "
            "fallback",
            exc_info=True,
        )
        coord_flat = positions[:, axis]
        rounded = np.round(coord_flat, decimals=6)
        unique_vals = np.sort(np.unique(rounded))
        if from_top:
            if len(unique_vals) <= n_layers:
                return set(range(len(positions)))
            top_vals = set(unique_vals[-n_layers:].tolist())
            return {i for i in range(len(positions)) if rounded[i] in top_vals}
        if len(unique_vals) <= n_layers:
            return set(range(len(positions)))
        cutoff = unique_vals[n_layers - 1]
        return {i for i in range(len(positions)) if rounded[i] <= cutoff + 1e-9}
