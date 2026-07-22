# fmt: off

from __future__ import annotations

"""Shared helpers used by more than one mutation operator."""

import numpy as np

from scgo.ase_ga_patches._vector_utils import (
    append_unique_unit_vector as _append_unique_unit_vector,
)
from scgo.ase_ga_patches._vector_utils import random_unit_vector as _random_unit_vector
from scgo.utils.rng_helpers import ensure_rng_or_create as _ensure_rng

__all__ = [
    "_append_unique_unit_vector",
    "_ensure_rng",
    "_geometry_candidate_directions",
    "_random_unit_vector",
]


def _geometry_candidate_directions(positions, center_of_mass, slab, rng, max_candidates):
    """Ranked unit directions from slab normal, PCA axes, and random fill."""
    centered = positions - center_of_mass
    candidates = []
    outward = None

    if len(slab) > 0:
        outward = center_of_mass - np.mean(slab.get_positions(), axis=0)
        _append_unique_unit_vector(candidates, outward)

    if len(centered) > 1:
        covariance = np.dot(centered.T, centered)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)
        axes = [eigenvectors[:, index] for index in order]

        for axis in axes:
            oriented_axis = axis
            if outward is not None and np.dot(oriented_axis, outward) < 0.0:
                oriented_axis = -oriented_axis
            _append_unique_unit_vector(candidates, oriented_axis)

        if len(axes) >= 2:
            blends = [axes[0] + axes[-1]]
            if len(axes) >= 3:
                blends.append(axes[1] + axes[-1])
            else:
                blends.append(axes[0] + axes[1])
            for axis in blends:
                oriented_axis = axis
                if outward is not None and np.dot(oriented_axis, outward) < 0.0:
                    oriented_axis = -oriented_axis
                _append_unique_unit_vector(candidates, oriented_axis)

        radial_norms = np.linalg.norm(centered, axis=1)
        if len(radial_norms) > 0:
            radial_axis = centered[int(np.argmax(radial_norms))]
            if outward is not None and np.dot(radial_axis, outward) < 0.0:
                radial_axis = -radial_axis
            _append_unique_unit_vector(candidates, radial_axis)
    else:
        _append_unique_unit_vector(candidates, np.array([0.0, 0.0, 1.0]))

    attempts = 0
    while len(candidates) < max_candidates and attempts < 100:
        axis = _random_unit_vector(rng)
        if outward is not None and np.dot(axis, outward) < 0.0:
            axis = -axis
        _append_unique_unit_vector(candidates, axis)
        attempts += 1

    return candidates[:max_candidates]

# fmt: on
