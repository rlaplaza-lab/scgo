"""Mutation that translates adsorbate atoms parallel to the slab surface."""

from __future__ import annotations


import numpy as np
from ase import Atoms

from scgo.ase_ga_patches.mutations._common import (
    MutationOperator,
    _accept_connected_candidate,
    _accept_connected_kwargs,
    _append_unique_unit_vector,
    _pca_sorted_eigenvectors,
    _prepare_mobile_target,
)
from scgo.system_types import SystemType

__all__ = ["InPlaneSlideMutation"]


class InPlaneSlideMutation(MutationOperator):
    descriptor = "InPlaneSlideMutation"
    """Translates adsorbate atoms parallel to the slab surface.

    Candidate shifts combine directions from slab repulsion, the in-plane
    principal axes and fixed in-plane axis/diagonal directions, scaled by
    fixed fractions of ``max_displacement``; the first shift that violates
    no *blmin* pair is accepted.

    Parameters
    ----------
    blmin : dict
        Minimum allowed interatomic distances.
    n_top : int
        Number of adsorbate atoms optimized by the GA.
    surface_normal_axis : int
        Cartesian axis index (0, 1, or 2) normal to the surface.
    max_displacement : float
        Maximum displacement magnitude (Å) per in-plane direction.
    rng : numpy.random.Generator or None
        Random number generator.
    max_inner_attempts : int
        Upper bound on the number of candidate shifts per call (capped at 12).
    """

    def __init__(
        self,
        blmin,
        n_top,
        system_type: SystemType,
        surface_normal_axis=2,
        max_displacement=2.0,
        min_displacement=1.0,
        target_tags=None,
        rng=None,
        verbose=False,
        max_inner_attempts=1000,
    ):
        self._init_operator(
            verbose,
            rng,
            system_type,
            n_top=n_top,
            blmin=blmin,
            target_tags=target_tags,
            surface_normal_axis=surface_normal_axis,
        )
        self.max_displacement = max_displacement
        self.min_displacement = min_displacement
        self.max_inner_attempts = max_inner_attempts
        self.last_attempt_count = 0

    def _candidate_shift_vectors(self, slab, positions, in_plane):
        if self.max_displacement <= 1e-12:
            return []

        max_candidates = max(1, min(int(self.max_inner_attempts), 12))
        positions_2d = positions[:, in_plane]
        center_2d = np.mean(positions_2d, axis=0)
        directions = []
        primary_direction = None

        if len(slab) > 0:
            slab_2d = slab.get_positions()[:, in_plane]
            delta = center_2d - slab_2d
            distance_sq = np.sum(delta * delta, axis=1)
            nearest = np.argsort(distance_sq)[: min(8, len(distance_sq))]
            if len(nearest) > 0:
                weights = 1.0 / np.maximum(distance_sq[nearest], 1e-3)
                repulsion = np.sum(delta[nearest] * weights[:, np.newaxis], axis=0)
                _append_unique_unit_vector(directions, repulsion)
                if len(directions) > 0:
                    primary_direction = directions[0]

        centered = positions_2d - center_2d
        if len(centered) > 1:
            for axis in _pca_sorted_eigenvectors(centered, descending=True):
                _append_unique_unit_vector(directions, axis)

        _append_unique_unit_vector(directions, np.array([1.0, 0.0]))
        _append_unique_unit_vector(directions, np.array([0.0, 1.0]))
        _append_unique_unit_vector(directions, np.array([1.0, 1.0]))
        _append_unique_unit_vector(directions, np.array([1.0, -1.0]))

        if primary_direction is None and len(directions) > 0:
            primary_direction = directions[0]

        ordered_directions = []
        if primary_direction is not None:
            ordered_directions.append(primary_direction)
            perpendicular = np.array([-primary_direction[1], primary_direction[0]])
            _append_unique_unit_vector(ordered_directions, perpendicular)
            _append_unique_unit_vector(ordered_directions, -perpendicular)
            _append_unique_unit_vector(ordered_directions, -primary_direction)

        for direction in directions:
            _append_unique_unit_vector(ordered_directions, direction)
            _append_unique_unit_vector(ordered_directions, -direction)

        raw_magnitudes = (
            self.max_displacement,
            0.5 * self.max_displacement,
            self.min_displacement,
        )
        magnitudes = []
        for value in raw_magnitudes:
            if value < self.min_displacement - 1e-9:
                continue
            if value > self.max_displacement + 1e-9:
                continue
            if any(abs(value - existing) <= 1e-8 for existing in magnitudes):
                continue
            magnitudes.append(float(value))
        candidate_shifts = []
        for magnitude in magnitudes:
            for direction in ordered_directions:
                if magnitude <= 1e-12:
                    continue
                shift_2d = magnitude * direction
                shift = np.zeros(3, dtype=float)
                shift[in_plane[0]] = shift_2d[0]
                shift[in_plane[1]] = shift_2d[1]
                duplicate = False
                for existing in candidate_shifts:
                    if np.linalg.norm(existing - shift) <= 1e-8:
                        duplicate = True
                        break
                if not duplicate:
                    candidate_shifts.append(shift)
                if len(candidate_shifts) >= max_candidates:
                    return candidate_shifts

        return candidate_shifts[:max_candidates]

    def mutate(self, atoms):
        prepared = _prepare_mobile_target(self, atoms)
        if prepared is None:
            return None
        slab, top, pos, num, cell, pbc, tags, mask = prepared

        # Determine in-plane axes (excluding surface normal)
        in_plane = [i for i in range(3) if i != self.surface_normal_axis]

        self.last_attempt_count = 0
        for shift in self._candidate_shift_vectors(slab, pos[mask], in_plane):
            self.last_attempt_count += 1
            new_pos = pos.copy()
            new_pos[mask] += shift
            cand = Atoms(num, positions=new_pos, cell=cell, pbc=pbc, tags=tags)
            accepted = _accept_connected_candidate(
                top, cand, **_accept_connected_kwargs(self, slab)
            )
            if accepted is None:
                continue
            return slab + accepted
        return None
