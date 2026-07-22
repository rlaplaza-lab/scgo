# fmt: off

from __future__ import annotations

"""Mutation that translates adsorbate atoms parallel to the slab surface."""

import numpy as np
from ase import Atoms
from ase_ga.offspring_creator import OffspringCreator
from ase_ga.utilities import atoms_too_close, atoms_too_close_two_sets

from scgo.ase_ga_patches.mutations._common import (
    _append_unique_unit_vector,
    _ensure_rng,
)
from scgo.ase_ga_patches.mutations._finalize import _finalize_mutant
from scgo.system_types import SystemType, get_system_policy

__all__ = ["InPlaneSlideMutation"]


class InPlaneSlideMutation(OffspringCreator):
    """Randomly translates adsorbate atoms parallel to the slab surface.

    Parameters
    ----------
    blmin : dict
        Minimum allowed interatomic distances.
    n_top : int
        Number of adsorbate atoms optimised by the GA.
    surface_normal_axis : int
        Cartesian axis index (0, 1, or 2) normal to the surface.
    max_displacement : float
        Maximum displacement magnitude (Å) per in-plane direction.
    rng : numpy.random.Generator or None
        Random number generator.
    max_inner_attempts : int
        Maximum number of random displacement attempts per call.
    """

    def __init__(self, blmin, n_top, system_type: SystemType, surface_normal_axis=2,
                 max_displacement=2.0, target_tags=None, rng=None, verbose=False,
                 max_inner_attempts=1000):
        rng = _ensure_rng(rng)
        OffspringCreator.__init__(self, verbose, rng=rng)
        self.blmin = blmin
        self.n_top = n_top
        self.surface_normal_axis = surface_normal_axis
        self.max_displacement = max_displacement
        self.target_tags = target_tags
        self.system_type = system_type
        self._policy = get_system_policy(system_type)
        self.max_inner_attempts = max_inner_attempts
        self.last_attempt_count = 0
        self.test_dist_to_slab = True
        self.descriptor = "InPlaneSlideMutation"
        self.min_inputs = 1

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
            nearest = np.argsort(distance_sq)[:min(8, len(distance_sq))]
            if len(nearest) > 0:
                weights = 1.0 / np.maximum(distance_sq[nearest], 1e-3)
                repulsion = np.sum(delta[nearest] * weights[:, np.newaxis], axis=0)
                _append_unique_unit_vector(directions, repulsion)
                if len(directions) > 0:
                    primary_direction = directions[0]

        centered = positions_2d - center_2d
        if len(centered) > 1:
            covariance = np.dot(centered.T, centered)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            for index in np.argsort(eigenvalues)[::-1]:
                _append_unique_unit_vector(directions, eigenvectors[:, index])

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
            _append_unique_unit_vector(ordered_directions, primary_direction)
            _append_unique_unit_vector(ordered_directions, perpendicular)
            _append_unique_unit_vector(ordered_directions, -perpendicular)
            _append_unique_unit_vector(ordered_directions, -primary_direction)

        for direction in directions:
            _append_unique_unit_vector(ordered_directions, direction)
            _append_unique_unit_vector(ordered_directions, -direction)

        magnitudes = [
            0.5 * self.max_displacement,
            self.max_displacement,
            0.25 * self.max_displacement,
        ]
        candidate_shifts = []
        for direction in ordered_directions:
            for magnitude in magnitudes:
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

    def get_new_individual(self, parents):
        f = parents[0]
        indi = self.mutate(f)
        return _finalize_mutant(self, f, indi, "mutation: in_plane_slide")

    def mutate(self, atoms):
        N = len(atoms) if self.n_top is None else self.n_top
        slab = atoms[:len(atoms) - N]
        top = atoms[-N:]
        pos = top.get_positions()
        num = top.get_atomic_numbers()
        cell = top.get_cell()
        pbc = top.get_pbc()
        tags = top.get_tags() if hasattr(top, "get_tags") else np.arange(N)

        # Determine which tags to target
        unique_tags = np.unique(tags)
        if self.target_tags is not None:
            target_tags_set = set(self.target_tags)
            unique_tags = np.array([t for t in unique_tags if t in target_tags_set])
            if len(unique_tags) == 0:
                return None

        # Filter positions to only include targeted tags
        mask = np.isin(tags, unique_tags)
        if not np.any(mask):
            return None

        # Determine in-plane axes (excluding surface normal)
        in_plane = [i for i in range(3) if i != self.surface_normal_axis]

        self.last_attempt_count = 0
        for shift in self._candidate_shift_vectors(slab, pos[mask], in_plane):
            self.last_attempt_count += 1
            new_pos = pos.copy()
            new_pos[mask] += shift
            cand = Atoms(num, positions=new_pos, cell=cell, pbc=pbc)
            if atoms_too_close(cand, self.blmin):
                continue
            if self.test_dist_to_slab and len(slab) > 0 and atoms_too_close_two_sets(slab, cand, self.blmin):
                continue
            return slab + cand
        return None

# fmt: on
