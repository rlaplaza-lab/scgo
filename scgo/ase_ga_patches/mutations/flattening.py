# fmt: off

from __future__ import annotations

"""Mutation that flattens a nanoparticle by projecting onto a random plane."""

import numpy as np
from ase_ga.offspring_creator import OffspringCreator
from ase_ga.utilities import atoms_too_close, atoms_too_close_two_sets

from scgo.ase_ga_patches.mutations._common import (
    _ensure_rng,
    _geometry_candidate_directions,
)
from scgo.ase_ga_patches.mutations._finalize import _finalize_mutant
from scgo.initialization.steric_scoring import get_blmin_distance as _get_blmin_distance
from scgo.system_types import SystemType, get_system_policy

__all__ = ["FlatteningMutation"]


class FlatteningMutation(OffspringCreator):
    """A mutation that flattens the nanoparticle by projecting the coordinates
    to a plane that cuts the structure in a random angle.
    Atoms are then perturbed perpendicular to the plane within a given thickness.

    Parameters
    ----------
    blmin: Dictionary defining the minimum allowed
        distance between atoms.

    n_top: Number of atoms the GA optimizes.

    thickness_factor: Factor to multiply with the average blmin to determine
        the thickness of the slab for projection.

    test_dist_to_slab: Whether also the distances to the slab
        should be checked to satisfy the blmin.

    rng: Random number generator
        By default numpy.random.

    """

    def __init__(self, blmin, n_top, system_type: SystemType, thickness_factor=0.5,
                 test_dist_to_slab=True, target_tags=None, rng=None, verbose=False,
                 max_inner_attempts=12):
        rng = _ensure_rng(rng)
        OffspringCreator.__init__(self, verbose, rng=rng)
        self.blmin = blmin
        self.n_top = n_top
        self.thickness_factor = thickness_factor
        self.test_dist_to_slab = test_dist_to_slab
        self.target_tags = target_tags
        self.max_inner_attempts = max_inner_attempts
        self.system_type = system_type
        self._policy = get_system_policy(system_type)
        self.last_attempt_count = 0

        self.descriptor = "FlatteningMutation"
        self.min_inputs = 1

    def _candidate_normals(self, positions, center_of_mass, slab):
        max_candidates = max(1, min(int(self.max_inner_attempts), 6))
        return _geometry_candidate_directions(
            positions,
            center_of_mass,
            slab,
            self.rng,
            max_candidates,
        )

    def _resolve_normal_offsets(
        self,
        projected_positions,
        target_offsets,
        atomic_numbers,
        clearance_margin,
    ):
        from scipy.spatial.distance import pdist, squareform

        n_atoms = len(projected_positions)
        if n_atoms <= 1:
            return np.zeros(n_atoms)

        order = np.argsort(target_offsets)
        ordered_targets = np.asarray(target_offsets, dtype=float)[order].copy()
        ordered_positions = projected_positions[order]
        ordered_numbers = atomic_numbers[order]

        # Vectorize lateral distance calculations
        lateral_distances = squareform(pdist(ordered_positions))

        # Vectorize blmin lookup
        blmin_matrix = np.zeros((n_atoms, n_atoms), dtype=float)
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                blmin_matrix[i, j] = _get_blmin_distance(self.blmin, ordered_numbers[i], ordered_numbers[j])

        required_distances = blmin_matrix + clearance_margin

        # Calculate required offsets only for upper triangle
        required_offsets = np.zeros((n_atoms, n_atoms), dtype=float)
        mask = lateral_distances + 1e-12 < required_distances
        sq_diff = required_distances**2 - lateral_distances**2
        required_offsets[mask] = np.sqrt(np.maximum(sq_diff[mask], 0.0))

        # Sequential solve optimized with vectorized lookback
        solved_offsets = ordered_targets
        for i in range(1, n_atoms):
            solved_offsets[i] = max(
                solved_offsets[i],
                np.max(solved_offsets[:i] + required_offsets[:i, i])
            )

        solved_offsets -= np.mean(solved_offsets)
        offsets = np.empty(n_atoms, dtype=float)
        offsets[order] = solved_offsets
        return offsets

    def _build_flatten_candidate(
        self,
        positions,
        center_of_mass,
        normal,
        atomic_numbers,
        desired_thickness,
        avg_blmin,
    ):
        centered = positions - center_of_mass
        original_offsets = np.dot(centered, normal)
        projected_positions = positions - original_offsets[:, np.newaxis] * normal
        current_span = float(np.ptp(original_offsets))

        if current_span <= 1e-12:
            target_offsets = np.zeros(len(positions), dtype=float)
        else:
            compression = min(1.0, desired_thickness / current_span)
            target_offsets = (original_offsets - np.mean(original_offsets)) * compression

        clearance_margin = max(1e-3, 1e-3 * avg_blmin)
        resolved_offsets = self._resolve_normal_offsets(
            projected_positions,
            target_offsets,
            atomic_numbers,
            clearance_margin,
        )
        candidate_positions = projected_positions + resolved_offsets[:, np.newaxis] * normal

        original_span = max(current_span, 1e-12)
        flattened_span = float(np.ptp(resolved_offsets))
        flatten_ratio = flattened_span / original_span
        rms_displacement = (
            np.linalg.norm(candidate_positions - positions)
            / max(1, len(positions)) ** 0.5
        )
        score = flatten_ratio + 0.15 * (rms_displacement / max(avg_blmin, 1e-12))
        return score, candidate_positions

    def get_new_individual(self, parents):
        f = parents[0]

        indi = self.mutate(f)

        return _finalize_mutant(self, f, indi, "mutation: flattening")

    def mutate(self, atoms):
        N = len(atoms) if self.n_top is None else self.n_top
        slab = atoms[:len(atoms) - N]
        top = atoms[-N:]

        mutant = top.copy()
        pos = mutant.get_positions()
        atomic_numbers = mutant.get_atomic_numbers()
        tags = mutant.get_tags() if hasattr(mutant, "get_tags") else np.arange(N)
        cm = np.average(pos, axis=0)

        # Determine which tags to target
        unique_tags = np.unique(tags)
        if self.target_tags is not None:
            target_tags_set = set(self.target_tags)
            unique_tags = np.array([t for t in unique_tags if t in target_tags_set])
            if len(unique_tags) == 0:
                return None

        avg_blmin = np.mean(list(self.blmin.values()))
        desired_thickness = max(0.05 * avg_blmin, avg_blmin * self.thickness_factor)

        candidate_positions = [
            self._build_flatten_candidate(
                pos,
                cm,
                normal,
                atomic_numbers,
                desired_thickness,
                avg_blmin,
            )
            for normal in self._candidate_normals(pos, cm, slab)
        ]

        candidate_positions.sort(key=lambda item: item[0])
        self.last_attempt_count = 0
        for _score, new_positions in candidate_positions:
            self.last_attempt_count += 1
            mutant.set_positions(new_positions)
            # Only center gas-phase clusters; surface adsorbates must keep
            # their positions relative to the slab.
            if not self._policy.uses_surface:
                mutant.center()

            too_close = atoms_too_close(mutant, self.blmin)
            if not too_close and self.test_dist_to_slab:
                too_close = atoms_too_close_two_sets(slab, mutant, self.blmin)

            if not too_close:
                return slab + mutant

        if len(candidate_positions) == 0:
            self.last_attempt_count = 0
            return None

        return None

# fmt: on
