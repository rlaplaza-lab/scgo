# fmt: off

from __future__ import annotations

"""Mutation that uniformly scales atom positions relative to the centre of mass."""

import numpy as np
from ase import Atoms
from ase_ga.offspring_creator import OffspringCreator
from ase_ga.utilities import atoms_too_close, atoms_too_close_two_sets

from scgo.ase_ga_patches.mutations._common import _ensure_rng
from scgo.ase_ga_patches.mutations._finalize import _finalize_mutant
from scgo.initialization.steric_scoring import get_blmin_distance as _get_blmin_distance
from scgo.system_types import SystemType, get_system_policy

__all__ = ["BreathingMutation"]


class BreathingMutation(OffspringCreator):
    """Uniformly scales all atom positions relative to the centre of mass.

    Each attempt samples a random scale factor in ``[scale_min, scale_max]``
    and accepts if no pair of atoms violates *blmin*.

    Parameters
    ----------
    blmin : dict
        Minimum allowed interatomic distances.
    n_top : int
        Number of atoms optimised by the GA.
    scale_min, scale_max : float
        Bounds for the uniform scale-factor distribution.
    test_dist_to_slab : bool
        Also check distances to slab atoms.
    rng : numpy.random.Generator or None
        Random number generator.
    max_inner_attempts : int
        Maximum number of random scale attempts per call.
    """

    def __init__(self, blmin, n_top, system_type: SystemType, scale_min=0.9, scale_max=1.1,
                 test_dist_to_slab=True, target_tags=None, rng=None, verbose=False,
                 max_inner_attempts=1000):
        rng = _ensure_rng(rng)
        OffspringCreator.__init__(self, verbose, rng=rng)
        self.blmin = blmin
        self.n_top = n_top
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.test_dist_to_slab = test_dist_to_slab
        self.target_tags = target_tags
        self.system_type = system_type
        self._policy = get_system_policy(system_type)
        self.max_inner_attempts = max_inner_attempts
        self.last_attempt_count = 0
        self.descriptor = "BreathingMutation"
        self.min_inputs = 1

    def _minimum_feasible_scale(self, positions, atomic_numbers):
        from scipy.spatial.distance import pdist

        n_atoms = len(positions)
        if n_atoms <= 1:
            return self.scale_min

        # Compute pairwise distances
        distances = pdist(positions)
        if np.any(distances <= 1e-12):
            return np.inf

        # Compute pairwise blmin requirements
        blmin_matrix = np.zeros((n_atoms, n_atoms), dtype=float)
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                blmin_matrix[i, j] = _get_blmin_distance(self.blmin, atomic_numbers[i], atomic_numbers[j])

        # We only need the condensed upper triangle for blmin
        required_condensed = blmin_matrix[np.triu_indices(n_atoms, k=1)]

        # Calculate minimum required scale to avoid clashes
        lower_bound = np.max(required_condensed / distances)
        # pdist ratios can sit on the blmin threshold; atoms_too_close needs slack.
        return max(self.scale_min, lower_bound * (1.0 + 1e-6))

    def _candidate_scales(self, positions, atomic_numbers, slab):
        feasible_lower = self._minimum_feasible_scale(positions, atomic_numbers)
        tol = 1e-9
        # Dense parents (e.g. tight random-spherical init) can require s > scale_max to
        # clear blmin. Apply the minimum relieving uniform expansion instead of giving up.
        if feasible_lower > self.scale_max + tol:
            return [float(feasible_lower)]

        feasible_lower = max(self.scale_min, feasible_lower)
        interval_width = max(0.0, self.scale_max - feasible_lower)
        max_candidates = max(1, min(int(self.max_inner_attempts), 8))
        candidates = []
        allow_unit_scale = interval_width <= tol

        def append_candidate(scale, force=False):
            scale = float(scale)
            if scale < feasible_lower - tol or scale > self.scale_max + tol:
                return
            if not allow_unit_scale and abs(scale - 1.0) <= tol and not force:
                return
            for existing in candidates:
                if abs(scale - existing) <= 1e-6:
                    return
            candidates.append(scale)

        contraction_width = max(0.0, 1.0 - feasible_lower)
        expansion_width = max(0.0, self.scale_max - 1.0)
        contraction_candidates = []
        expansion_candidates = []

        if contraction_width > tol:
            contraction_candidates = [
                1.0 - 0.5 * contraction_width,
                feasible_lower,
            ]
        if expansion_width > tol:
            expansion_candidates = [
                1.0 + 0.5 * expansion_width,
                self.scale_max,
            ]

        ordered_groups = []
        if len(slab) > 0:
            ordered_groups = [contraction_candidates, expansion_candidates]
        elif expansion_width >= contraction_width:
            ordered_groups = [expansion_candidates, contraction_candidates]
        else:
            ordered_groups = [contraction_candidates, expansion_candidates]

        for group in ordered_groups:
            for scale in group:
                append_candidate(scale)

        if contraction_candidates and expansion_candidates:
            append_candidate(0.5 * (feasible_lower + self.scale_max))
        # Always include unit scale (1.0) as a candidate, regardless of interval width,
        # as long as it's within the valid range. This is important for cases where
        # the cluster is already in a good configuration and scaling would make it worse.
        if feasible_lower <= 1.0 <= self.scale_max + tol or allow_unit_scale:
            append_candidate(1.0, force=True)

        # Ensure unit scale (1.0) is included in the final candidates if it's valid,
        # even if it means slightly exceeding max_candidates. This is important because
        # scale=1.0 (no scaling) is often the safest option and should always be tried.
        if feasible_lower <= 1.0 <= self.scale_max + tol and 1.0 not in candidates:
            # Make room for scale=1.0 by removing the last candidate if necessary
            if len(candidates) >= max_candidates:
                candidates = candidates[:max_candidates - 1]
            append_candidate(1.0, force=True)

        # Ensure scale=1.0 is in the final candidates if it's valid
        # (it might have been truncated off if it was added last)
        if feasible_lower <= 1.0 <= self.scale_max + tol and 1.0 not in candidates[:max_candidates]:
            # Replace the last candidate with scale=1.0
            if len(candidates) >= max_candidates:
                candidates = candidates[:max_candidates - 1]
            else:
                candidates = candidates[:]
            append_candidate(1.0, force=True)
            candidates = candidates[:max_candidates]
        else:
            candidates = candidates[:max_candidates]

        return candidates

    def get_new_individual(self, parents):
        f = parents[0]
        indi = self.mutate(f)
        return _finalize_mutant(self, f, indi, "mutation: breathing")

    def mutate(self, atoms):
        N = len(atoms) if self.n_top is None else self.n_top
        slab = atoms[:len(atoms) - N]
        top = atoms[-N:]
        pos = top.get_positions()
        cm = np.average(pos, axis=0)
        num = top.get_atomic_numbers()
        cell = top.get_cell()
        pbc = top.get_pbc()

        self.last_attempt_count = 0
        for scale in self._candidate_scales(pos, num, slab):
            self.last_attempt_count += 1
            s = scale
            new_pos = cm + s * (pos - cm)
            # Check with proper PBC to avoid periodic image violations
            cand = Atoms(num, positions=new_pos, cell=cell, pbc=pbc)
            if atoms_too_close(cand, self.blmin):
                continue
            if self.test_dist_to_slab and len(slab) > 0:
                # Disable PBC for slab-candidate check to avoid periodic image artifacts
                slab_np = slab.copy()
                slab_np.pbc = [False, False, False]
                cand_np = cand.copy()
                cand_np.pbc = [False, False, False]
                if atoms_too_close_two_sets(slab_np, cand_np, self.blmin):
                    continue
            # Return with original PBC
            return slab + cand
        return None

# fmt: on
