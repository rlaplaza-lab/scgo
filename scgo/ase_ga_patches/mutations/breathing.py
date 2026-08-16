"""Mutation that uniformly scales atom positions about the center of positions."""

# fmt: off

from __future__ import annotations


import numpy as np
from ase import Atoms
from ase_ga.offspring_creator import OffspringCreator
from ase_ga.utilities import atoms_too_close, atoms_too_close_two_sets

from scgo.ase_ga_patches.mutations._common import (
    _IDENTITY_ATOL,
    _ensure_rng,
    _preserves_mobile_connectivity,
    _resolve_op_connectivity_factor,
    _reanchor_mobile_to_slab,
)
from scgo.ase_ga_patches.mutations._finalize import _finalize_mutant
from scgo.initialization.steric_scoring import _blmin_matrix
from scgo.system_types import SystemType, get_system_policy

__all__ = ["BreathingMutation"]


class BreathingMutation(OffspringCreator):
    """Uniformly scales all atom positions relative to the center of positions.

    Candidate scale factors are enumerated deterministically within
    ``[scale_min, scale_max]``, always including 1.0 when it is valid, and the
    first candidate that violates no *blmin* pair is accepted. When no scale in
    that window can clear *blmin*, the smallest relieving uniform expansion is
    used even if it exceeds ``scale_max``.

    Parameters
    ----------
    blmin : dict
        Minimum allowed interatomic distances.
    n_top : int
        Number of atoms optimized by the GA.
    scale_min, scale_max : float
        Bounds for the candidate scale factors.
    test_dist_to_slab : bool
        Also check distances to slab atoms.
    rng : numpy.random.Generator or None
        Random number generator.
    max_inner_attempts : int
        Upper bound on the number of candidate scales per call (capped at 8).
    """

    def __init__(self, blmin, n_top, system_type: SystemType, scale_min=0.9, scale_max=1.1,
                 test_dist_to_slab=True, target_tags=None, rng=None, verbose=False,
                 max_inner_attempts=1000, surface_normal_axis=2):
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
        self.surface_normal_axis = surface_normal_axis
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

        # Compute pairwise blmin requirements (condensed upper triangle)
        blmin_matrix = _blmin_matrix(atomic_numbers, self.blmin)
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
        # Always include unit scale (1.0) when valid, since an already-optimal
        # cluster should not be forced to scale.
        if feasible_lower <= 1.0 <= self.scale_max + tol or allow_unit_scale:
            append_candidate(1.0, force=True)

        return candidates[:max_candidates]

    def get_new_individual(self, parents):
        f = parents[0]
        indi = self.mutate(f)
        return _finalize_mutant(self, f, indi, "mutation: breathing")

    def mutate(self, atoms):
        N = len(atoms) if self.n_top is None else self.n_top
        slab = atoms[:len(atoms) - N]
        top = atoms[len(atoms) - N:]
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

        # Only the targeted tag groups are scaled; the rest stay put.
        mask = np.isin(tags, unique_tags)
        if not np.any(mask):
            return None

        cm = np.average(pos[mask], axis=0)
        use_mic = bool(self._policy.uses_surface)

        self.last_attempt_count = 0
        for scale in self._candidate_scales(pos[mask], num[mask], slab):
            self.last_attempt_count += 1
            if abs(float(scale) - 1.0) <= _IDENTITY_ATOL:
                continue
            s = scale
            new_pos = pos.copy()
            new_pos[mask] = cm + s * (pos[mask] - cm)
            cand = Atoms(num, positions=new_pos, cell=cell, pbc=pbc, tags=tags)
            if self._policy.uses_surface:
                cand = _reanchor_mobile_to_slab(
                    top, cand, slab, self.surface_normal_axis)
            if atoms_too_close(cand, self.blmin):
                continue
            # Keep slab PBC (typically in-plane only) so lateral MIC clashes
            # are rejected the same way as other surface mutations.
            if (
                self.test_dist_to_slab
                and len(slab) > 0
                and atoms_too_close_two_sets(slab, cand, self.blmin)
            ):
                continue
            if not _preserves_mobile_connectivity(
                top,
                cand,
                use_mic=use_mic,
                connectivity_factor=_resolve_op_connectivity_factor(self),
            ):
                continue
            return slab + cand
        return None

# fmt: on
