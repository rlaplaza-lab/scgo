"""Mutation that uniformly scales atom positions about the center of positions."""

from __future__ import annotations


import numpy as np
from ase import Atoms
from scipy.spatial.distance import pdist
from scgo.ase_ga_patches.mutations._common import (
    MutationOperator,
    _accept_connected_candidate,
    _accept_connected_kwargs,
    _prepare_mobile_target,
    _reanchor_mobile_to_slab,
)
from scgo.initialization.steric_scoring import _blmin_matrix
from scgo.system_types import SystemType

__all__ = ["BreathingMutation"]


class BreathingMutation(MutationOperator):
    descriptor = "BreathingMutation"
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

    def __init__(
        self,
        blmin,
        n_top,
        system_type: SystemType,
        scale_min=0.9,
        scale_max=1.1,
        test_dist_to_slab=True,
        target_tags=None,
        rng=None,
        verbose=False,
        max_inner_attempts=1000,
        surface_normal_axis=2,
        min_scale_delta=0.05,
    ):
        self._init_operator(
            verbose,
            rng,
            system_type,
            n_top=n_top,
            blmin=blmin,
            test_dist_to_slab=test_dist_to_slab,
            target_tags=target_tags,
            surface_normal_axis=surface_normal_axis,
        )
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.max_inner_attempts = max_inner_attempts
        self.min_scale_delta = float(min_scale_delta)
        self.last_attempt_count = 0

    def _minimum_feasible_scale(self, positions, atomic_numbers):
        n_atoms = len(positions)
        if n_atoms <= 1:
            return self.scale_min

        distances = pdist(positions)
        if np.any(distances <= 1e-12):
            return np.inf

        blmin_matrix = _blmin_matrix(atomic_numbers, self.blmin)
        required_condensed = blmin_matrix[np.triu_indices(n_atoms, k=1)]

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

    def mutate(self, atoms):
        prepared = _prepare_mobile_target(self, atoms)
        if prepared is None:
            return None
        slab, top, pos, num, cell, pbc, tags, mask = prepared

        cm = np.average(pos[mask], axis=0)

        self.last_attempt_count = 0
        for scale in self._candidate_scales(pos[mask], num[mask], slab):
            self.last_attempt_count += 1
            if abs(float(scale) - 1.0) < self.min_scale_delta:
                continue
            s = scale
            new_pos = pos.copy()
            new_pos[mask] = cm + s * (pos[mask] - cm)
            cand = Atoms(num, positions=new_pos, cell=cell, pbc=pbc, tags=tags)
            if self._policy.uses_surface:
                cand = _reanchor_mobile_to_slab(
                    top, cand, slab, self.surface_normal_axis
                )
            accepted = _accept_connected_candidate(
                top, cand, **_accept_connected_kwargs(self, slab)
            )
            if accepted is None:
                continue
            return slab + accepted
        return None
