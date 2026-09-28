"""Mutation that resolves steric clashes with bounded geometric sweeps."""

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
    _random_unit_vector,
)
from scgo.ase_ga_patches.mutations._finalize import _finalize_mutant
from scgo.initialization.steric_scoring import _blmin_matrix
from scgo.system_types import SystemType, get_system_policy

__all__ = ["OverlapReliefMutation"]


def _clash_unit_directions(vectors, distances, fallback):
    directions = np.zeros_like(vectors)
    nonzero = distances > 1e-12
    if np.any(nonzero):
        directions[nonzero] = vectors[nonzero] / distances[nonzero, None]
    missing = np.flatnonzero(~nonzero)
    if callable(fallback):
        for flat_k in missing:
            directions[flat_k] = fallback()
    elif missing.size:
        directions[missing] = fallback
    return directions


class OverlapReliefMutation(OffspringCreator):
    """Resolve steric clashes with bounded geometric sweeps.

    The operator accumulates pairwise displacements for atoms that violate
    ``blmin``. The repaired geometry is tried with exploratory jitter; a
    jitter-free repaired geometry is accepted only when the repair actually
    moved atoms. An unchanged parent is never returned.
    """

    def __init__(
        self,
        blmin,
        n_top,
        system_type: SystemType,
        n_sweeps=4,
        jitter=0.02,
        n_jitter_tries=32,
        margin=0.04,
        test_dist_to_slab=True,
        use_tags=False,
        rng=None,
        verbose=False,
    ):
        rng = _ensure_rng(rng)
        OffspringCreator.__init__(self, verbose, rng=rng)
        self.blmin = blmin
        self.n_top = n_top
        self.n_sweeps = n_sweeps
        self.jitter = jitter
        self.n_jitter_tries = n_jitter_tries
        self.margin = margin
        self.test_dist_to_slab = test_dist_to_slab
        self.use_tags = use_tags
        self.system_type = system_type
        self._policy = get_system_policy(system_type)

        self.descriptor = "OverlapReliefMutation"
        self.min_inputs = 1

    def get_new_individual(self, parents):
        f = parents[0]

        indi = self.mutate(f)

        return _finalize_mutant(self, f, indi, "mutation: overlap_relief")

    def mutate(self, atoms):
        N = len(atoms) if self.n_top is None else self.n_top
        slab = atoms[: len(atoms) - N]
        top = atoms[len(atoms) - N:]
        positions = top.get_positions().copy()
        numbers = top.get_atomic_numbers()
        cell = top.get_cell()
        pbc = top.get_pbc()
        tags = top.get_tags()

        # Precompute the pairwise blmin threshold matrices once (constant across
        # sweeps): within-set (symmetric) and slab cross-set.
        req_matrix = _blmin_matrix(numbers, self.blmin)
        slab_numbers = (
            slab.get_atomic_numbers() if len(slab) > 0 else np.array([], dtype=int)
        )
        req_cross = _blmin_matrix(numbers, self.blmin, slab_numbers)

        for _ in range(self.n_sweeps):
            displacements = np.zeros_like(positions)
            moved = False
            n_mobile = len(positions)

            if n_mobile >= 2:
                iu, ju = np.triu_indices(n_mobile, k=1)
                if self.use_tags:
                    keep = tags[iu] != tags[ju]
                    iu = iu[keep]
                    ju = ju[keep]
                if iu.size > 0:
                    vectors = positions[ju] - positions[iu]
                    distances = np.linalg.norm(vectors, axis=1)
                    required = req_matrix[iu, ju]
                    clash = distances + 1e-12 < required
                    if np.any(clash):
                        moved = True
                        clash_i = iu[clash]
                        clash_j = ju[clash]
                        clash_d = distances[clash]
                        clash_req = required[clash]
                        directions = _clash_unit_directions(
                            vectors[clash],
                            clash_d,
                            lambda: _random_unit_vector(self.rng),
                        )
                        shifts = (
                            0.5 * (clash_req - clash_d + self.margin)
                        )[:, None] * directions
                        np.add.at(displacements, clash_i, -shifts)
                        np.add.at(displacements, clash_j, shifts)

            if self.test_dist_to_slab and len(slab) > 0:
                slab_positions = slab.get_positions()
                vectors = positions[:, None, :] - slab_positions[None, :, :]
                distances = np.linalg.norm(vectors, axis=2)
                clash = distances + 1e-12 < req_cross
                if np.any(clash):
                    moved = True
                    ii, jj = np.nonzero(clash)
                    clash_d = distances[ii, jj]
                    clash_req = req_cross[ii, jj]
                    directions = _clash_unit_directions(
                        vectors[ii, jj],
                        clash_d,
                        np.array([0.0, 0.0, 1.0]),
                    )
                    shifts = (clash_req - clash_d + self.margin)[:, None] * directions
                    if self.use_tags:
                        for atom_i, shift in zip(ii, shifts, strict=True):
                            select = np.where(tags == tags[atom_i])[0]
                            displacements[select] += shift
                    else:
                        np.add.at(displacements, ii, shifts)

            positions += displacements
            if not moved:
                break

        original_positions = top.get_positions()
        repaired_positions = positions.copy()
        repair_changed = not np.allclose(
            repaired_positions, original_positions, atol=_IDENTITY_ATOL
        )
        use_mic = bool(self._policy.uses_surface)
        n_jitter_tries = max(0, int(self.n_jitter_tries)) if self.jitter > 0.0 else 0

        def _accept(trial_positions):
            if np.allclose(trial_positions, original_positions, atol=_IDENTITY_ATOL):
                return None
            candidate = Atoms(
                numbers,
                positions=trial_positions,
                cell=cell,
                pbc=pbc,
                tags=tags,
            )
            if not self._policy.uses_surface:
                candidate.center()
            if atoms_too_close(candidate, self.blmin, use_tags=self.use_tags):
                return None
            if (
                self.test_dist_to_slab
                and len(slab) > 0
                and atoms_too_close_two_sets(slab, candidate, self.blmin)
            ):
                return None
            if not _preserves_mobile_connectivity(
                top,
                candidate,
                use_mic=use_mic,
                connectivity_factor=_resolve_op_connectivity_factor(self),
            ):
                return None
            return slab + candidate

        for _ in range(n_jitter_tries):
            trial_positions = repaired_positions.copy()
            if self.use_tags:
                for tag in np.unique(tags):
                    select = np.where(tags == tag)[0]
                    trial_positions[select] += self.rng.normal(
                        0.0,
                        self.jitter,
                        size=(1, 3),
                    )
            else:
                trial_positions += self.rng.normal(
                    0.0,
                    self.jitter,
                    size=trial_positions.shape,
                )
            accepted = _accept(trial_positions)
            if accepted is not None:
                return accepted

        # Packed untagged clusters often reject all-atom jitter; displace one atom.
        if n_jitter_tries > 0 and not self.use_tags and len(repaired_positions) > 0:
            for _ in range(n_jitter_tries):
                trial_positions = repaired_positions.copy()
                idx = int(self.rng.integers(len(trial_positions)))
                trial_positions[idx] += self.rng.normal(0.0, self.jitter, size=3)
                accepted = _accept(trial_positions)
                if accepted is not None:
                    return accepted

        if repair_changed:
            accepted = _accept(repaired_positions)
            if accepted is not None:
                return accepted
        return None

# fmt: on
