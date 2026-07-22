# fmt: off

from __future__ import annotations

"""Mutation that resolves steric clashes with bounded geometric sweeps."""

import numpy as np
from ase import Atoms
from ase_ga.offspring_creator import OffspringCreator
from ase_ga.utilities import atoms_too_close, atoms_too_close_two_sets

from scgo.ase_ga_patches.mutations._common import _ensure_rng, _random_unit_vector
from scgo.ase_ga_patches.mutations._finalize import _finalize_mutant
from scgo.initialization.steric_scoring import get_blmin_distance as _get_blmin_distance
from scgo.system_types import SystemType, get_system_policy

__all__ = ["OverlapReliefMutation"]


class OverlapReliefMutation(OffspringCreator):
    """Resolve steric clashes with bounded geometric sweeps.

    The operator accumulates pairwise displacements for atoms that violate
    ``blmin`` and applies a small exploratory jitter only after the repaired
    geometry is valid.
    """

    def __init__(
        self,
        blmin,
        n_top,
        system_type: SystemType,
        n_sweeps=4,
        jitter=0.02,
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
        top = atoms[-N:]
        positions = top.get_positions().copy()
        numbers = top.get_atomic_numbers()
        cell = top.get_cell()
        pbc = top.get_pbc()
        tags = top.get_tags()

        for _ in range(self.n_sweeps):
            displacements = np.zeros_like(positions)
            moved = False

            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    if self.use_tags and tags[i] == tags[j]:
                        continue
                    required = _get_blmin_distance(self.blmin, numbers[i], numbers[j])
                    vector = positions[j] - positions[i]
                    distance = np.linalg.norm(vector)
                    if distance + 1e-12 < required:
                        direction = (
                            vector / distance
                            if distance > 1e-12
                            else _random_unit_vector(self.rng)
                        )
                        shift = 0.5 * (required - distance + self.margin)
                        displacements[i] -= shift * direction
                        displacements[j] += shift * direction
                        moved = True

            if self.test_dist_to_slab and len(slab) > 0:
                slab_positions = slab.get_positions()
                slab_numbers = slab.get_atomic_numbers()
                for i in range(len(positions)):
                    for j in range(len(slab_positions)):
                        required = _get_blmin_distance(
                            self.blmin,
                            numbers[i],
                            slab_numbers[j],
                        )
                        vector = positions[i] - slab_positions[j]
                        distance = np.linalg.norm(vector)
                        if distance + 1e-12 < required:
                            direction = (
                                vector / distance
                                if distance > 1e-12
                                else np.array([0.0, 0.0, 1.0])
                            )
                            shift = (required - distance + self.margin) * direction
                            if self.use_tags:
                                select = np.where(tags == tags[i])[0]
                                displacements[select] += shift
                            else:
                                displacements[i] += shift
                            moved = True

            positions += displacements
            if not moved:
                break

        repaired_positions = positions.copy()
        for add_jitter in (True, False):
            trial_positions = repaired_positions.copy()
            if add_jitter and self.jitter > 0.0:
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

            candidate = Atoms(
                numbers,
                positions=trial_positions,
                cell=cell,
                pbc=pbc,
                tags=tags,
            )
            if not self._policy.uses_surface:
                candidate.center()
            if atoms_too_close(candidate, self.blmin):
                continue
            if (
                self.test_dist_to_slab
                and len(slab) > 0
                and atoms_too_close_two_sets(slab, candidate, self.blmin)
            ):
                continue
            return slab + candidate

        return None

# fmt: on
