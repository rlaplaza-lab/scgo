# fmt: off

from __future__ import annotations

"""Mirror mutation that reflects half of a cluster across a random plane."""

import numpy as np
from ase import Atoms
from ase_ga.offspring_creator import OffspringCreator
from ase_ga.utilities import atoms_too_close, atoms_too_close_two_sets

from scgo.ase_ga_patches.mutations._common import (
    _append_unique_unit_vector,
    _ensure_rng,
    _geometry_candidate_directions,
    _random_unit_vector,
)
from scgo.ase_ga_patches.mutations._finalize import _finalize_mutant
from scgo.initialization.steric_scoring import steric_deficit as _steric_deficit
from scgo.initialization.steric_scoring import (
    steric_deficit_two_sets as _steric_deficit_two_sets,
)
from scgo.system_types import SystemType, get_system_policy

__all__ = ["MirrorMutation"]


class MirrorMutation(OffspringCreator):
    """A mirror mutation, as described in
    TO BE PUBLISHED.

    This mutation mirrors half of the cluster in a
    randomly oriented cutting plane discarding the other half.

    Parameters
    ----------
    blmin: Dictionary defining the minimum allowed
        distance between atoms.

    n_top: Number of atoms the GA optimizes.

    reflect: Defines if the mirrored half is also reflected
        perpendicular to the mirroring plane.

    rng: Random number generator
        By default numpy.random.

    """

    def __init__(self, blmin, n_top, system_type: SystemType, reflect=True,
                 target_tags=None, rng=None, verbose=False, max_tries=12):
        rng = _ensure_rng(rng)
        OffspringCreator.__init__(self, verbose, rng=rng)
        self.blmin = blmin
        self.n_top = n_top
        self.max_tries = max_tries
        self.reflect = reflect
        self.target_tags = target_tags
        self.system_type = system_type
        self._policy = get_system_policy(system_type)
        self.last_attempt_count = 0

        self.descriptor = "MirrorMutation"
        self.min_inputs = 1

    def get_new_individual(self, parents):
        f = parents[0]

        indi = self.mutate(f)

        return _finalize_mutant(self, f, indi, "mutation: mirror")

    def _candidate_planes(self, positions, center_of_mass, slab):
        max_candidates = max(1, min(int(self.max_tries), 12))
        candidates = _geometry_candidate_directions(
            positions,
            center_of_mass,
            slab,
            self.rng,
            min(6, max_candidates),
        )
        outward = None
        if len(slab) > 0:
            outward = center_of_mass - np.mean(slab.get_positions(), axis=0)

        attempts = 0
        while len(candidates) < max_candidates and attempts < 100:
            axis = _random_unit_vector(self.rng)
            if outward is not None and np.dot(axis, outward) < 0.0:
                axis = -axis
            _append_unique_unit_vector(candidates, axis)
            attempts += 1

        return candidates[:max_candidates]

    def _build_mirror_top(self, num, pos, center_of_mass, plane, reflect):
        unique_types = list(set(num))
        nu = {u: sum(num == u) for u in unique_types}

        distances = [
            (index, float(np.dot(pos[index] - center_of_mass, plane)))
            for index in range(len(pos))
        ]
        distances.sort(key=lambda item: item[1])
        nu_taken = {}

        p_use = []
        n_use = []
        for index, _distance in distances:
            element = num[index]
            if element not in nu_taken:
                nu_taken[element] = 0
            if nu_taken[element] < nu[element] / 2.0:
                p_use.append(pos[index])
                n_use.append(element)
                nu_taken[element] += 1

        mirrored = []
        for point in p_use:
            mirrored_point = point - 2.0 * np.dot(point - center_of_mass, plane) * plane
            if reflect:
                mirrored_point = (
                    -mirrored_point
                    + 2.0 * center_of_mass
                    + 2.0 * plane * np.dot(mirrored_point - center_of_mass, plane)
                )
            mirrored.append(mirrored_point)

        n_use.extend(n_use)
        p_use.extend(mirrored)

        for element in nu:
            if nu[element] % 2 == 0:
                continue
            while n_use.count(element) > nu[element]:
                for index in range(int(len(n_use) / 2), len(n_use)):
                    if n_use[index] == element:
                        del p_use[index]
                        del n_use[index]
                        break
            assert n_use.count(element) == nu[element]

        for index in range(len(n_use)):
            if num[index] == n_use[index]:
                continue
            for swap_index in range(index + 1, len(n_use)):
                if n_use[swap_index] == num[index]:
                    n_use[index], n_use[swap_index] = n_use[swap_index], n_use[index]
                    p_use[index], p_use[swap_index] = p_use[swap_index], p_use[index]
                    break

        return Atoms(num, p_use)

    def mutate(self, atoms):
        """Do the mutation of the atoms input."""
        slab = atoms[0:len(atoms) - self.n_top]
        top = atoms[len(atoms) - self.n_top: len(atoms)]
        num = top.numbers
        pos = top.get_positions().copy()
        tags = top.get_tags() if hasattr(top, "get_tags") else np.arange(len(top))

        if self.target_tags is not None:
            target_mask = np.isin(tags, list(self.target_tags))
            if not np.any(target_mask):
                return None
            target_pos = pos[target_mask]
            target_num = num[target_mask]
            center_of_mass = np.average(target_pos, axis=0)
        else:
            target_mask = np.ones(len(top), dtype=bool)
            target_pos = pos
            target_num = num
            center_of_mass = np.average(pos, axis=0)

        reflect_options = [self.reflect]
        if not self.reflect:
            reflect_options.append(True)
        else:
            reflect_options.append(False)

        max_candidates = max(1, min(int(self.max_tries), 12))
        ranked_candidates = []
        for plane in self._candidate_planes(target_pos, center_of_mass, slab):
            for reflect in reflect_options:
                mirrored_top = self._build_mirror_top(
                    target_num,
                    target_pos,
                    center_of_mass,
                    plane,
                    reflect,
                )
                new_pos = pos.copy()
                new_pos[target_mask] = mirrored_top.get_positions()
                mutant = Atoms(num, new_pos)
                mutant.set_cell(slab.get_cell())
                mutant.set_pbc(slab.get_pbc())
                mutant.set_tags(tags)
                score = _steric_deficit(mutant.get_positions(), num, self.blmin)
                if len(slab) > 0:
                    score += _steric_deficit_two_sets(
                        mutant.get_positions(),
                        num,
                        slab.get_positions(),
                        slab.numbers,
                        self.blmin,
                    )
                ranked_candidates.append((score, mutant))

        ranked_candidates.sort(key=lambda item: item[0])
        ranked_candidates = ranked_candidates[:max_candidates]

        self.last_attempt_count = 0
        for _score, mutant in ranked_candidates:
            self.last_attempt_count += 1
            if atoms_too_close(mutant, self.blmin):
                continue
            if atoms_too_close_two_sets(slab, mutant, self.blmin):
                continue
            return slab + mutant

        self.last_attempt_count = len(ranked_candidates)
        return None

# fmt: on
