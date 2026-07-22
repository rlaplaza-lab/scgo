# fmt: off

from __future__ import annotations

"""Mutation that applies random rotations to multi-atom moieties."""

import numpy as np
from ase_ga.offspring_creator import OffspringCreator
from ase_ga.utilities import (
    atoms_too_close,
    atoms_too_close_two_sets,
    gather_atoms_by_tag,
    get_rotation_matrix,
)

from scgo.ase_ga_patches.mutations._common import (
    _append_unique_unit_vector,
    _ensure_rng,
    _random_unit_vector,
)
from scgo.ase_ga_patches.mutations._finalize import _finalize_mutant
from scgo.initialization.steric_scoring import steric_deficit as _steric_deficit
from scgo.initialization.steric_scoring import (
    steric_deficit_two_sets as _steric_deficit_two_sets,
)
from scgo.system_types import SystemType, get_system_policy

__all__ = ["RotationalMutation"]


class RotationalMutation(OffspringCreator):
    """Mutates a candidate by applying random rotations
    to multi-atom moieties in the structure (atoms with
    the same tag are considered part of one such moiety).

    Only performs whole-molecule rotations, no internal
    rotations.

    For more information, see also:

      * `Zhu Q., Oganov A.R., Glass C.W., Stokes H.T,
        Acta Cryst. (2012), B68, 215-226.`__

        __ https://dx.doi.org/10.1107/S0108768112017466

    Parameters
    ----------
    blmin: dict
        The closest allowed interatomic distances on the form:
        {(Z, Z*): dist, ...}, where Z and Z* are atomic numbers.

    n_top: int or None
        The number of atoms to optimize (None = include all).

    fraction: float
        Fraction of the moieties to be rotated.

    tags: None or list of integers
        Specifies, respectively, whether all moieties or only those
        with matching tags are eligible for rotation.

    min_angle: float
        Minimal angle (in radians) for each rotation;
        should lie in the interval [0, pi].

    test_dist_to_slab: boolean
        Whether also the distances to the slab
        should be checked to satisfy the blmin.

    rng: Random number generator
        By default numpy.random.

    """

    def __init__(self, blmin, system_type: SystemType, n_top=None, fraction=0.33, tags=None,
                 min_angle=1.57, test_dist_to_slab=True, target_tags=None,
                 use_tags=False, rng=None, verbose=False, max_inner_attempts=24):
        rng = _ensure_rng(rng)
        OffspringCreator.__init__(self, verbose, rng=rng)
        self.blmin = blmin
        self.n_top = n_top
        self.fraction = fraction
        self.tags = tags
        self.min_angle = min_angle
        self.test_dist_to_slab = test_dist_to_slab
        self.target_tags = target_tags
        self.use_tags = use_tags
        self.system_type = system_type
        self._policy = get_system_policy(system_type)
        self.max_inner_attempts = max_inner_attempts
        self.last_attempt_count = 0
        self.descriptor = "RotationalMutation"
        self.min_inputs = 1

    def get_new_individual(self, parents):
        f = parents[0]

        indi = self.mutate(f)

        return _finalize_mutant(self, f, indi, "mutation: rotational")

    def _candidate_rotation_axes(self, positions):
        candidates = []
        if len(positions) == 2:
            bond = positions[1] - positions[0]
            bond /= np.linalg.norm(bond)
            for alt in (
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([0.0, 0.0, 1.0]),
            ):
                axis = np.cross(bond, alt)
                norm = np.linalg.norm(axis)
                if norm > 1e-12:
                    _append_unique_unit_vector(candidates, axis / norm)
        else:
            centered = positions - np.mean(positions, axis=0)
            if len(centered) > 1:
                covariance = np.dot(centered.T, centered)
                eigenvalues, eigenvectors = np.linalg.eigh(covariance)
                for index in np.argsort(eigenvalues):
                    _append_unique_unit_vector(candidates, eigenvectors[:, index])

        attempts = 0
        while len(candidates) < 3 and attempts < 20:
            _append_unique_unit_vector(candidates, _random_unit_vector(self.rng))
            attempts += 1

        return candidates[:3]

    def _candidate_angles(self):
        min_angle = self.min_angle
        mid_angle = 0.5 * (min_angle + np.pi)
        angles = [min_angle, mid_angle, np.pi]
        max_angles = max(1, min(int(self.max_inner_attempts), 4))
        angles = angles[:max_angles]
        if len(angles) < max_angles:
            angles.append(min_angle + (np.pi - min_angle) * self.rng.random())
        return angles

    def _rotation_configs(self, chosen_tags, indices, positions):
        axis_sets = {
            tag: self._candidate_rotation_axes(positions[indices[tag]])
            for tag in chosen_tags
        }
        angles = self._candidate_angles()
        configs = []

        for angle in angles:
            config = {}
            for tag in chosen_tags:
                axes = axis_sets[tag]
                if not axes:
                    continue
                config[tag] = (axes[0], angle)
            if len(config) == len(chosen_tags):
                configs.append(config)

        for axis_index in range(1, 3):
            for angle in angles:
                config = {}
                for tag in chosen_tags:
                    axes = axis_sets[tag]
                    if len(axes) <= axis_index:
                        continue
                    config[tag] = (axes[axis_index], angle)
                if len(config) == len(chosen_tags):
                    configs.append(config)

        max_configs = max(1, min(int(self.max_inner_attempts), 24))
        return configs[:max_configs]

    def _apply_rotation_config(self, positions, indices, config):
        new_positions = np.copy(positions)
        for tag, (axis, angle) in config.items():
            moiety = new_positions[indices[tag]]
            center = np.mean(moiety, axis=0)
            rotation = get_rotation_matrix(axis, angle)
            new_positions[indices[tag]] = np.dot(rotation, (moiety - center).T).T + center
        return new_positions

    def mutate(self, atoms):
        """Does the actual mutation."""
        N = len(atoms) if self.n_top is None else self.n_top
        slab = atoms[:len(atoms) - N]
        atoms = atoms[-N:]

        mutant = atoms.copy()
        gather_atoms_by_tag(mutant)
        pos = mutant.get_positions()
        tags = mutant.get_tags()
        numbers = mutant.get_atomic_numbers()

        # Determine which tags to target
        unique_tags = np.unique(tags)
        if self.target_tags is not None:
            target_tags_set = set(self.target_tags)
            unique_tags = np.array([t for t in unique_tags if t in target_tags_set])
            if len(unique_tags) == 0:
                return None

        eligible_tags = self.tags if self.tags is not None else unique_tags
        # Filter eligible_tags to only include tags we're targeting
        eligible_tags = [t for t in eligible_tags if t in unique_tags]

        indices = {}
        for tag in eligible_tags:
            hits = np.where(tags == tag)[0]
            if len(hits) > 1:
                indices[tag] = hits

        n_rot = int(np.ceil(len(indices) * self.fraction))
        if n_rot > 0 and len(indices) > 0:
            chosen_tags = self.rng.choice(
                list(indices.keys()),
                size=min(n_rot, len(indices)),
                replace=False,
            )
        else:
            return None

        configs = self._rotation_configs(chosen_tags, indices, pos)
        if not configs:
            return None

        ranked = []
        for config in configs:
            newpos = self._apply_rotation_config(pos, indices, config)
            score = _steric_deficit(newpos, numbers, self.blmin)
            if len(slab) > 0:
                score += _steric_deficit_two_sets(
                    newpos,
                    numbers,
                    slab.get_positions(),
                    slab.numbers,
                    self.blmin,
                )
            ranked.append((score, newpos))

        ranked.sort(key=lambda item: item[0])
        self.last_attempt_count = 0
        for _score, newpos in ranked:
            self.last_attempt_count += 1
            mutant.set_positions(newpos)
            if not self._policy.uses_surface:
                mutant.center()

            too_close = atoms_too_close(mutant, self.blmin, use_tags=self.use_tags)
            if not too_close and self.test_dist_to_slab:
                too_close = atoms_too_close_two_sets(slab, mutant, self.blmin)

            if not too_close:
                return slab + mutant

        return None

# fmt: on
