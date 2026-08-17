"""Mutation that rotates the whole mobile region about the surface normal."""

# fmt: off

from __future__ import annotations


import numpy as np
from ase import Atoms
from ase_ga.offspring_creator import OffspringCreator
from ase_ga.utilities import (
    atoms_too_close,
    atoms_too_close_two_sets,
    get_rotation_matrix,
)

from scgo.ase_ga_patches.mutations._common import (
    _ensure_rng,
    _preserves_mobile_connectivity,
    _reanchor_mobile_to_slab,
    _resolve_op_connectivity_factor,
)
from scgo.ase_ga_patches.mutations._finalize import _finalize_mutant
from scgo.initialization.steric_scoring import steric_deficit as _steric_deficit
from scgo.initialization.steric_scoring import (
    steric_deficit_two_sets as _steric_deficit_two_sets,
)
from scgo.system_types import SystemType, get_system_policy

__all__ = ["InPlaneRotateMutation"]


class InPlaneRotateMutation(OffspringCreator):
    """Rigidly rotate all mobile atoms about the surface normal through the CoM."""

    def __init__(
        self,
        blmin,
        n_top,
        system_type: SystemType,
        surface_normal_axis: int = 2,
        min_angle: float = 0.25,
        rng=None,
        verbose: bool = False,
        max_inner_attempts: int = 24,
    ):
        rng = _ensure_rng(rng)
        OffspringCreator.__init__(self, verbose, rng=rng)
        self.blmin = blmin
        self.n_top = n_top
        self.system_type = system_type
        self._policy = get_system_policy(system_type)
        self.surface_normal_axis = surface_normal_axis
        self.min_angle = min_angle
        self.max_inner_attempts = max_inner_attempts
        self.last_attempt_count = 0
        self.test_dist_to_slab = True
        self.descriptor = "InPlaneRotateMutation"
        self.min_inputs = 1

    def _candidate_angles(self) -> list[float]:
        min_angle = self.min_angle
        small = max(0.10, 0.25 * min_angle)
        angles = [
            small,
            min_angle,
            0.5 * (min_angle + np.pi),
            np.pi,
            min_angle + (np.pi - min_angle) * self.rng.random(),
        ]
        max_angles = max(1, min(int(self.max_inner_attempts), len(angles)))
        return angles[:max_angles]

    def get_new_individual(self, parents):
        parent = parents[0]
        mutant = self.mutate(parent)
        return _finalize_mutant(self, parent, mutant, "mutation: in_plane_rotate")

    def mutate(self, atoms):
        n_top = int(self.n_top)
        slab = atoms[: len(atoms) - n_top]
        mutant = atoms[-n_top:].copy()
        pos = mutant.get_positions()
        numbers = mutant.get_atomic_numbers()
        center = np.mean(pos, axis=0)

        axis = np.zeros(3, dtype=float)
        axis[self.surface_normal_axis] = 1.0
        in_plane = [i for i in range(3) if i != self.surface_normal_axis]
        rescue_offset = 0.5 * min(self.blmin.values())
        use_mic = bool(self._policy.uses_surface)

        ranked: list[tuple[float, np.ndarray]] = []
        for angle in self._candidate_angles():
            rotation = get_rotation_matrix(axis, angle)
            newpos = np.dot(rotation, (pos - center).T).T + center
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
            if self._policy.uses_surface:
                newpos = _reanchor_mobile_to_slab(
                    mutant,
                    Atoms(
                        numbers=numbers,
                        positions=newpos,
                        cell=mutant.get_cell(),
                        pbc=mutant.get_pbc(),
                        tags=mutant.get_tags(),
                    ),
                    slab,
                    self.surface_normal_axis,
                ).get_positions()
            mutant.set_positions(newpos)

            too_close = atoms_too_close(mutant, self.blmin, use_tags=False)
            if not too_close and self.test_dist_to_slab and len(slab) > 0:
                too_close = atoms_too_close_two_sets(slab, mutant, self.blmin)
            if too_close:
                continue
            if _preserves_mobile_connectivity(
                atoms[-n_top:],
                mutant,
                use_mic=use_mic,
                connectivity_factor=_resolve_op_connectivity_factor(self),
            ):
                return slab + mutant

            if self._policy.uses_surface:
                for dx in (0.0, rescue_offset, -rescue_offset):
                    for dy in (0.0, rescue_offset, -rescue_offset):
                        if dx == 0.0 and dy == 0.0:
                            continue
                        rescue = newpos.copy()
                        rescue[:, in_plane[0]] += dx
                        rescue[:, in_plane[1]] += dy
                        mutant.set_positions(rescue)
                        too_close = atoms_too_close(mutant, self.blmin, use_tags=False)
                        if not too_close and self.test_dist_to_slab and len(slab) > 0:
                            too_close = atoms_too_close_two_sets(
                                slab, mutant, self.blmin
                            )
                        if (
                            not too_close
                            and _preserves_mobile_connectivity(
                                atoms[-n_top:],
                                mutant,
                                use_mic=use_mic,
                                connectivity_factor=_resolve_op_connectivity_factor(
                                    self
                                ),
                            )
                        ):
                            return slab + mutant

        return None

# fmt: on
