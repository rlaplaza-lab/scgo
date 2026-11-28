"""Mutation that rotates the whole mobile region about the surface normal."""

from __future__ import annotations


import numpy as np
from ase import Atoms
from ase_ga.utilities import get_rotation_matrix

from scgo.ase_ga_patches.mutations._common import (
    _mutation_steric_score,
    MutationOperator,
    _accept_connected_candidate,
    _accept_connected_kwargs,
    _reanchor_mobile_to_slab,
)
from scgo.system_types import SystemType

__all__ = ["InPlaneRotateMutation"]


class InPlaneRotateMutation(MutationOperator):
    descriptor = "InPlaneRotateMutation"
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
        self._init_operator(
            verbose,
            rng,
            system_type,
            n_top=n_top,
            blmin=blmin,
            surface_normal_axis=surface_normal_axis,
        )
        self.min_angle = min_angle
        self.max_inner_attempts = max_inner_attempts
        self.last_attempt_count = 0

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
        accept_kwargs = _accept_connected_kwargs(self, slab)

        ranked: list[tuple[float, np.ndarray]] = []
        for angle in self._candidate_angles():
            rotation = get_rotation_matrix(axis, angle)
            newpos = np.dot(rotation, (pos - center).T).T + center
            score = _mutation_steric_score(newpos, numbers, slab, self.blmin)
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

            accepted = _accept_connected_candidate(
                atoms[-n_top:], mutant, **accept_kwargs
            )
            if accepted is not None:
                return slab + accepted

            if self._policy.uses_surface:
                for dx in (0.0, rescue_offset, -rescue_offset):
                    for dy in (0.0, rescue_offset, -rescue_offset):
                        if dx == 0.0 and dy == 0.0:
                            continue
                        rescue = newpos.copy()
                        rescue[:, in_plane[0]] += dx
                        rescue[:, in_plane[1]] += dy
                        mutant.set_positions(rescue)
                        accepted = _accept_connected_candidate(
                            atoms[-n_top:], mutant, **accept_kwargs
                        )
                        if accepted is not None:
                            return slab + accepted

        return None
