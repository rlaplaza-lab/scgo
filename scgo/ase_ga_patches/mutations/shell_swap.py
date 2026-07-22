# fmt: off

from __future__ import annotations

"""Mutation that swaps atom groups between inner and outer radial shells."""

import numpy as np
from ase import Atoms
from ase_ga.offspring_creator import OffspringCreator
from ase_ga.utilities import (
    atoms_too_close,
    atoms_too_close_two_sets,
    gather_atoms_by_tag,
)

from scgo.ase_ga_patches.mutations._common import _ensure_rng
from scgo.ase_ga_patches.mutations._finalize import _finalize_mutant
from scgo.system_types import SystemType, get_system_policy

__all__ = ["ShellSwapMutation"]


class ShellSwapMutation(OffspringCreator):
    """Swap atom groups between inner and outer radial shells.

    This targets alloy ordering directly by preferring swaps between groups with
    different chemical signatures and large radial separation.
    """

    def __init__(
        self,
        n_top,
        system_type: SystemType,
        inner_fraction=0.33,
        outer_fraction=0.33,
        test_dist_to_slab=True,
        use_tags=False,
        target_tags=None,
        blmin=None,
        max_pair_trials=12,
        rng=None,
        verbose=False,
    ):
        rng = _ensure_rng(rng)
        OffspringCreator.__init__(self, verbose, rng=rng)
        self.n_top = n_top
        self.inner_fraction = inner_fraction
        self.outer_fraction = outer_fraction
        self.test_dist_to_slab = test_dist_to_slab
        self.use_tags = use_tags
        self.target_tags = target_tags
        self.blmin = blmin
        self.max_pair_trials = max_pair_trials
        self.system_type = system_type
        self._policy = get_system_policy(system_type)

        self.descriptor = "ShellSwapMutation"
        self.min_inputs = 1

    def get_new_individual(self, parents):
        f = parents[0]

        indi = self.mutate(f)

        return _finalize_mutant(self, f, indi, "mutation: shell_swap")

    def mutate(self, atoms):
        N = len(atoms) if self.n_top is None else self.n_top
        slab = atoms[: len(atoms) - N]
        top = atoms[-N:].copy()
        if self.use_tags:
            gather_atoms_by_tag(top)

        tags = top.get_tags() if self.use_tags else np.arange(N)
        positions = top.get_positions().copy()
        numbers = top.get_atomic_numbers()
        symbols = top.get_chemical_symbols()
        cell = top.get_cell()
        pbc = top.get_pbc()
        unique_tags = np.unique(tags)

        # Determine which tags to target
        if self.target_tags is not None:
            target_tags_set = set(self.target_tags)
            unique_tags = np.array([t for t in unique_tags if t in target_tags_set])
            if len(unique_tags) == 0:
                return None

        group_indices = []
        group_symbols = []
        group_centers = []
        for tag in unique_tags:
            indices = np.where(tags == tag)[0]
            group_indices.append(indices)
            group_symbols.append("".join(symbols[idx] for idx in indices))
            group_centers.append(np.mean(positions[indices], axis=0))

        if len(np.unique(group_symbols)) <= 1:
            return None

        centers = np.asarray(group_centers)
        radial_center = np.mean(centers, axis=0)
        radial_distances = np.linalg.norm(centers - radial_center, axis=1)
        order = np.argsort(radial_distances)
        inner_count = max(1, min(len(order) - 1, int(np.ceil(len(order) * self.inner_fraction))))
        outer_count = max(1, min(len(order) - 1, int(np.ceil(len(order) * self.outer_fraction))))
        inner_groups = order[:inner_count]
        outer_groups = order[-outer_count:]

        candidate_pairs = []
        for inner_idx in inner_groups:
            for outer_idx in outer_groups:
                if inner_idx == outer_idx:
                    continue
                if group_symbols[inner_idx] == group_symbols[outer_idx]:
                    continue
                radial_gap = abs(radial_distances[inner_idx] - radial_distances[outer_idx])
                candidate_pairs.append((radial_gap, inner_idx, outer_idx))

        if not candidate_pairs:
            for left in range(len(unique_tags)):
                for right in range(left + 1, len(unique_tags)):
                    if group_symbols[left] == group_symbols[right]:
                        continue
                    radial_gap = abs(radial_distances[left] - radial_distances[right])
                    candidate_pairs.append((radial_gap, left, right))

        if not candidate_pairs:
            return None

        candidate_pairs.sort(key=lambda item: item[0], reverse=True)
        n_trials = min(self.max_pair_trials, len(candidate_pairs))
        pair_order = self.rng.permutation(n_trials) if n_trials > 1 else np.array([0])

        for pair_idx in pair_order:
            _, left, right = candidate_pairs[pair_idx]
            new_positions = positions.copy()
            left_indices = group_indices[left]
            right_indices = group_indices[right]
            left_center = np.mean(new_positions[left_indices], axis=0)
            right_center = np.mean(new_positions[right_indices], axis=0)
            new_positions[left_indices] += right_center - left_center
            new_positions[right_indices] += left_center - right_center

            candidate = Atoms(
                numbers,
                positions=new_positions,
                cell=cell,
                pbc=pbc,
                tags=tags,
            )
            if self.blmin is None:
                return slab + candidate
            if atoms_too_close(candidate, self.blmin, use_tags=self.use_tags):
                continue
            if (
                self.test_dist_to_slab
                and len(slab) > 0
                and atoms_too_close_two_sets(candidate, slab, self.blmin)
            ):
                continue
            return slab + candidate

        return None

# fmt: on
