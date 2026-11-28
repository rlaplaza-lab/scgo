"""Mutation that swaps atom groups between inner and outer radial shells."""

from __future__ import annotations


import numpy as np
from ase import Atoms

from scgo.ase_ga_patches._tag_gather import (
    gather_atoms_by_tag,
    periodic_sheet_tag_to_skip,
)
from scgo.ase_ga_patches.mutations._common import (
    MutationOperator,
    _IDENTITY_ATOL,
    _accept_connected_candidate,
    _accept_connected_kwargs,
    _filter_unique_tags,
    _preserves_mobile_connectivity,
    _resolve_op_connectivity_factor,
)
from scgo.system_types import SystemType

__all__ = ["ShellSwapMutation"]


class ShellSwapMutation(MutationOperator):
    descriptor = "ShellSwapMutation"
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
        surface_normal_axis=2,
        rng=None,
        verbose=False,
    ):
        self._init_operator(
            verbose,
            rng,
            system_type,
            n_top=n_top,
            blmin=blmin,
            test_dist_to_slab=test_dist_to_slab,
            use_tags=use_tags,
            target_tags=target_tags,
            surface_normal_axis=int(surface_normal_axis),
        )
        self.inner_fraction = inner_fraction
        self.outer_fraction = outer_fraction
        self.max_pair_trials = max_pair_trials

    def mutate(self, atoms):
        N = len(atoms) if self.n_top is None else self.n_top
        slab = atoms[: len(atoms) - N]
        top = atoms[len(atoms) - N :].copy()
        if self.use_tags:
            gather_atoms_by_tag(
                top,
                skip_tag=periodic_sheet_tag_to_skip(self.system_type),
            )

        tags = top.get_tags() if self.use_tags else np.arange(N)
        positions = top.get_positions().copy()
        numbers = top.get_atomic_numbers()
        symbols = top.get_chemical_symbols()
        cell = top.get_cell()
        pbc = top.get_pbc()
        unique_tags = _filter_unique_tags(np.unique(tags), self.target_tags)
        if unique_tags is None:
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
        if len(slab) > 0:
            radial_distances = centers[:, self.surface_normal_axis]
        else:
            radial_center = np.mean(centers, axis=0)
            radial_distances = np.linalg.norm(centers - radial_center, axis=1)
        order = np.argsort(radial_distances)
        inner_count = max(
            1, min(len(order) - 1, int(np.ceil(len(order) * self.inner_fraction)))
        )
        outer_count = max(
            1, min(len(order) - 1, int(np.ceil(len(order) * self.outer_fraction)))
        )
        inner_groups = order[:inner_count]
        outer_groups = order[-outer_count:]

        candidate_pairs = []
        for inner_idx in inner_groups:
            for outer_idx in outer_groups:
                if inner_idx == outer_idx:
                    continue
                if group_symbols[inner_idx] == group_symbols[outer_idx]:
                    continue
                radial_gap = abs(
                    radial_distances[inner_idx] - radial_distances[outer_idx]
                )
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
            # Coincident centers translate nothing. Try the next pair.
            if float(np.linalg.norm(right_center - left_center)) <= _IDENTITY_ATOL:
                continue
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
                if not _preserves_mobile_connectivity(
                    top,
                    candidate,
                    use_mic=bool(self._policy.uses_surface),
                    connectivity_factor=_resolve_op_connectivity_factor(self),
                ):
                    continue
                return slab + candidate
            accepted = _accept_connected_candidate(
                top,
                candidate,
                **_accept_connected_kwargs(self, slab),
            )
            if accepted is None:
                continue
            return slab + accepted

        return None
