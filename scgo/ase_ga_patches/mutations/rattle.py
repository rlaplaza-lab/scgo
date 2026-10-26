"""Rattle-style mutations that perturb atomic positions locally."""

# fmt: off

from __future__ import annotations


import numpy as np
from ase import Atoms
from ase_ga.offspring_creator import OffspringCreator

from scgo.ase_ga_patches.mutations._common import (
    _accept_connected_candidate,
    _accept_connected_kwargs,
    _disconnection_gap,
    _ensure_rng,
    _filter_unique_tags,
    _mobile_clashes,
    _preserves_mobile_connectivity,
    _reconnect_mobile_components,
    _reanchor_mobile_to_slab,
)
from scgo.ase_ga_patches.mutations._finalize import _finalize_mutant
from scgo.system_types import SystemType, get_system_policy

# Eight full-strength tries, then eight half-strength single-group tries.
_STRENGTH_SCALES = (1.0,) * 8 + (0.5,) * 8
_KICK_TRIES = 8
_KICK_MIN_DISPLACEMENT = 0.8
# Half-strength draws can land near zero. Stretch those to a real step.
_HALF_STEP_FLOOR_FRACTION = 0.5

__all__ = ["AnisotropicRattleMutation", "RattleMutation"]


def _ensure_step_length(step, floor):
    """Stretch a near-zero draw so a half-strength try is a real move."""
    length = float(np.linalg.norm(step))
    minimum = float(floor)
    if length >= minimum:
        return step
    if length < 1e-12:
        step = np.array([1.0, 0.0, 0.0])
        length = 1.0
    return step * (minimum / length)


def _tags_to_move(rng, unique_tags, proportion, *, single_group):
    """One guaranteed group, plus a fraction of the rest at full strength."""
    guaranteed = int(rng.integers(len(unique_tags)))
    selected = []
    for idx, tag in enumerate(unique_tags):
        if idx == guaranteed or (not single_group and rng.random() < proportion):
            selected.append(tag)
    return selected


def _apply_group_step(rng, positions, indices, step, *, break_rigid):
    """Translate ``indices`` by ``step``.

    On a gas cluster, one step shared by every atom is undone by centering.
    Move a proper subset so the internal distances change.
    """
    selected = np.asarray(indices, dtype=int)
    if break_rigid and len(selected) == len(positions):
        n_atoms = len(positions)
        if n_atoms < 2:
            return
        count = int(rng.integers(1, n_atoms))
        selected = rng.choice(n_atoms, size=count, replace=False)
    positions[selected] += step


def _consider_local_candidate(operator, parent, candidate, slab, best):
    """Accept a connected move, or keep the clash-free candidate closest to bonding.

    ``best`` is ``(gap, atoms)`` or ``None``. Repair happens once, after the
    attempt loop, so a disconnected sample does not pay for a pull.
    """
    kwargs = _accept_connected_kwargs(operator, slab)
    if _mobile_clashes(
        candidate,
        blmin=kwargs["blmin"],
        slab=slab,
        test_dist_to_slab=kwargs["test_dist_to_slab"],
        use_tags=kwargs["use_tags"],
    ):
        return None, best
    if _preserves_mobile_connectivity(
        parent,
        candidate,
        use_mic=kwargs["use_mic"],
        connectivity_factor=kwargs["connectivity_factor"],
    ):
        return candidate, best
    gap = _disconnection_gap(
        candidate,
        use_mic=kwargs["use_mic"],
        connectivity_factor=kwargs["connectivity_factor"],
    )
    if best is None or gap < best[0]:
        best = (gap, candidate.copy())
    return None, best


def _repair_best(operator, parent, slab, best):
    if best is None:
        return None
    kwargs = _accept_connected_kwargs(operator, slab)
    repaired = _reconnect_mobile_components(
        best[1],
        use_mic=kwargs["use_mic"],
        connectivity_factor=kwargs["connectivity_factor"],
    )
    return _accept_connected_candidate(parent, repaired, **kwargs)


def _as_offspring(operator, slab, mobile):
    mutant = slab + mobile
    if not operator._policy.uses_surface:
        mutant.center()
    return mutant


class RattleMutation(OffspringCreator):
    """An implementation of the rattle mutation as described in:

    R.L. Johnston Dalton Transactions, Vol. 22,
    No. 22. (2003), pp. 4193-4207

    Parameters
    ----------
    blmin: Dictionary defining the minimum distance between atoms
        after the rattle.

    n_top: Number of atoms optimized by the GA.

    rattle_strength: Strength with which the atoms are moved; each
        Cartesian component is displaced uniformly in
        [-rattle_strength, rattle_strength).

    rattle_prop: The probability with which each atom (or each tag group,
        when use_tags is True) is rattled. One group is always rattled
        per attempt, regardless of this probability.

    test_dist_to_slab: whether to also make sure that the distances
        between the atoms and the slab satisfy the blmin.

    use_tags: if True, the atomic tags will be used to preserve
        molecular identity. Same-tag atoms will then be
        displaced collectively, so that the internal
        geometry is preserved.

    system_type: System type (e.g., "gas_cluster", "surface_cluster",
        "gas_cluster_adsorbate", "surface_cluster_adsorbate").
        Used to ensure physical validity of mutations.

    rng: Random number generator
        Must be an instance of ``np.random.Generator`` or ``None``.

    verbose: bool
        If True, print verbose output.

    """

    def __init__(self, blmin, n_top, system_type: SystemType, rattle_strength=0.8,
                 rattle_prop=0.4, test_dist_to_slab=True, use_tags=False,
                 target_tags=None, verbose=False, rng=None,
                 surface_normal_axis=2):
        rng = _ensure_rng(rng)
        OffspringCreator.__init__(self, verbose, rng=rng)
        self.blmin = blmin
        self.n_top = n_top
        self.rattle_strength = rattle_strength
        self.rattle_prop = rattle_prop
        self.test_dist_to_slab = test_dist_to_slab
        self.use_tags = use_tags
        self.target_tags = target_tags
        self.system_type = system_type
        self.surface_normal_axis = surface_normal_axis
        self._policy = get_system_policy(system_type)

        self.descriptor = "RattleMutation"
        self.min_inputs = 1
        self.last_attempt_count = 0

    def get_new_individual(self, parents):
        """Generates a new individual by applying the rattle mutation to a parent.

        Args:
            parents: A list containing the parent Atoms object.

        Returns:
            A tuple containing the new Atoms object and a description of the mutation.

        """
        f = parents[0]

        indi = self.mutate(f)

        return _finalize_mutant(self, f, indi, "mutation: rattle")

    def mutate(self, atoms):
        """Applies the rattle mutation to the given Atoms object.

        Args:
            atoms: The Atoms object to be mutated.

        Returns:
            A new Atoms object after applying the rattle mutation, or None if mutation fails.

        """
        N = len(atoms) if self.n_top is None else self.n_top
        slab = atoms[:len(atoms) - N]
        atoms = atoms[len(atoms) - N:]
        tags = atoms.get_tags() if self.use_tags else np.arange(N)
        pos_ref = atoms.get_positions()
        num = atoms.get_atomic_numbers()
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()

        unique_tags = _filter_unique_tags(np.unique(tags), self.target_tags)
        if unique_tags is None:
            return None
        # Centering undoes the only move a one-atom gas cluster can make.
        if not self._policy.uses_surface and N < 2:
            self.last_attempt_count = 0
            return None

        parent_mobile = atoms
        self.last_attempt_count = 0
        best = None
        break_rigid = not self._policy.uses_surface
        for scale in _STRENGTH_SCALES:
            strength = float(self.rattle_strength) * scale
            st = 2. * strength
            # Half strength moves one group. A partial rattle disconnects a loose cluster.
            single_group = scale < 1.0
            pos = pos_ref.copy()
            for tag in _tags_to_move(
                self.rng, unique_tags, self.rattle_prop, single_group=single_group
            ):
                delta = st * (self.rng.random(3) - 0.5)
                if single_group:
                    delta = _ensure_step_length(
                        delta, _HALF_STEP_FLOOR_FRACTION * strength
                    )
                _apply_group_step(
                    self.rng,
                    pos,
                    np.flatnonzero(tags == tag),
                    delta,
                    break_rigid=break_rigid,
                )

            self.last_attempt_count += 1
            top = Atoms(num, positions=pos, cell=cell, pbc=pbc, tags=tags)
            if self._policy.uses_surface:
                top = _reanchor_mobile_to_slab(
                    atoms, top, slab, self.surface_normal_axis)
            accepted, best = _consider_local_candidate(
                self, parent_mobile, top, slab, best
            )
            if accepted is not None:
                return _as_offspring(self, slab, accepted)

        accepted = _repair_best(self, parent_mobile, slab, best)
        if accepted is None:
            return None
        return _as_offspring(self, slab, accepted)

    def mutate_few_atom_kick(
        self,
        atoms,
        *,
        n_atoms: int = 3,
        kick_strength: float = 1.6,
        max_tries: int = _KICK_TRIES,
        min_displacement: float = _KICK_MIN_DISPLACEMENT,
    ):
        """Kick one to a few atoms or tag groups by a large Gaussian step."""
        n_mobile = len(atoms) if self.n_top is None else self.n_top
        slab = atoms[: len(atoms) - n_mobile]
        mobile = atoms[len(atoms) - n_mobile :]
        tags = mobile.get_tags() if self.use_tags else np.arange(n_mobile)
        unique_tags = _filter_unique_tags(np.unique(tags), self.target_tags)
        if unique_tags is None:
            return None
        if not self._policy.uses_surface and n_mobile < 2:
            return None
        kick_count = max(1, min(int(n_atoms), len(unique_tags)))
        pos_ref = mobile.get_positions()
        num = mobile.get_atomic_numbers()
        cell = mobile.get_cell()
        pbc = mobile.get_pbc()
        floor = float(min_displacement)
        best = None
        break_rigid = not self._policy.uses_surface
        for _ in range(max(1, int(max_tries))):
            chosen = self.rng.choice(len(unique_tags), size=kick_count, replace=False)
            pos = pos_ref.copy()
            long_enough = True
            for index in np.atleast_1d(chosen):
                step = self.rng.normal(0.0, float(kick_strength), size=3)
                if float(np.linalg.norm(step)) < floor:
                    long_enough = False
                    break
                _apply_group_step(
                    self.rng,
                    pos,
                    np.flatnonzero(tags == unique_tags[int(index)]),
                    step,
                    break_rigid=break_rigid,
                )
            if not long_enough:
                continue
            top = Atoms(num, positions=pos, cell=cell, pbc=pbc, tags=tags)
            if self._policy.uses_surface:
                top = _reanchor_mobile_to_slab(
                    mobile, top, slab, self.surface_normal_axis
                )
            accepted, best = _consider_local_candidate(
                self, mobile, top, slab, best
            )
            if accepted is not None:
                return _as_offspring(self, slab, accepted)
        accepted = _repair_best(self, mobile, slab, best)
        if accepted is None:
            return None
        return _as_offspring(self, slab, accepted)


class AnisotropicRattleMutation(OffspringCreator):
    """Rattle mutation with stronger in-plane and weaker normal displacement.

    A random plane is sampled every attempt. Selected atoms (or tag groups) are
    displaced primarily in-plane to encourage exploration of flat/pseudolinear
    regions while still allowing smaller out-of-plane motion.
    """

    def __init__(
        self,
        blmin,
        n_top,
        system_type: SystemType,
        in_plane_strength=1.0,
        normal_strength=0.2,
        rattle_prop=0.5,
        test_dist_to_slab=True,
        use_tags=False,
        target_tags=None,
        rng=None,
        verbose=False,
        surface_normal_axis=2,
    ):
        rng = _ensure_rng(rng)
        OffspringCreator.__init__(self, verbose, rng=rng)
        self.blmin = blmin
        self.n_top = n_top
        self.in_plane_strength = in_plane_strength
        self.normal_strength = normal_strength
        self.rattle_prop = rattle_prop
        self.test_dist_to_slab = test_dist_to_slab
        self.use_tags = use_tags
        self.target_tags = target_tags
        self.system_type = system_type
        self.surface_normal_axis = surface_normal_axis
        self._policy = get_system_policy(system_type)

        self.descriptor = "AnisotropicRattleMutation"
        self.min_inputs = 1
        self.last_attempt_count = 0

    def get_new_individual(self, parents):
        f = parents[0]

        indi = self.mutate(f)

        return _finalize_mutant(self, f, indi, "mutation: anisotropic_rattle")

    def mutate(self, atoms):
        N = len(atoms) if self.n_top is None else self.n_top
        slab = atoms[: len(atoms) - N]
        atoms = atoms[len(atoms) - N:]
        tags = atoms.get_tags() if self.use_tags else np.arange(N)
        pos_ref = atoms.get_positions()
        num = atoms.get_atomic_numbers()
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()

        unique_tags = _filter_unique_tags(np.unique(tags), self.target_tags)
        if unique_tags is None:
            return None
        if not self._policy.uses_surface and N < 2:
            self.last_attempt_count = 0
            return None

        parent_mobile = atoms
        self.last_attempt_count = 0
        best = None
        break_rigid = not self._policy.uses_surface
        for scale in _STRENGTH_SCALES:
            in_plane = self.in_plane_strength * scale
            normal_s = self.normal_strength * scale
            pos = pos_ref.copy()

            if self._policy.uses_surface:
                normal = np.zeros(3)
                normal[int(self.surface_normal_axis)] = 1.0
            else:
                normal = self.rng.normal(0.0, 1.0, 3)
                normal_norm = np.linalg.norm(normal)
                if normal_norm <= 1e-12:
                    normal = np.array([0.0, 0.0, 1.0])
                else:
                    normal = normal / normal_norm

            # Pick a helper vector not parallel to normal.
            helper = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(helper, normal)) > 0.9:
                helper = np.array([0.0, 1.0, 0.0])

            # Build orthonormal in-plane basis. A degenerate sample falls back
            # to a fixed axis instead of spinning the attempt loop.
            u = np.cross(normal, helper)
            u_norm = np.linalg.norm(u)
            if u_norm <= 1e-12:
                helper = np.array([0.0, 0.0, 1.0])
                u = np.cross(normal, helper)
                u_norm = np.linalg.norm(u)
            u = np.array([1.0, 0.0, 0.0]) if u_norm <= 1e-12 else u / u_norm
            v = np.cross(normal, u)
            v_norm = np.linalg.norm(v)
            v = np.array([0.0, 1.0, 0.0]) if v_norm <= 1e-12 else v / v_norm

            single_group = scale < 1.0
            for tag in _tags_to_move(
                self.rng, unique_tags, self.rattle_prop, single_group=single_group
            ):
                a = self.rng.uniform(-in_plane, in_plane)
                b = self.rng.uniform(-in_plane, in_plane)
                c = self.rng.uniform(-normal_s, normal_s)
                delta = a * u + b * v + c * normal
                if single_group:
                    delta = _ensure_step_length(
                        delta, _HALF_STEP_FLOOR_FRACTION * in_plane
                    )
                _apply_group_step(
                    self.rng,
                    pos,
                    np.flatnonzero(tags == tag),
                    delta,
                    break_rigid=break_rigid,
                )

            top = Atoms(num, positions=pos, cell=cell, pbc=pbc, tags=tags)
            if self._policy.uses_surface:
                top = _reanchor_mobile_to_slab(
                    atoms, top, slab, self.surface_normal_axis)
            self.last_attempt_count += 1
            accepted, best = _consider_local_candidate(
                self, parent_mobile, top, slab, best
            )
            if accepted is not None:
                return _as_offspring(self, slab, accepted)

        accepted = _repair_best(self, parent_mobile, slab, best)
        if accepted is None:
            return None
        return _as_offspring(self, slab, accepted)

# fmt: on
