"""Mutation that applies random rotations to multi-atom moieties."""

from __future__ import annotations


import numpy as np
from ase import Atoms

from scgo.ase_ga_patches._tag_gather import (
    gather_atoms_by_tag,
    periodic_sheet_tag_to_skip,
)
from scgo.ase_ga_patches.mutations._common import (
    _mutation_steric_score,
    MutationOperator,
    _accept_connected_candidate,
    _accept_connected_kwargs,
    _append_unique_unit_vector,
    _fails_only_slab_clash,
    _filter_unique_tags,
    _geometry_candidate_directions,
    _pca_sorted_eigenvectors,
    _random_unit_vector,
    _reanchor_mobile_to_slab,
)
from scgo.initialization.steric_scoring import _blmin_matrix
from scgo.system_types import SystemType
from scgo.utils.geometry_math import rotation_matrix_axis_angle

__all__ = ["RotationalMutation"]


class RotationalMutation(MutationOperator):
    descriptor = "RotationalMutation"
    """Mutates a candidate by rotating a random subset of the
    multi-atom moieties in the structure (atoms with
    the same tag are considered part of one such moiety).

    The moieties to rotate are drawn at random, while the rotation axes
    are derived from each moiety's geometry (topped up with random
    directions). Tagged-moiety rotations span [min_angle, pi] via
    ``_candidate_angles`` and are ranked by steric deficit. When no
    eligible multi-atom tags remain (untagged gas clusters, or a whole-
    mobile group), ``_twist_mobile`` uses a fixed descending angle ladder
    ``[pi, 2.0, 1.0, 0.5, 0.2]`` independent of ``min_angle``, then lifts
    the cap along the twist axis by the blmin gap.

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
        Fraction of the eligible multi-atom moieties to be rotated,
        rounded up.

    tags: None or list of integers
        Specifies, respectively, whether all moieties or only those
        with matching tags are eligible for rotation.

    min_angle: float
        Minimal angle (in radians) for tagged-moiety rotations;
        should lie in the interval [0, pi]. Ignored by the untagged
        whole-mobile twist ladder.

    test_dist_to_slab: boolean
        Whether also the distances to the slab
        should be checked to satisfy the blmin.

    rng: Random number generator
        Must be an instance of ``np.random.Generator`` or ``None``.

    """

    def __init__(
        self,
        blmin,
        system_type: SystemType,
        n_top=None,
        fraction=0.33,
        tags=None,
        min_angle=1.57,
        test_dist_to_slab=True,
        target_tags=None,
        use_tags=False,
        rng=None,
        verbose=False,
        max_inner_attempts=24,
        surface_normal_axis=2,
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
            surface_normal_axis=surface_normal_axis,
        )
        self.fraction = fraction
        self.tags = tags
        self.min_angle = min_angle
        self.max_inner_attempts = max_inner_attempts
        self.last_attempt_count = 0

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
                for axis in _pca_sorted_eigenvectors(centered):
                    _append_unique_unit_vector(candidates, axis)

        attempts = 0
        while len(candidates) < 3 and attempts < 20:
            _append_unique_unit_vector(candidates, _random_unit_vector(self.rng))
            attempts += 1

        return candidates[:3]

    def _candidate_angles(self):
        min_angle = self.min_angle
        # Prefer modest angles first so tagged adsorbate fragments stay bonded
        # to the core; large angles remain available when they still connect.
        small = max(0.25, 0.25 * min_angle)
        mid_small = 0.5 * min_angle
        mid_angle = 0.5 * (min_angle + np.pi)
        angles = [small, mid_small, min_angle, mid_angle, np.pi]
        max_angles = max(1, min(int(self.max_inner_attempts), 5))
        return angles[:max_angles]

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
            rotation = rotation_matrix_axis_angle(axis, angle)
            new_positions[indices[tag]] = (
                np.dot(rotation, (moiety - center).T).T + center
            )
        return new_positions

    def _place_trial(self, mobile, positions, slab):
        trial = mobile.copy()
        trial.set_positions(positions)
        if self._policy.uses_surface:
            return _reanchor_mobile_to_slab(
                mobile, trial, slab, self.surface_normal_axis
            )
        trial.center()
        return trial

    def _one_plane_shift(self, positions):
        """Single in-plane nudge of half the smallest blmin."""
        if not self.blmin:
            return None
        offset = 0.5 * min(self.blmin.values())
        if offset <= 0.0:
            return None
        in_plane = [i for i in range(3) if i != self.surface_normal_axis]
        shifted = np.array(positions, dtype=float, copy=True)
        shifted[:, in_plane[0]] += offset
        return shifted

    def _twist_angles(self):
        """Descending twist magnitudes; smaller angles recover when large ones clash."""
        angles = [np.pi, 2.0, 1.0, 0.5, 0.2]
        limit = max(1, min(len(angles), int(self.max_inner_attempts)))
        return angles[:limit]

    def _twist_cap(self, positions, numbers, normal, cap, pivot, angle):
        """Rotate the cap by ``angle`` and lift it clear of blmin along ``normal``."""
        angle = float(angle)
        axis = np.asarray(normal, dtype=float)
        axis /= np.linalg.norm(axis)
        cap = np.asarray(cap, dtype=int)
        rotated = (positions[cap] - pivot) @ rotation_matrix_axis_angle(
            axis, angle
        ).T + pivot
        newpos = positions.copy()
        newpos[cap] = rotated
        stationary = np.setdiff1d(np.arange(len(positions)), cap, assume_unique=False)
        if stationary.size == 0:
            return newpos
        diff = rotated[:, None, :] - positions[stationary][None, :, :]
        axial = diff @ axis
        plane = np.linalg.norm(diff - axial[..., None] * axis, axis=2)
        height = (positions[cap][:, None, :] - positions[stationary][None, :, :]) @ axis
        required = _blmin_matrix(numbers[cap], self.blmin, numbers[stationary])
        missing = np.sqrt(np.maximum(required * required - plane * plane, 0.0))
        need = np.where(plane < required, np.maximum(0.0, missing - height), 0.0)
        newpos[cap] = rotated + float(need.max()) * axis
        return newpos

    def _twist_mobile(self, slab, mobile):
        """Rotate a few outer atoms about the nearest core atom. Large angles first."""
        positions = mobile.get_positions()
        numbers = mobile.get_atomic_numbers()
        if len(positions) < 3:
            return None
        center = np.mean(positions, axis=0)
        angles = self._twist_angles()
        n_normals = max(1, min(4, int(self.max_inner_attempts) // len(angles)))
        normals = _geometry_candidate_directions(
            positions, center, slab, self.rng, n_normals
        )
        accept_kwargs = _accept_connected_kwargs(self, slab)
        self.last_attempt_count = 0
        best_slab: tuple[float, np.ndarray] | None = None
        # At least two atoms stay in the core, so this is not a rigid turn
        # about a single vertex.
        n_cap = min(4, len(positions) - 2)
        if n_cap < 2:
            n_cap = min(2, len(positions) - 1)
        for normal in normals:
            signed = np.dot(positions - center, normal)
            if float(np.ptp(signed)) <= 1e-6:
                continue
            order = np.argsort(signed)
            cap = order[-n_cap:]
            core = order[:-n_cap]
            delta = positions[core][:, None, :] - positions[cap][None, :, :]
            distances = np.linalg.norm(delta, axis=2)
            core_index, _cap_index = np.unravel_index(
                int(np.argmin(distances)), distances.shape
            )
            pivot = positions[int(core[int(core_index)])]
            for angle in angles:
                newpos = self._twist_cap(
                    positions, numbers, normal, cap, pivot, angle=angle
                )
                self.last_attempt_count += 1
                trial = self._place_trial(mobile, newpos, slab)
                accepted = _accept_connected_candidate(mobile, trial, **accept_kwargs)
                if accepted is not None:
                    return slab + accepted
                if not self._policy.uses_surface:
                    continue
                if not _fails_only_slab_clash(mobile, trial, **accept_kwargs):
                    continue
                score = _mutation_steric_score(
                    trial.get_positions(), numbers, slab, self.blmin
                )
                if best_slab is None or score < best_slab[0]:
                    best_slab = (score, trial.get_positions().copy())

        if best_slab is None:
            return None
        shifted = self._one_plane_shift(best_slab[1])
        if shifted is None:
            return None
        self.last_attempt_count += 1
        trial = mobile.copy()
        trial.set_positions(shifted)
        accepted = _accept_connected_candidate(mobile, trial, **accept_kwargs)
        if accepted is not None:
            return slab + accepted
        return None

    def mutate_twist(self, atoms):
        """Twist the mobile cluster even when multi-atom tags are present."""
        count = len(atoms) if self.n_top is None else self.n_top
        slab = atoms[: len(atoms) - count]
        mobile = atoms[len(atoms) - count :]
        return self._twist_mobile(slab, mobile)

    def mutate(self, atoms):
        """Does the actual mutation."""
        N = len(atoms) if self.n_top is None else self.n_top
        slab = atoms[: len(atoms) - N]
        atoms = atoms[len(atoms) - N :]

        mutant = atoms.copy()
        gather_atoms_by_tag(
            mutant,
            skip_tag=periodic_sheet_tag_to_skip(self.system_type),
        )
        pos = mutant.get_positions()
        tags = mutant.get_tags()
        numbers = mutant.get_atomic_numbers()

        unique_tags = _filter_unique_tags(np.unique(tags), self.target_tags)
        if unique_tags is None:
            return None

        eligible_tags = self.tags if self.tags is not None else unique_tags
        # Filter eligible_tags to only include tags we're targeting
        eligible_tags = [t for t in eligible_tags if t in unique_tags]

        indices = {}
        for tag in eligible_tags:
            hits = np.where(tags == tag)[0]
            if len(hits) > 1:
                indices[tag] = hits

        if len(slab) == 0:
            # Without a slab, rotating a group that spans the whole mobile region
            # is a rigid-body move of the entire (re-centred) cluster. Untagged
            # clusters have one such group, so twist a subset instead.
            n_mobile = len(mutant)
            indices = {
                tag: hits for tag, hits in indices.items() if len(hits) < n_mobile
            }

        if not indices:
            return self._twist_mobile(slab, atoms)

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
            score = _mutation_steric_score(newpos, numbers, slab, self.blmin)
            ranked.append((score, newpos))

        ranked.sort(key=lambda item: item[0])
        self.last_attempt_count = 0
        accept_kwargs = _accept_connected_kwargs(self, slab)
        best_slab: tuple[float, np.ndarray] | None = None
        for score, newpos in ranked:
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
            if not self._policy.uses_surface:
                mutant.center()

            accepted = _accept_connected_candidate(atoms, mutant, **accept_kwargs)
            if accepted is not None:
                return slab + accepted
            slab_only = self._policy.uses_surface and _fails_only_slab_clash(
                atoms, mutant, **accept_kwargs
            )
            if slab_only and (best_slab is None or score < best_slab[0]):
                best_slab = (score, mutant.get_positions().copy())

        if best_slab is None:
            return None
        shifted = self._one_plane_shift(best_slab[1])
        if shifted is None:
            return None
        self.last_attempt_count += 1
        mutant.set_positions(shifted)
        accepted = _accept_connected_candidate(atoms, mutant, **accept_kwargs)
        if accepted is not None:
            return slab + accepted
        return None
