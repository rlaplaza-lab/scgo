"""Mirror mutation that reflects the mobile region across a cutting plane."""

# fmt: off

from __future__ import annotations


import numpy as np
from ase import Atoms
from ase_ga.offspring_creator import OffspringCreator
from ase_ga.utilities import atoms_too_close, atoms_too_close_two_sets

from scgo.ase_ga_patches.mutations._common import (
    _append_unique_unit_vector,
    _ensure_rng,
    _geometry_candidate_directions,
    _random_unit_vector,
    _reanchor_mobile_to_slab,
)
from scgo.ase_ga_patches.mutations._finalize import _finalize_mutant
from scgo.initialization.steric_scoring import steric_deficit as _steric_deficit
from scgo.initialization.steric_scoring import (
    steric_deficit_two_sets as _steric_deficit_two_sets,
)
from scgo.system_types import SystemType, get_system_policy

__all__ = ["MirrorMutation"]


class MirrorMutation(OffspringCreator):
    """Reflect the target mobile atoms across a ranked cutting plane.

    Candidate planes come from the structure geometry (outward slab
    direction, principal axes) and random fill, then ranked by steric
    deficit. The reflection is an isometry of the target set, so it is
    only a useful GA move when a reference remains: a slab, or leftover
    mobile atoms (for example a tagged adsorbate). In gas phase, if the
    target covers every mobile atom the operator returns ``None`` instead
    of emitting an energy-identical duplicate.

    Parameters
    ----------
    blmin: Dictionary defining the minimum allowed
        distance between atoms.

    n_top: Number of atoms the GA optimizes.

    reflect: Defines if the mirrored half is also reflected
        perpendicular to the mirroring plane. Both settings are always
        evaluated; this one wins among equally ranked candidates.

    rng: Random number generator
        Must be an instance of ``np.random.Generator`` or ``None``.

    """

    def __init__(self, blmin, n_top, system_type: SystemType, reflect=True,
                 target_tags=None, rng=None, verbose=False, max_tries=12,
                 surface_normal_axis=2):
        rng = _ensure_rng(rng)
        OffspringCreator.__init__(self, verbose, rng=rng)
        self.blmin = blmin
        self.n_top = n_top
        self.max_tries = max_tries
        self.reflect = reflect
        self.target_tags = target_tags
        self.system_type = system_type
        self._policy = get_system_policy(system_type)
        self.surface_normal_axis = surface_normal_axis
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

    def _build_mirror_top(self, num, pos, center_of_mass, plane, reflect, plane_origin=None):
        if plane_origin is None:
            plane_origin = center_of_mass

        # Reflect the *entire* mobile region across the cutting plane (through
        # ``plane_origin``). Reflection is an isometry: it preserves every pairwise
        # distance, so a blmin-valid, connectivity-valid parent maps to a
        # blmin-valid, connectivity-valid mutant. This is robust even on highly
        # symmetric compact clusters (e.g. an icosahedral 55-atom Pt cluster whose
        # central atom lies exactly on the plane -- it is simply left in place
        # rather than duplicated, as the old "keep half / mirror half" scheme did,
        # which collapsed the cluster below the bond-length minimum and created
        # coincident atoms). ``reflect`` additionally applies the improper
        # inversion used by the original operator.
        new_pos = pos.copy()
        for index in range(len(pos)):
            point = pos[index]
            distance = float(np.dot(point - plane_origin, plane))
            mirrored_point = point - 2.0 * distance * plane
            if reflect:
                mirrored_point = (
                    -mirrored_point
                    + 2.0 * center_of_mass
                    + 2.0 * plane * np.dot(mirrored_point - center_of_mass, plane)
                )
            new_pos[index] = mirrored_point

        return Atoms(num, new_pos)

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

        # Full-cluster reflection in gas is an isometry of the mobile region
        # (same internals as the parent). Decline unless some mobile atoms
        # stay put as a reference (e.g. a tagged adsorbate) or a slab is present.
        if len(slab) == 0 and bool(np.all(target_mask)):
            return None

        reflect_options = [self.reflect]
        if not self.reflect:
            reflect_options.append(True)
        else:
            reflect_options.append(False)

        max_candidates = max(1, min(int(self.max_tries), 12))
        radii = float(np.max(np.linalg.norm(pos - center_of_mass, axis=1))) or 1.0
        ranked_candidates = []
        # Center-plane reflection of a compact, roughly spherical cluster overlaps
        # itself, so also sweep off-center plane origins along the normal. This
        # keeps the operator robust on symmetric deposited/compact clusters.
        offsets = [0.0, 0.35 * radii, -0.35 * radii, 0.7 * radii, -0.7 * radii]
        for plane in self._candidate_planes(target_pos, center_of_mass, slab):
            for reflect in reflect_options:
                for offset in offsets:
                    plane_origin = center_of_mass + offset * plane
                    mirrored_top = self._build_mirror_top(
                        target_num,
                        target_pos,
                        center_of_mass,
                        plane,
                        reflect,
                        plane_origin=plane_origin,
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
            if self._policy.uses_surface:
                mutant = _reanchor_mobile_to_slab(
                    top, mutant, slab, self.surface_normal_axis
                )
            if atoms_too_close(mutant, self.blmin):
                continue
            if atoms_too_close_two_sets(slab, mutant, self.blmin):
                continue
            return slab + mutant

        # Rescue: a compact/symmetric cluster can overlap after a center-plane
        # reflection. Slide the mirrored (negative-side) atoms along the plane
        # normal away from the cluster until the clash clears; this keeps the
        # mutation valid without giving up.
        if ranked_candidates:
            best = min(ranked_candidates, key=lambda item: item[0])[1]
            for plane in self._candidate_planes(target_pos, center_of_mass, slab):
                rescue = best.copy()
                for step in range(1, 12):
                    shift = step * 0.5 * min(self.blmin.values()) * plane
                    rescue.positions = best.get_positions() + shift
                    if not atoms_too_close(rescue, self.blmin) and not (
                        len(slab) > 0
                        and atoms_too_close_two_sets(slab, rescue, self.blmin)
                    ):
                        self.last_attempt_count += 1
                        return slab + rescue

        return None

# fmt: on
