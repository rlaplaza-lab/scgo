"""Implementation of the cut-and-splice paring operator."""

# fmt: off

from __future__ import annotations

from scgo.exceptions import (
    SCGOValidationError,
)

import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from ase_ga.offspring_creator import OffspringCreator
from ase_ga.utilities import get_rotation_matrix
from scgo.ase_ga_patches._slab_clash import (
    SlabClashChecker,
    mobile_too_close_no_copy,
    mobile_too_close_tagged,
)
from scgo.ase_ga_patches._tag_gather import (
    gather_atoms_by_tag,
    periodic_sheet_tag_to_skip,
)
from scgo.ase_ga_patches._vector_utils import (
    append_unique_unit_vector as _append_unique_unit_vector,
    random_unit_vector as _random_unit_vector,
)
from scgo.system_types import uses_surface
from scgo.utils.helpers import get_composition_counts
from scgo.utils.rng_helpers import ensure_rng_or_create as _ensure_rng


def _assert_offspring_integrity(child: Atoms, parent: Atoms) -> None:
    """Post-crossover guard: offspring must keep parent atom count and stoichiometry.

    Catches bugs where cut-and-splice drops or duplicates atoms (e.g. the slab
    prepend, the mobile-slice selection, or the symbol-ordering reconstruction).
    """
    if len(child) != len(parent):
        raise SCGOValidationError(
            f"{parent.info.get('key', 'CutAndSplicePairing')} produced "
            f"{len(child)} atoms; expected {len(parent)} "
            f"(slab + n_top)."
        )
    child_counts = get_composition_counts(child.get_chemical_symbols())
    parent_counts = get_composition_counts(parent.get_chemical_symbols())
    if child_counts != parent_counts:
        raise SCGOValidationError(
            f"CutAndSplicePairing changed stoichiometry: "
            f"{dict(child_counts)} vs {dict(parent_counts)}."
        )


def _plane_distance_magnitude(distance) -> float:
    """Smallest absolute distance of a group to the cutting plane."""
    values = np.atleast_1d(np.asarray(distance, dtype=float)).reshape(-1)
    return float(np.min(np.abs(values)))


def _rotation_matrix(axis, angle):
    """Right-handed rotation. Row vectors transform as ``p @ R.T``."""
    axis = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        return np.eye(3)
    return get_rotation_matrix(axis / norm, angle)


def _principal_axes(positions):
    """Orthonormal axes, largest variance first, with a positive determinant."""
    centered = np.asarray(positions, dtype=float)
    centered = centered - centered.mean(axis=0)
    _values, vectors = np.linalg.eigh(centered.T @ centered)
    axes = vectors[:, np.argsort(_values)[::-1]]
    if float(np.linalg.det(axes)) < 0.0:
        axes = axes.copy()
        axes[:, -1] *= -1.0
    return axes


def _proper_axis_signs(dimension):
    """Sign patterns that keep a right-handed axis frame."""
    if dimension == 2:
        return (np.ones(2), np.array([-1.0, -1.0]))
    return (
        np.ones(3),
        np.array([-1.0, -1.0, 1.0]),
        np.array([-1.0, 1.0, -1.0]),
        np.array([1.0, -1.0, -1.0]),
    )


def _pop_closest_to_plane(items):
    """Remove and return the group whose center lies closest to the cut."""
    index = min(
        range(len(items)),
        key=lambda i: (_plane_distance_magnitude(items[i].distance), i),
    )
    return items.pop(index)


class Positions:
    """Helper object to simplify the pairing process.

    Parameters
    ----------
    scaled_positions: (Nx3) array
        Positions in scaled coordinates
    cop: (1x3) array
        Center-of-positions (also in scaled coordinates)
    symbols: str
        String with the atomic symbols
    distance: float
        Signed distance to the cutting plane
    origin: int (0 or 1)
        Determines at which side of the plane the position should be.

    """

    def __init__(
        self,
        scaled_positions,
        cop,
        symbols,
        distance,
        origin,
        *,
        tag=None,
        n_atoms=None,
        is_target_tag=True,
    ):
        self.scaled_positions = scaled_positions
        self.cop = cop
        self.symbols = symbols
        self.distance = distance
        self.origin = origin
        self.tag = tag
        self.n_atoms = len(scaled_positions) if n_atoms is None else int(n_atoms)
        self.is_target_tag = bool(is_target_tag)

    def to_use(self):
        """Tells whether this position is at the right side."""
        return bool(self.distance > 0.0 and self.origin == 0 or self.distance < 0.0 and self.origin == 1)


class CutAndSplicePairing(OffspringCreator):
    """The Cut and Splice operator by Deaven and Ho.

    Creates offspring from two parent structures using
    a randomly generated cutting plane.

    The parents may have different unit cells, in which
    case the offspring unit cell will be a random combination
    of the parent cells.

    The basic implementation (for fixed unit cells) is
    described in:

    :doi:`L.B. Vilhelmsen and B. Hammer, PRL, 108, 126101 (2012)
    <10.1103/PhysRevLett.108.126101>`

    The extension to variable unit cells is similar to:

    * :doi:`Glass, Oganov, Hansen, Comp. Phys. Comm. 175 (2006) 713-720
      <10.1016/j.cpc.2006.07.020>`

    * :doi:`Lonie, Zurek, Comp. Phys. Comm. 182 (2011) 372-387
      <10.1016/j.cpc.2010.07.048>`

    The operator can furthermore preserve molecular identity
    if desired (see the *use_tags* kwarg). Atoms with the same
    tag will then be considered as belonging to the same molecule,
    and their internal geometry will not be changed by the operator.

    If use_tags is enabled, the operator will also conserve the
    number of molecules of each kind (in addition to conserving
    the overall stoichiometry). Currently, molecules are considered
    to be of the same kind if their chemical symbol strings are
    identical. In rare cases where this may not be sufficient
    (i.e. when desiring to keep the same ratio of isomers), the
    different isomers can be differentiated by giving them different
    elemental orderings (e.g. 'XY2' and 'Y2X').

    Parameters
    ----------
    slab: Atoms object
        Specifies the cell vectors and periodic boundary conditions
        to be applied to the randomly generated structures.
        Any included atoms (e.g. representing an underlying slab)
        are copied to these new structures.

    n_top: int
        The number of atoms to optimize

    blmin: dict
        Dictionary with minimal interatomic distances.
        Note: when preserving molecular identity (see use_tags),
        the blmin dict will (naturally) only be applied
        to intermolecular distances (not the intramolecular
        ones).

    number_of_variable_cell_vectors: int (default 0)
        The number of variable cell vectors (0, 1, 2 or 3).
        To keep things simple, it is the 'first' vectors which
        will be treated as variable, i.e. the 'a' vector in the
        univariate case, the 'a' and 'b' vectors in the bivariate
        case, etc.

    p1: float or int between 0 and 1
        Probability that a parent is shifted over a random
        distance along the normal of the cutting plane
        (only operative if number_of_variable_cell_vectors > 0).

    p2: float or int between 0 and 1
        Same as p1, but for shifting along the directions
        in the cutting plane (only operative if
        number_of_variable_cell_vectors > 0).

    minfrac: float between 0 and 1, or None (default)
        Minimal fraction of atoms a parent must contribute
        to the child. If None, each parent must contribute
        at least one atom.

    cellbounds: ase.ga.utilities.CellBounds instance
        Describing limits on the cell shape, see
        :class:`~ase.ga.utilities.CellBounds`.
        Note that it only make sense to impose conditions
        regarding cell vectors which have been marked as
        variable (see number_of_variable_cell_vectors).

    use_tags: bool
        Whether to use the atomic tags to preserve
        molecular identity.

    test_dist_to_slab: bool (default True)
        Whether to make sure that the distances between
        the atoms and the slab satisfy the blmin.

    rng: Random number generator
        Must be an instance of ``np.random.Generator`` or ``None``.

    """

    def __init__(self, slab, n_top, blmin, number_of_variable_cell_vectors=0,
                 p1=1, p2=0.05, minfrac=None, cellbounds=None,
                 test_dist_to_slab=True, use_tags=False, target_tags=None, rng=None,
                 verbose=False, system_type="gas_cluster",
                 max_pairing_attempts: int | None = None,
                 fixed_cell_max_passes: int = 3,
                 variable_cell_max_passes: int = 3,
                 prefer_unbalanced: bool = False,
                 surface_normal_axis: int = 2):

        rng = _ensure_rng(rng)
        OffspringCreator.__init__(self, verbose, rng=rng)
        self.slab = slab
        self.n_top = n_top
        self.blmin = blmin
        if number_of_variable_cell_vectors not in range(4):
            raise SCGOValidationError(
                "number_of_variable_cell_vectors must be 0, 1, 2, or 3"
            )
        self.number_of_variable_cell_vectors = number_of_variable_cell_vectors
        self.p1 = p1
        self.p2 = p2
        self.minfrac = minfrac
        self.cellbounds = cellbounds
        self.test_dist_to_slab = test_dist_to_slab
        self.use_tags = use_tags
        self.target_tags = target_tags
        self.last_attempt_count = 0
        self.last_cell_attempt_count = 0
        if len(self.slab) > 0 and not uses_surface(system_type):
            raise SCGOValidationError(
                "CutAndSplicePairing received slab atoms with non-surface "
                f"system_type={system_type!r}. Set system_type to a surface_* type."
            )
        self.system_type = system_type
        if max_pairing_attempts is None:
            max_pairing_attempts = 150 if not uses_surface(system_type) else 1000
        self.max_pairing_attempts = int(max_pairing_attempts)
        # Both cell modes walk a short chain of ranked cuts. Fixed-cell passes
        # jitter then reorient; variable-cell passes translate and re-rank.
        # ``max_pairing_attempts`` still caps the pass count.
        self.fixed_cell_max_passes = max(1, int(fixed_cell_max_passes))
        self.variable_cell_max_passes = max(1, int(variable_cell_max_passes))
        self.prefer_unbalanced = bool(prefer_unbalanced)
        self.surface_normal_axis = int(surface_normal_axis)

        self.scaling_volume = None
        self.descriptor = "CutAndSplicePairing"
        self.min_inputs = 2

        # Pre-build the cached slab clash checker so slab positions and PBC
        # images are expanded only once per pairing operator instance.
        # The checker is rebuilt if test_dist_to_slab is False or slab is empty.
        if self.test_dist_to_slab and len(self.slab) > 0:
            self._slab_clash_checker: SlabClashChecker | None = SlabClashChecker(
                slab_numbers=slab.get_atomic_numbers(),
                slab_positions=slab.get_positions(),
                cell=slab.get_cell(),
                pbc=slab.get_pbc(),
                blmin=blmin,
            )
        else:
            self._slab_clash_checker = None

    def _group_centers_and_weights(self, atoms):
        tags = atoms.get_tags() if self.use_tags else np.arange(len(atoms))
        centers = []
        weights = []
        for tag in np.unique(tags):
            if self.target_tags is not None and tag not in self.target_tags:
                continue
            indices = np.where(tags == tag)[0]
            centers.append(np.mean(atoms.positions[indices], axis=0))
            weights.append(len(indices))
        return np.asarray(centers, dtype=float), np.asarray(weights, dtype=int)

    def _candidate_cut_normals(self, a1, a2):
        combined = np.vstack([a1.get_positions(), a2.get_positions()])
        centered = combined - np.mean(combined, axis=0)
        candidates = []

        _append_unique_unit_vector(
            candidates,
            np.mean(a2.get_positions(), axis=0) - np.mean(a1.get_positions(), axis=0),
        )

        if len(centered) > 1:
            covariance = np.dot(centered.T, centered)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            order = np.argsort(eigenvalues)[::-1]
            axes = [eigenvectors[:, index] for index in order]
            for axis in axes:
                _append_unique_unit_vector(candidates, axis)
            if len(axes) >= 2:
                _append_unique_unit_vector(candidates, axes[0] + axes[-1])

        parent_delta = a2.get_positions() - a1.get_positions()
        if len(parent_delta) > 0:
            _append_unique_unit_vector(
                candidates,
                parent_delta[int(np.argmax(np.linalg.norm(parent_delta, axis=1)))],
            )

        attempts = 0
        while len(candidates) < 6 and attempts < 100:
            _append_unique_unit_vector(candidates, _random_unit_vector(self.rng))
            attempts += 1

        return candidates[:6]

    def _estimate_cut_balance(
        self,
        centers_a1,
        centers_a2,
        weights,
        cut_point_cart,
        cut_normal_cart,
    ):
        distances_a1 = np.dot(centers_a1 - cut_point_cart, cut_normal_cart)
        distances_a2 = np.dot(centers_a2 - cut_point_cart, cut_normal_cart)

        count1 = int(np.sum(weights[distances_a1 > 0.0]))
        count2 = int(np.sum(weights[distances_a2 < 0.0]))
        nmin = 1 if self.minfrac is None else max(1, int(np.ceil(self.minfrac * self.n_top)))
        target = 0.5 * self.n_top
        balance = abs(count1 - target) + abs(count2 - target)
        deficit = max(0, nmin - count1) + max(0, nmin - count2)
        if self.prefer_unbalanced:
            return (deficit, -balance)
        return (deficit, balance)

    def _break_score_ties(self, configs):
        """Shuffle equal scores. A legal cut still stays ahead of an illegal one."""
        if len(configs) < 2:
            return configs
        grouped = []
        current = [configs[0]]
        for item in configs[1:]:
            if item[0] == current[0][0]:
                current.append(item)
            else:
                grouped.append(current)
                current = [item]
        grouped.append(current)
        ordered = []
        for group in grouped:
            if len(group) == 1:
                ordered.extend(group)
                continue
            order = np.arange(len(group))
            self.rng.shuffle(order)
            ordered.extend(group[int(index)] for index in order)
        return ordered

    def _axis_cut_normal(self, cell_array, axis):
        """Unit normal of the constant-fraction plane along ``axis``, or None."""
        other = [index for index in range(3) if index != axis]
        normal = np.cross(cell_array[other[0]], cell_array[other[1]])
        norm = float(np.linalg.norm(normal))
        if not np.isfinite(norm) or norm <= 1e-12:
            return None
        return normal / norm

    def _variable_cell_cut_configurations(self, a1, a2, cell):
        """Up to 12 axis-aligned cuts, best balance first.

        Only the fractional coordinate along the cut axis matters: the plane
        normal is perpendicular to the other two cell vectors. Callers rebuild
        the list every pass because translations move the groups.
        """
        centers_a1, weights = self._group_centers_and_weights(a1)
        centers_a2, _ = self._group_centers_and_weights(a2)
        if centers_a1.ndim != 2 or centers_a2.shape != centers_a1.shape:
            return []
        cell_array = np.asarray(cell, dtype=float)
        try:
            scaled_a1 = np.linalg.solve(cell_array.T, centers_a1.T).T
            scaled_a2 = np.linalg.solve(cell_array.T, centers_a2.T).T
        except np.linalg.LinAlgError:
            return []

        candidates = []
        for axis in range(self.number_of_variable_cell_vectors):
            normal = self._axis_cut_normal(cell_array, axis)
            if normal is None:
                continue
            projections = np.concatenate((scaled_a1[:, axis], scaled_a2[:, axis]))
            fractions = [0.25, 0.5, 0.75]
            if len(projections) > 1:
                proj_sorted = np.sort(projections)
                gaps = np.diff(proj_sorted)
                midpoints = proj_sorted[:-1] + gaps / 2.0
                largest = np.argsort(gaps)[-min(3, len(gaps)) :]
                fractions.extend(float(midpoints[int(idx)]) for idx in largest)
                fractions.append(float(np.median(projections)))

            unique_fractions: list[float] = []
            for value in fractions:
                if any(abs(value - existing) <= 1e-6 for existing in unique_fractions):
                    continue
                unique_fractions.append(float(value))

            for frac in unique_fractions:
                cut_p = np.zeros((1, 3))
                cut_p[0, axis] = frac
                cut_point_cart = np.dot(cut_p, cell_array)[0]
                score = self._estimate_cut_balance(
                    centers_a1,
                    centers_a2,
                    weights,
                    cut_point_cart,
                    normal,
                )
                candidates.append((score, cut_p, axis))

        candidates.sort(key=lambda item: item[0])
        return self._break_score_ties(candidates[:12])

    def _candidate_cut_configurations(self, a1, a2, cell):
        if self.number_of_variable_cell_vectors != 0:
            return self._variable_cell_cut_configurations(a1, a2, cell)

        centers_a1, weights = self._group_centers_and_weights(a1)
        centers_a2, _ = self._group_centers_and_weights(a2)
        base_cut_point = 0.5 * (
            np.mean(a1.get_positions(), axis=0) + np.mean(a2.get_positions(), axis=0)
        )
        cell_array = np.asarray(cell)
        cut_candidates = []

        for cut_normal_cart in self._candidate_cut_normals(a1, a2):
            projections = np.concatenate(
                [
                    np.dot(centers_a1 - base_cut_point, cut_normal_cart),
                    np.dot(centers_a2 - base_cut_point, cut_normal_cart),
                ]
            )
            shifts = [0.0]
            if len(projections) > 1:
                proj_sorted = np.sort(projections)
                gaps = proj_sorted[1:] - proj_sorted[:-1]
                midpoints = proj_sorted[:-1] + gaps / 2.0

                # Use the midpoints of the 3 largest gaps
                largest_gap_indices = np.argsort(gaps)[-3:]
                shifts.extend([float(midpoints[idx]) for idx in largest_gap_indices])

                # Still include the median as a robust central fallback
                shifts.append(float(np.median(projections)))
            elif len(projections) == 1:
                shifts.append(float(projections[0]))

            cut_normal = np.linalg.solve(cell_array.T, cut_normal_cart)
            for shift in shifts:
                cut_point_cart = base_cut_point + shift * cut_normal_cart
                cut_point = np.linalg.solve(cell_array.T, cut_point_cart)[np.newaxis, :]
                score = self._estimate_cut_balance(
                    centers_a1,
                    centers_a2,
                    weights,
                    cut_point_cart,
                    cut_normal_cart,
                )
                cut_candidates.append((score, cut_point, cut_normal))

        cut_candidates.sort(key=lambda item: item[0])
        return self._break_score_ties(cut_candidates[:12])

    def _jittered_cut_configurations(self, configs, a1, a2, cell):
        """Randomly offset cached cut points along their normals and re-rank.

        Each cut point is displaced along its own normal by an offset drawn
        uniformly from ±25 % of the parent group-center span projected onto
        that normal. The displaced candidates are re-scored with
        :meth:`_estimate_cut_balance` so the ranked-first-walk order is
        preserved. Configs with degenerate normals pass through unchanged.
        """
        centers_a1, weights = self._group_centers_and_weights(a1)
        centers_a2, _ = self._group_centers_and_weights(a2)
        cell_array = np.asarray(cell)
        jittered = []
        for score, cut_point, cut_normal in configs:
            cut_normal_cart = np.asarray(cut_normal, dtype=float) @ cell_array
            norm = float(np.linalg.norm(cut_normal_cart))
            if not np.isfinite(norm) or norm <= 1e-12:
                jittered.append((score, cut_point, cut_normal))
                continue
            unit = cut_normal_cart / norm
            projections = np.concatenate(
                [
                    np.dot(centers_a1, unit),
                    np.dot(centers_a2, unit),
                ]
            )
            span = float(np.max(projections) - np.min(projections))
            offset = self.rng.uniform(-0.25 * span, 0.25 * span)
            cut_point_cart = (
                np.dot(np.asarray(cut_point, dtype=float), cell_array)[0]
                + offset * unit
            )
            new_cut_point = np.linalg.solve(
                cell_array.T, cut_point_cart
            )[np.newaxis, :]
            new_score = self._estimate_cut_balance(
                centers_a1,
                centers_a2,
                weights,
                cut_point_cart,
                unit,
            )
            jittered.append((new_score, new_cut_point, cut_normal))

        jittered.sort(key=lambda item: item[0])
        return self._break_score_ties(jittered)

    def _align_moving_to_reference(self, reference, moving):
        """Align the moving parent's shape to the reference.

        Principal axes, not atom indices. The sign pattern with the smaller
        residual wins. On a surface the turn stays in the slab plane and the
        height is left alone.
        """
        reference_positions = np.asarray(reference.get_positions(), dtype=float)
        moving_positions = np.asarray(moving.get_positions(), dtype=float)
        if len(reference_positions) < 2 or len(reference_positions) != len(moving_positions):
            return
        reference_center = reference_positions.mean(axis=0)
        moving_center = moving_positions.mean(axis=0)
        centered_reference = reference_positions - reference_center
        centered_moving = moving_positions - moving_center
        plane = None
        if uses_surface(self.system_type):
            plane = [i for i in range(3) if i != self.surface_normal_axis]
        reference_frame = centered_reference if plane is None else centered_reference[:, plane]
        moving_frame = centered_moving if plane is None else centered_moving[:, plane]
        reference_axes = _principal_axes(reference_frame)
        moving_axes = _principal_axes(moving_frame)
        best_error = None
        best_aligned = None
        for signs in _proper_axis_signs(reference_axes.shape[1]):
            minor = reference_axes @ np.diag(signs) @ moving_axes.T
            if plane is None:
                rotation = minor
            else:
                rotation = np.eye(3)
                rotation[np.ix_(plane, plane)] = minor
            aligned = centered_moving @ rotation.T
            residual = aligned if plane is None else aligned[:, plane]
            target = centered_reference if plane is None else centered_reference[:, plane]
            error = float(np.linalg.norm(residual - target))
            if best_error is None or error < best_error:
                best_error = error
                best_aligned = aligned
        if plane is None:
            moving.set_positions(best_aligned + reference_center)
            return
        placed = best_aligned + moving_center
        shift = reference_center - placed.mean(axis=0)
        shift[self.surface_normal_axis] = 0.0
        moving.set_positions(placed + shift)

    def _random_relative_rotation(self, moving):
        """One large relative orientation. Surfaces rotate about the slab normal."""
        positions = np.asarray(moving.get_positions(), dtype=float)
        if len(positions) < 2:
            return
        center = positions.mean(axis=0)
        if uses_surface(self.system_type):
            axis = np.zeros(3)
            axis[int(self.surface_normal_axis)] = 1.0
        else:
            axis = _random_unit_vector(self.rng)
        angle = float(self.rng.uniform(0.5, np.pi))
        rotation = _rotation_matrix(axis, angle)
        moving.set_positions((positions - center) @ rotation.T + center)

    def update_scaling_volume(self, population, w_adapt=0.5, n_adapt=0):
        """Updates the scaling volume that is used in the pairing.

        Parameters
        ----------
        w_adapt : float
            Weight of the new vs. the old scaling volume.
        n_adapt : int
            Number of best candidates in the population that
            are used to calculate the new scaling volume.
        """
        if not n_adapt:
            # take best 20% of the population
            n_adapt = int(np.ceil(0.2 * len(population)))
        v_new = np.mean([a.get_volume() for a in population[:n_adapt]])

        if not self.scaling_volume:
            self.scaling_volume = v_new
        else:
            volumes = [self.scaling_volume, v_new]
            weights = [1 - w_adapt, w_adapt]
            self.scaling_volume = np.average(volumes, weights=weights)

    def get_new_individual(self, parents):
        """The method called by the user that
        returns the paired structure.
        """
        f, m = parents

        indi = self.cross(f, m)
        desc = f"pairing: {f.info['confid']} {m.info['confid']}"
        # It is ok for an operator to return None
        # It means that it could not make a legal offspring
        # within a reasonable amount of time
        if indi is None:
            return indi, desc
        indi = self.initialize_individual(f, indi)
        indi.info["data"]["parents"] = [f.info.get("confid"),
                                        m.info.get("confid")]

        return self.finalize_individual(indi), desc

    def cross(self, a1, a2):
        """Crosses the two atoms objects and returns one"""
        parent_full = a1
        if len(a1) != len(self.slab) + self.n_top:
            raise SCGOValidationError("Wrong size of structure to optimize")
        if len(a1) != len(a2):
            raise SCGOValidationError("The two structures do not have the same length")

        N = self.n_top

        # Only consider the atoms to optimize
        a1 = a1[len(a1) - N: len(a1)]
        a2 = a2[len(a2) - N: len(a2)]

        if not np.array_equal(a1.numbers, a2.numbers):
            err = "Trying to pair two structures with different stoichiometry"
            raise SCGOValidationError(err)

        if self.use_tags and not np.array_equal(a1.get_tags(), a2.get_tags()):
            err = "Trying to pair two structures with different tags"
            raise SCGOValidationError(err)

        cell1 = a1.get_cell()
        cell2 = a2.get_cell()
        for i in range(self.number_of_variable_cell_vectors, 3):
            err = "Unit cells are supposed to be identical in direction %d"
            if not np.allclose(cell1[i], cell2[i]):
                raise SCGOValidationError(err % i)

        counter = 0
        maxcount = self.max_pairing_attempts
        if self.number_of_variable_cell_vectors == 0:
            # Fixed-cell retries are a short chain: shape-aligned cuts, then
            # jitter, then one new orientation. ``fixed_cell_max_passes`` caps it.
            maxcount = min(maxcount, self.fixed_cell_max_passes)
        else:
            # Variable-cell retries re-rank axis-aligned cuts after each
            # translation. ``variable_cell_max_passes`` caps that chain.
            maxcount = min(maxcount, self.variable_cell_max_passes)
        a1_copy = a1.copy()
        a2_copy = a2.copy()
        # Crossover builds cartesian coordinates. SHAKE/FixBondLengths belongs
        # to local relaxation, not to pairing geometry.
        a1_copy.set_constraint()
        a2_copy.set_constraint()
        self.last_attempt_count = 0

        # The cell is drawn once. Fixed-cell passes then vary the cut and, on
        # the last pass, the relative orientation.
        newcell = self.generate_unit_cell(cell1, cell2)
        if newcell is None:
            return None

        cached_cut_configs = None

        # Run until a valid pairing is made or maxcount pairings are tested.
        while counter < maxcount:
            counter += 1

            cut_n = None
            if self.number_of_variable_cell_vectors != 0:
                # Pick one of the 'variable' cell vectors.
                cut_n = self.rng.choice(self.number_of_variable_cell_vectors)

            # Randomly translate parent structures
            for a_copy, a in zip([a1_copy, a2_copy], [a1, a2], strict=False):
                a_copy.set_positions(a.get_positions())

                cell = a_copy.get_cell()
                for i in range(self.number_of_variable_cell_vectors):
                    r = self.rng.random()
                    cond1 = i == cut_n and r < self.p1
                    cond2 = i != cut_n and r < self.p2
                    if cond1 or cond2:
                        a_copy.positions += self.rng.random() * cell[i]

                if self.use_tags:
                    # For correct determination of the center-
                    # of-position of the multi-atom blocks,
                    # we need to group their constituent atoms
                    # together
                    gather_atoms_by_tag(
                        a_copy,
                        skip_tag=periodic_sheet_tag_to_skip(self.system_type),
                    )
                else:
                    # For clusters, just center without wrapping. For slab+adsorbate
                    # the top fragment must stay in the parent frame (no global center).
                    if not uses_surface(self.system_type):
                        a_copy.center()

            if self.number_of_variable_cell_vectors == 0:
                # Positions are reset above, so passes 1 and 2 align again.
                # Pass 2 jitters the pass-1 cuts. Later passes turn once.
                if counter == 1:
                    self._align_moving_to_reference(a1_copy, a2_copy)
                    cut_configurations = self._candidate_cut_configurations(
                        a1_copy, a2_copy, newcell
                    )
                    cached_cut_configs = cut_configurations
                elif counter == 2 and cached_cut_configs:
                    self._align_moving_to_reference(a1_copy, a2_copy)
                    cut_configurations = self._jittered_cut_configurations(
                        cached_cut_configs, a1_copy, a2_copy, newcell
                    )
                else:
                    self._random_relative_rotation(a2_copy)
                    cut_configurations = self._candidate_cut_configurations(
                        a1_copy, a2_copy, newcell
                    )
            else:
                # Translations above move the groups, so the ranked cuts are
                # rebuilt instead of replaying the first pass.
                self._align_moving_to_reference(a1_copy, a2_copy)
                cut_configurations = self._candidate_cut_configurations(
                    a1_copy, a2_copy, newcell
                )

            for _score, cut_p, cut_normal in cut_configurations:
                self.last_attempt_count += 1

                child = self._get_pairing(a1_copy, a2_copy, cut_p, cut_normal, newcell)
                if child is None:
                    continue

                if self.use_tags:
                    if mobile_too_close_tagged(
                        child, self.blmin, system_type=self.system_type
                    ):
                        continue
                else:
                    if mobile_too_close_no_copy(
                        child.get_atomic_numbers(),
                        child.get_positions(),
                        child.get_pbc(),
                        child.get_cell(),
                        self.blmin,
                    ):
                        continue

                if (
                    self._slab_clash_checker is not None
                    and self._slab_clash_checker.is_too_close(
                        child.get_atomic_numbers(), child.get_positions()
                    )
                ):
                    continue

                child = self.slab + child
                _assert_offspring_integrity(child, parent_full)
                child.set_cell(newcell, scale_atoms=False)
                if not uses_surface(self.system_type):
                    child.center()
                return child

        return None

    def generate_unit_cell(self, cell1, cell2, maxcount=10000):
        """Generates a new unit cell by a random linear combination
        of the parent cells. The new cell must satisfy the
        self.cellbounds constraints. Returns None if no such cell
        was generated within a given number of attempts.

        Parameters
        ----------
        maxcount: int
            The maximal number of attempts.

        """
        # First calculate the scaling volume
        if not self.scaling_volume:
            v1 = np.abs(np.linalg.det(cell1))
            v2 = np.abs(np.linalg.det(cell2))
            r = self.rng.random()
            v_ref = r * v1 + (1 - r) * v2
        else:
            v_ref = self.scaling_volume

        # Now the cell vectors
        if self.number_of_variable_cell_vectors == 0:
            if not np.allclose(cell1, cell2):
                raise SCGOValidationError("Parent cells are not the same")
            newcell = np.copy(cell1)
            self.last_cell_attempt_count = 1
        else:
            count = 0
            while count < maxcount:
                r = self.rng.random()
                newcell = r * cell1 + (1 - r) * cell2

                vol = abs(np.linalg.det(newcell))
                scaling = v_ref / vol
                scaling **= 1. / self.number_of_variable_cell_vectors
                newcell[:self.number_of_variable_cell_vectors] *= scaling

                found = True
                if self.cellbounds is not None:
                    found = self.cellbounds.is_within_bounds(newcell)
                if found:
                    break

                count += 1
            else:
                # Did not find acceptable cell
                newcell = None
            self.last_cell_attempt_count = count if newcell is None else count + 1

        return newcell

    def _get_pairing(self, a1, a2, cutting_point, cutting_normal, cell):
        """Creates a child from two parents using the given cut.

        Returns None if the generated structure does not contain
        a large enough fraction of each parent (see self.minfrac).

        Does not check whether atoms are too close.

        Assumes the 'slab' parts have been removed from the parent
        structures and that these have been checked for equal
        lengths, stoichiometries, and tags (if self.use_tags).

        Parameters
        ----------
        cutting_normal: int or (1x3) array

        cutting_point: (1x3) array
            In fractional coordinates

        cell: (3x3) array
            The unit cell for the child structure

        """
        symbols = a1.get_chemical_symbols()
        tags = a1.get_tags() if self.use_tags else np.arange(len(a1))

        # Generate list of all atoms / atom groups:
        p1, p2, sym = [], [], []
        for tag_value in np.unique(tags):
            is_target_tag = self.target_tags is None or tag_value in self.target_tags
            indices = np.where(tags == tag_value)[0]
            s = "".join([symbols[j] for j in indices])
            sym.append(s)

            for parent_idx, (a, p) in enumerate(zip([a1, a2], [p1, p2], strict=False)):
                c = a.get_cell()
                cop = np.mean(a.positions[indices], axis=0)
                cut_p = np.dot(cutting_point, c)
                if isinstance(cutting_normal, int):
                    vecs = [c[j] for j in range(3) if j != cutting_normal]
                    cut_n = np.cross(vecs[0], vecs[1])
                else:
                    cut_n = np.dot(cutting_normal, c)
                d = np.dot(cop - cut_p, cut_n)
                # Keep non-target tag groups fixed to parent 0 when tag-aware
                # crossover is requested for only a subset (e.g., core-only).
                # Without this, those groups disappear from ``sym`` and offspring
                # generation can become impossible for odd-sized target subsets.
                # ``d > 0`` makes ``to_use()`` True only for the parent-0 copy
                # (``origin == 0``), so the parent-1 copy is deterministically
                # discarded instead of being pruned at random.
                if self.target_tags is not None and not is_target_tag:
                    d = 1.0
                spos = a.get_scaled_positions()[indices]
                scop = np.mean(spos, axis=0)
                p.append(
                    Positions(
                        spos,
                        scop,
                        s,
                        d,
                        parent_idx,
                        tag=int(tag_value),
                        n_atoms=len(indices),
                        is_target_tag=is_target_tag,
                    )
                )

        all_points = p1 + p2
        unique_sym = np.unique(sym)
        types = {s: sym.count(s) for s in unique_sym}

        # Sort these by chemical symbols:
        all_points.sort(key=lambda x: x.symbols, reverse=True)

        # For each atom type make the pairing
        unique_sym.sort()
        use_total = {}
        for s in unique_sym:
            used = []
            not_used = []
            # The list is looked trough in
            # reverse order so atoms can be removed
            # from the list along the way.
            for i in reversed(range(len(all_points))):
                # If there are no more atoms of this type
                if all_points[i].symbols != s:
                    break
                # Check if the atom should be included
                if all_points[i].to_use():
                    used.append(all_points.pop(i))
                else:
                    not_used.append(all_points.pop(i))

            if len(used) + len(not_used) != types[s] * 2:
                raise SCGOValidationError(
                    f"Internal pairing bookkeeping mismatch for type {s}"
                )

            # Too few or too many groups of this symbol: move the groups
            # closest to the cutting plane so the geometric cut stays intact.
            while len(used) < types[s]:
                used.append(_pop_closest_to_plane(not_used))

            while len(used) > types[s]:
                not_used.append(_pop_closest_to_plane(used))

            use_total[s] = used

        n_tot = sum(len(ll) for ll in use_total.values())
        if n_tot != len(sym):
            raise SCGOValidationError("Internal pairing bookkeeping mismatch")

        # Check if the generated structure contains enough atoms from both
        # parents. For target-tag crossover, enforce this on the target subset
        # only (e.g., core), not frozen non-target groups (e.g., adsorbate).
        count1, count2 = 0, 0
        N_total = len(a1)
        target_group_count = 0
        if self.target_tags is None:
            N_mix = N_total
            for x in use_total.values():
                count1 += sum(y.n_atoms for y in x if y.origin == 0)
                count2 += sum(y.n_atoms for y in x if y.origin == 1)
        else:
            N_mix = 0
            for x in use_total.values():
                for y in x:
                    if not y.is_target_tag:
                        continue
                    target_group_count += 1
                    N_mix += y.n_atoms
                    if y.origin == 0:
                        count1 += y.n_atoms
                    elif y.origin == 1:
                        count2 += y.n_atoms

        nmin = 1 if self.minfrac is None else max(1, int(np.ceil(self.minfrac * N_mix)))
        # Guard against impossible requirements for odd-sized crossover domains.
        max_feasible = N_mix // 2
        nmin = min(nmin, max_feasible)
        # If the target crossover domain has only one group (e.g., all core atoms
        # share one tag), both-parent contribution is impossible by construction.
        if self.target_tags is not None and target_group_count <= 1:
            nmin = 0
        if count1 < nmin or count2 < nmin:
            return None

        # Construct the cartesian positions and reorder the atoms
        # to follow the original order
        newpos = []
        pbc = a1.get_pbc()
        for s in sym:
            p = use_total[s].pop()
            c = a1.get_cell() if p.origin == 0 else a2.get_cell()
            pos = np.dot(p.scaled_positions, c)
            cop = np.dot(p.cop, c)
            vectors, _lengths = find_mic(pos - cop, c, pbc)
            newcop = np.dot(p.cop, cell)
            pos = newcop + vectors
            newpos.extend(pos)

        newpos = np.reshape(newpos, (N_total, 3))
        num = a1.get_atomic_numbers()
        child = Atoms(numbers=num, positions=newpos, pbc=pbc, cell=cell,
                      tags=tags)
        if not uses_surface(self.system_type):
            child.center()
        return child


class DualCutAndSplicePairing:
    """Wrapper that probabilistically delegates to one of two pairing operators.

    With probability ``exploratory_probability`` the exploratory (more
    asymmetric) cut-and-splice variant is used; otherwise the primary one.

    Parameters
    ----------
    primary : CutAndSplicePairing
        The default pairing operator.
    exploratory : CutAndSplicePairing
        A more exploratory variant (e.g. lower ``minfrac``).
    exploratory_probability : float
        Probability of choosing *exploratory* per call.
    rng : numpy.random.Generator or None
        Random number generator for the selection coin flip.
    """

    def __init__(
        self,
        primary: CutAndSplicePairing,
        exploratory: CutAndSplicePairing,
        exploratory_probability: float = 0.2,
        *,
        rng=None,
    ):
        self.primary = primary
        self.exploratory = exploratory
        self.exploratory_probability = exploratory_probability
        self.rng = _ensure_rng(rng)
        self.last_attempt_count = 0

    def get_new_individual(self, parents):
        f, _ = parents
        use_exploratory = self.rng.random() < self.exploratory_probability
        first, second = (
            (self.exploratory, self.primary)
            if use_exploratory
            else (self.primary, self.exploratory)
        )
        indi, desc = first.get_new_individual(parents)
        self.last_attempt_count = int(getattr(first, "last_attempt_count", 0))
        # A failed exploratory (or primary) cut still gets one try of the other
        # operator. A child from the first choice is kept.
        if indi is None:
            indi, desc = second.get_new_individual(parents)
            self.last_attempt_count += int(getattr(second, "last_attempt_count", 0))
        if indi is not None and len(indi) != len(f):
            raise SCGOValidationError(
                "DualCutAndSplicePairing produced "
                f"{len(indi)} atoms; expected {len(f)}."
            )
        return indi, desc

    def update_scaling_volume(self, population, **kwargs):
        self.primary.update_scaling_volume(population, **kwargs)
        self.exploratory.update_scaling_volume(population, **kwargs)


# fmt: on
