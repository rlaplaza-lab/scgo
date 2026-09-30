"""Surface-NEB helpers: moiety unwrap, slab symmetries, variable springs, fidelity.

Gated by surface PBC (slab prefix or exactly two periodic axes). Gas bands are
untouched. Does not change NEB ``status``; fidelity fields are diagnostic only.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable, Sequence
from math import sqrt
from typing import Any

import numpy as np
from ase import Atoms, units
from ase.constraints import FixAtoms
from ase.geometry import find_mic

from scgo.exceptions import SCGOValidationError
from scgo.initialization.geometry_helpers import (
    _bonded_pairs,
    _union_find_components,
)
from scgo.metadata.atoms import get_tag
from scgo.system_types.connectivity_factor import ConnectivityFactorInput
from scgo.utils.logging import get_logger

# Symmetry-aligned product single-point may not match the GO minimum.
SYMMETRY_DRIFT_EV: float = 0.1
# Variable spring range (eV/Å²); k_min is the band's existing scalar.
K_MAX: float = 4.0
K_UPDATE_EVERY: int = 40
# Finite-difference Hessian for imaginary-mode count.
HESSIAN_MAX_MOBILE: int = 12
FD_DELTA_A: float = 0.01
IMAGINARY_CUTOFF_CM: float = -50.0
# ASE Vibrations conversion (see ase.vibrations.data / vibrations.py).
_HBAR_TO_ENERGY = units._hbar * units.m / sqrt(units._e * units._amu)

logger = get_logger(__name__)


def _requires_surface_pbc_alignment(reactant: Atoms, *, n_slab: int) -> bool:
    """True when surface MIC / in-plane lattice logic applies."""
    if int(n_slab) > 0:
        return True
    pbc = np.asarray(reactant.pbc, dtype=bool)
    return int(np.count_nonzero(pbc)) == 2


def _cell_array(cell: Any) -> np.ndarray:
    if hasattr(cell, "array"):
        return np.asarray(cell.array, dtype=float)
    return np.asarray(cell, dtype=float)


def _infer_surface_normal_axis(pbc: np.ndarray | list[bool]) -> int:
    pbc_arr = np.asarray(pbc, dtype=bool)
    open_axes = [i for i in range(3) if not pbc_arr[i]]
    if len(open_axes) == 1:
        return int(open_axes[0])
    return 2


def _inplane_periodic_axes(pbc: np.ndarray | list[bool]) -> tuple[int, int]:
    pbc_arr = np.asarray(pbc, dtype=bool)
    periodic = [i for i in range(3) if pbc_arr[i]]
    if len(periodic) == 2:
        return int(periodic[0]), int(periodic[1])
    return 0, 1


def _pbc_for_mic_alignment(pbc: np.ndarray | list[bool]) -> np.ndarray:
    pbc_arr = np.asarray(pbc, dtype=bool).copy()
    pbc_arr[_infer_surface_normal_axis(pbc_arr)] = False
    return pbc_arr


def half_inplane_cell_length(cell: Any, pbc: np.ndarray | list[bool]) -> float:
    """Half the shorter in-plane lattice vector (MIC rewrap threshold)."""
    cell_arr = _cell_array(cell)
    axis_a, axis_b = _inplane_periodic_axes(_pbc_for_mic_alignment(pbc))
    la = float(np.linalg.norm(cell_arr[axis_a]))
    lb = float(np.linalg.norm(cell_arr[axis_b]))
    return 0.5 * min(la, lb)


def unwrap_breaks_mic(
    pre_unwrap: np.ndarray,
    unwrapped: np.ndarray,
    *,
    cell: Any,
    pbc: np.ndarray | list[bool],
) -> bool:
    """True when a moiety unwrap is large enough that ASE ``mic=True`` would undo it.

    Tiny bonded-fragment polishes (≪ half-cell) must keep MIC interpolation:
    disabling MIC for sub-Å unwraps turns surface_cluster IDPP bands into
    multi-tens-of-eV discontinuous paths (Pt5-on-graphite regression).
    """
    delta = np.asarray(unwrapped, dtype=float) - np.asarray(pre_unwrap, dtype=float)
    if delta.size == 0:
        return False
    max_disp = float(np.max(np.linalg.norm(delta, axis=1)))
    if max_disp <= 1e-8:
        return False
    return max_disp > half_inplane_cell_length(cell, pbc) - 1e-6


def _bond_edge_set(
    atoms: Atoms,
    connectivity_factor: ConnectivityFactorInput | None,
    *,
    use_mic: bool,
) -> set[tuple[int, int]]:
    i_idx, j_idx = _bonded_pairs(atoms, connectivity_factor, use_mic=use_mic)
    return {
        (int(i), int(j)) for i, j in zip(i_idx.tolist(), j_idx.tolist(), strict=True)
    }


def _fixed_atom_indices(atoms: Atoms) -> set[int]:
    """Indices frozen by ``FixAtoms`` constraints (empty if none)."""
    out: set[int] = set()
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            out.update(int(i) for i in constraint.get_indices())
    return out


def consistent_product_positions(
    reactant: Atoms,
    product_positions: np.ndarray,
    *,
    n_slab: int = 0,
    connectivity_factor: ConnectivityFactorInput | None = None,
    max_lattice_shift: int = 1,
) -> np.ndarray:
    """Unwrap intact mobile fragments so IDPP does not split them across PBC.

    Intact edges are covalent bonds present under MIC in both endpoints. Lone
    adatoms (size-1 components) keep per-atom MIC. Anchors (slab prefix and
    ``FixAtoms``) stay on the reactant coordinates. The in-plane image search
    span matches ``neb_surface_max_lattice_shift`` (default ±1 cell).
    """
    prod = np.asarray(product_positions, dtype=float).copy()
    if not _requires_surface_pbc_alignment(reactant, n_slab=n_slab):
        return prod

    n_atoms = len(reactant)
    n_slab_i = max(0, int(n_slab))
    if n_slab_i >= n_atoms:
        return prod

    product = reactant.copy()
    product.set_positions(prod, apply_constraint=False)

    anchor = set(range(n_slab_i)) | _fixed_atom_indices(reactant)
    react_edges = _bond_edge_set(reactant, connectivity_factor, use_mic=True)
    prod_edges = _bond_edge_set(product, connectivity_factor, use_mic=True)
    intact = {
        e
        for e in (react_edges & prod_edges)
        if e[0] not in anchor and e[1] not in anchor
    }
    if not intact:
        return prod

    i_list = np.array([e[0] for e in intact], dtype=int)
    j_list = np.array([e[1] for e in intact], dtype=int)
    components = _union_find_components(n_atoms, i_list, j_list)

    cell = _cell_array(reactant.cell)
    pbc_mic = _pbc_for_mic_alignment(reactant.pbc)
    axis_a, axis_b = _inplane_periodic_axes(pbc_mic)
    a_vec = cell[axis_a]
    b_vec = cell[axis_b]
    shift_span = max(0, int(max_lattice_shift))
    images = [
        n1 * a_vec + n2 * b_vec
        for n1 in range(-shift_span, shift_span + 1)
        for n2 in range(-shift_span, shift_span + 1)
    ]

    adjacency: dict[int, list[int]] = defaultdict(list)
    for i, j in intact:
        adjacency[i].append(j)
        adjacency[j].append(i)

    ref_pos = reactant.get_positions()
    out = prod.copy()
    for idx in anchor:
        out[idx] = ref_pos[idx]

    for members in components.values():
        mobile_members = [m for m in members if m not in anchor]
        if len(mobile_members) < 2:
            continue
        root = min(mobile_members)
        placed = {root}
        queue: deque[int] = deque([root])
        while queue:
            parent = queue.popleft()
            for nb in adjacency[parent]:
                if nb in anchor or nb in placed:
                    continue
                target_vec = ref_pos[nb] - ref_pos[parent]
                best_pos = out[nb]
                best_err = float("inf")
                for shift in images:
                    cand = prod[nb] + shift
                    err = float(np.linalg.norm((cand - out[parent]) - target_vec))
                    if err < best_err:
                        best_err = err
                        best_pos = cand
                out[nb] = best_pos
                placed.add(nb)
                queue.append(nb)

    return out


def _rotation_about_normal(angle_deg: float, normal_axis: int) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    c, s = float(np.cos(angle)), float(np.sin(angle))
    rot = np.eye(3, dtype=float)
    axes = [0, 1, 2]
    axes.remove(normal_axis)
    i, j = axes[0], axes[1]
    rot[i, i] = c
    rot[i, j] = -s
    rot[j, i] = s
    rot[j, j] = c
    return rot


def _slab_maps_to_itself(
    slab_pos: np.ndarray,
    rot: np.ndarray,
    center: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    *,
    tol: float = 0.1,
) -> bool:
    """True when rotating slab about ``center`` maps every atom onto a slab atom."""
    if len(slab_pos) == 0:
        return False
    rotated = (slab_pos - center) @ rot.T + center
    for atom in rotated:
        disp, _ = find_mic(slab_pos - atom, cell=cell, pbc=pbc)
        if float(np.linalg.norm(disp, axis=1).min()) > tol:
            return False
    return True


def inplane_symmetry_matrices(
    reactant: Atoms,
    *,
    n_slab: int = 0,
) -> list[np.ndarray]:
    """Discrete proper in-plane rotations validated on this slab.

    Identity is always first. Non-identity ops are kept only when they map every
    slab atom onto another slab atom under MIC within 0.1 Å. Orthogonal cells
    with unequal edges still propose 180° when the slab maps.
    """
    identity = np.eye(3, dtype=float)
    n_slab_i = max(0, int(n_slab))
    if n_slab_i <= 0 or not _requires_surface_pbc_alignment(reactant, n_slab=n_slab_i):
        return [identity]

    cell = _cell_array(reactant.cell)
    pbc_mic = _pbc_for_mic_alignment(reactant.pbc)
    axis_a, axis_b = _inplane_periodic_axes(pbc_mic)
    normal_axis = _infer_surface_normal_axis(reactant.pbc)
    a_vec = cell[axis_a]
    b_vec = cell[axis_b]
    la = float(np.linalg.norm(a_vec))
    lb = float(np.linalg.norm(b_vec))
    if la < 1e-12 or lb < 1e-12:
        return [identity]
    cos_ab = float(np.dot(a_vec, b_vec) / (la * lb))
    cos_ab = max(-1.0, min(1.0, cos_ab))
    theta = float(np.rad2deg(np.arccos(cos_ab)))
    equal_len = abs(la - lb) / max(la, lb) < 1e-6
    orthogonal = abs(theta - 90.0) <= 1.0

    candidates: list[np.ndarray] = [identity]
    if equal_len and (abs(theta - 60.0) <= 1.0 or abs(theta - 120.0) <= 1.0):
        for ang in (60.0, 120.0, 180.0, 240.0, 300.0):
            candidates.append(_rotation_about_normal(ang, normal_axis))
    elif equal_len and orthogonal:
        for ang in (90.0, 180.0, 270.0):
            candidates.append(_rotation_about_normal(ang, normal_axis))
    elif orthogonal:
        # ASE fcc111(..., orthogonal=True) and other rectangular cells.
        candidates.append(_rotation_about_normal(180.0, normal_axis))
    else:
        return [identity]

    ref_pos = reactant.get_positions()
    slab_pos = ref_pos[:n_slab_i]
    mobile = ref_pos[n_slab_i:]
    com = mobile.mean(axis=0) if len(mobile) > 0 else slab_pos.mean(axis=0)
    disp, _ = find_mic(slab_pos - com, cell=cell, pbc=pbc_mic)
    anchor_idx = int(np.argmin(np.linalg.norm(disp, axis=1)))
    center = slab_pos[anchor_idx]

    kept = [identity]
    for rot in candidates[1:]:
        if _slab_maps_to_itself(slab_pos, rot, center, cell, pbc_mic, tol=0.1):
            kept.append(rot)
    return kept


def apply_inplane_symmetry(
    positions: np.ndarray,
    rot: np.ndarray,
    *,
    center: np.ndarray,
    mobile_mask: np.ndarray,
    anchor_mask: np.ndarray,
    ref_pos: np.ndarray,
) -> np.ndarray:
    """Rotate mobile atoms about ``center``; reset anchors to ``ref_pos``."""
    out = np.asarray(positions, dtype=float).copy()
    out[mobile_mask] = (positions[mobile_mask] - center) @ rot.T + center
    out[anchor_mask] = ref_pos[anchor_mask]
    return out


def symmetry_anchor_center(
    reactant: Atoms,
    *,
    n_slab: int,
) -> np.ndarray | None:
    """Slab atom nearest (MIC) to the reactant mobile COM, or None."""
    n_slab_i = max(0, int(n_slab))
    if n_slab_i <= 0:
        return None
    ref_pos = reactant.get_positions()
    mobile = ref_pos[n_slab_i:]
    if len(mobile) == 0:
        return None
    cell = _cell_array(reactant.cell)
    pbc_mic = _pbc_for_mic_alignment(reactant.pbc)
    slab_pos = ref_pos[:n_slab_i]
    com = mobile.mean(axis=0)
    disp, _ = find_mic(slab_pos - com, cell=cell, pbc=pbc_mic)
    anchor_idx = int(np.argmin(np.linalg.norm(disp, axis=1)))
    return slab_pos[anchor_idx].copy()


def symmetry_product_rejected(
    product_energy: float | None,
    reference_product_energy: float | None,
    *,
    used_symmetry_copy: bool,
) -> bool:
    """True when a non-identity symmetry copy drifted above ``SYMMETRY_DRIFT_EV``."""
    if (
        not used_symmetry_copy
        or product_energy is None
        or reference_product_energy is None
    ):
        return False
    return (
        abs(float(product_energy) - float(reference_product_energy)) > SYMMETRY_DRIFT_EV
    )


def ensure_symmetry_copy_energy(
    product_energy: float | None,
    reference_product_energy: float | None,
    *,
    used_symmetry_copy: bool,
) -> None:
    """Raise when a non-identity symmetry copy is not the stored product minimum.

    Uses the same drift limit as :func:`symmetry_product_rejected`
    (``SYMMETRY_DRIFT_EV``).

    Raises:
        SCGOValidationError: If the drift check fails.
    """
    if not symmetry_product_rejected(
        product_energy,
        reference_product_energy,
        used_symmetry_copy=used_symmetry_copy,
    ):
        return
    drift = abs(float(product_energy) - float(reference_product_energy))
    raise SCGOValidationError(
        "Initial NEB path rejected (symmetry copy): "
        f"aligned product energy drifted by {drift:.3f} eV "
        f"(limit {SYMMETRY_DRIFT_EV:.3f} eV)"
    )


def variable_spring_constants(energies: np.ndarray, k_min: float) -> np.ndarray:
    """Energy-weighted springs (length ``n_images - 1``), NEBscape / Ásgeirsson form.

    Peak stiffness is ``max(K_MAX, k_min)`` so ``k_min > K_MAX`` cannot invert
    the barrier.
    """
    e = np.asarray(energies, dtype=float)
    n_spring = max(0, e.size - 1)
    k_min_f = float(k_min)
    k_peak = max(K_MAX, k_min_f)
    if n_spring == 0:
        return np.zeros(0, dtype=float)
    e_ref = float(max(e[0], e[-1]))
    e_max = float(np.max(e))
    out = np.full(n_spring, k_min_f, dtype=float)
    denom = e_max - e_ref
    if denom < 1e-8:
        return out
    for i in range(n_spring):
        e_i = float(max(e[i], e[i + 1]))
        if e_i > e_ref:
            out[i] = k_peak - (k_peak - k_min_f) * (e_max - e_i) / denom
    return out


def refresh_surface_neb_springs(neb: Any, *, k_min: float) -> None:
    """Set ``neb.k`` from image energies when the band is surface-like.

    Prefers the calculator energy; falls back to the ``potential_energy`` tag
    written by TorchSim / ASE single-point attachment (ASE FIRE invalidates
    ``SinglePointCalculator`` after a step). Skips the update when any image
    lacks both.
    """
    images = getattr(neb, "images", None)
    if not images:
        return
    n_slab = int(getattr(neb, "_scgo_n_slab", 0) or 0)
    if not _requires_surface_pbc_alignment(images[0], n_slab=n_slab):
        return
    energies: list[float] = []
    for img in images:
        try:
            energy = float(img.get_potential_energy())
        except Exception:
            stored = get_tag(img, "potential_energy", default=None)
            if stored is None:
                return
            energy = float(stored)
        energies.append(energy)
    neb.k = variable_spring_constants(np.asarray(energies, dtype=float), k_min)


def bond_set(
    atoms: Atoms,
    *,
    n_slab: int = 0,
    connectivity_factor: ConnectivityFactorInput | None = None,
) -> frozenset[tuple[int, int]]:
    """Mobile–mobile covalent pairs (MIC), indices at or above ``n_slab``."""
    n_slab_i = max(0, int(n_slab))
    edges = _bond_edge_set(atoms, connectivity_factor, use_mic=True)
    return frozenset(e for e in edges if e[0] >= n_slab_i and e[1] >= n_slab_i)


def classify_band(
    images: Sequence[Atoms],
    energies: Sequence[float] | np.ndarray,
    *,
    n_slab: int = 0,
    connectivity_factor: ConnectivityFactorInput | None = None,
) -> tuple[bool | None, bool | None]:
    """Return ``(fidelity_single_step, fidelity_energy_at_bond_change)``.

    Both are ``None`` when endpoints share the same bond set (no bond change).
    """
    if len(images) < 2:
        return None, None
    e = np.asarray(energies, dtype=float)
    if e.size != len(images) or not np.all(np.isfinite(e)):
        return None, None

    react_bonds = bond_set(
        images[0], n_slab=n_slab, connectivity_factor=connectivity_factor
    )
    prod_bonds = bond_set(
        images[-1], n_slab=n_slab, connectivity_factor=connectivity_factor
    )
    if react_bonds == prod_bonds:
        return None, None

    labels: list[str] = []
    for img in images:
        b = bond_set(img, n_slab=n_slab, connectivity_factor=connectivity_factor)
        if b == react_bonds:
            labels.append("reactant")
        elif b == prod_bonds:
            labels.append("product")
        else:
            labels.append("transition")

    transition_idx = [i for i, lab in enumerate(labels) if lab == "transition"]
    if not transition_idx:
        return False, False

    a, b = transition_idx[0], transition_idx[-1]
    contiguous = transition_idx == list(range(a, b + 1))
    max_idx = int(np.argmax(e))
    energy_in_transition = a <= max_idx <= b

    unimodal = True
    if b - a >= 2:
        for i in range(a + 1, b):
            if e[i] < e[i - 1] and e[i] < e[i + 1]:
                unimodal = False
                break

    single_step = bool(contiguous and energy_in_transition and unimodal)
    return single_step, bool(energy_in_transition)


def displaced_mobile_images(
    atoms: Atoms,
    *,
    n_slab: int = 0,
    delta: float = FD_DELTA_A,
) -> list[Atoms]:
    """``6 * n_mobile`` copies for central-difference Hessian (mobile only)."""
    n_slab_i = max(0, int(n_slab))
    n_atoms = len(atoms)
    mobile = list(range(n_slab_i, n_atoms))
    out: list[Atoms] = []
    pos0 = atoms.get_positions()
    for idx in mobile:
        for c in range(3):
            for sign in (+1.0, -1.0):
                img = atoms.copy()
                pos = pos0.copy()
                pos[idx, c] += sign * float(delta)
                img.set_positions(pos, apply_constraint=False)
                out.append(img)
    return out


def count_imaginary_frequencies(
    omega2: np.ndarray,
    *,
    cutoff_cm: float = IMAGINARY_CUTOFF_CM,
) -> int:
    """Count mass-weighted Hessian eigenvalues below ``cutoff_cm`` (cm^-1)."""
    # ASE VibrationsData: omega2 -> complex energies (eV) -> cm^-1; imag -> signed.
    energies = _HBAR_TO_ENERGY * np.asarray(omega2, dtype=complex) ** 0.5
    freqs = energies / units.invcm
    signed = np.where(
        np.abs(freqs.imag) > 1e-8,
        -np.abs(freqs.imag),
        freqs.real,
    )
    return int(np.sum(signed < float(cutoff_cm)))


def mass_weighted_hessian_omega2(
    hessian: np.ndarray,
    masses: np.ndarray,
) -> np.ndarray:
    """Return eigenvalues of ``M^{-1/2} H M^{-1/2}`` (ASE convention)."""
    masses = np.asarray(masses, dtype=float)
    im = np.repeat(masses**-0.5, 3)
    mw = im[:, None] * np.asarray(hessian, dtype=float) * im
    omega2, _ = np.linalg.eigh(mw)
    return omega2


def finite_difference_hessian(
    atoms: Atoms,
    force_fn: Callable[[list[Atoms]], list[np.ndarray]],
    *,
    n_slab: int = 0,
    delta: float = FD_DELTA_A,
) -> np.ndarray:
    """Cartesian Hessian on mobile atoms from central differences of forces.

    ``H[:, j] = (F(x - d e_j) - F(x + d e_j)) / (2d)`` so a restoring force
    ``F = -k x`` yields positive curvature.
    """
    n_slab_i = max(0, int(n_slab))
    mobile = list(range(n_slab_i, len(atoms)))
    n_dof = 3 * len(mobile)
    displaced = displaced_mobile_images(atoms, n_slab=n_slab_i, delta=delta)
    force_list = force_fn(displaced)
    if len(force_list) != len(displaced):
        raise ValueError(
            f"force_fn returned {len(force_list)} force arrays for "
            f"{len(displaced)} displaced images"
        )
    h = np.zeros((n_dof, n_dof), dtype=float)
    # displaced_mobile_images order: atom, component, (+, -).
    pair = 0
    for m_i in range(len(mobile)):
        for c in range(3):
            f_plus = np.asarray(force_list[pair], dtype=float).reshape(-1, 3)
            f_minus = np.asarray(force_list[pair + 1], dtype=float).reshape(-1, 3)
            pair += 2
            col = m_i * 3 + c
            fp = f_plus[mobile].ravel()
            fm = f_minus[mobile].ravel()
            h[:, col] = (fm - fp) / (2.0 * float(delta))
    return h


def count_imaginary_modes_from_forces(
    atoms: Atoms,
    force_fn: Callable[[list[Atoms]], list[np.ndarray]],
    *,
    n_slab: int = 0,
) -> int:
    """Finite-difference imaginary-mode count for mobile atoms."""
    n_slab_i = max(0, int(n_slab))
    mobile = list(range(n_slab_i, len(atoms)))
    if not mobile:
        return 0
    hessian = finite_difference_hessian(atoms, force_fn, n_slab=n_slab_i)
    masses = atoms.get_masses()[mobile]
    omega2 = mass_weighted_hessian_omega2(hessian, masses)
    return count_imaginary_frequencies(omega2)
