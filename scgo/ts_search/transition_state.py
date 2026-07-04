"""Utilities for finding transition states with NEB and path interpolation."""

from __future__ import annotations

import contextlib
import json
import os
import sys
from copy import deepcopy
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms
from ase.geometry import find_mic
from ase.io import write
from ase.mep import NEB
from ase.optimize import FIRE
from ase.optimize.optimize import Optimizer
from scipy.optimize import linear_sum_assignment

from scgo.calculators import torchsim_helpers as _tsh
from scgo.constants import (
    DEFAULT_COMPARATOR_TOL,
    DEFAULT_NEB_TANGENT_METHOD,
    DEFAULT_PAIR_COR_MAX,
)
from scgo.database.metadata import get_metadata
from scgo.system_types import SystemType, get_system_policy
from scgo.utils.helpers import extract_energy_from_atoms
from scgo.utils.logging import get_logger
from scgo.utils.run_helpers import cleanup_torch_cuda
from scgo.utils.timing_report import (
    build_timing_payload,
    log_timing_summary,
    write_timing_file,
)
from scgo.utils.torchsim_policy import (
    _require_torchsim,
    _require_torchsim_fairchem,
    is_uma_like_calculator,
)
from scgo.utils.ts_provenance import is_cuda_oom_error, ts_output_provenance
from scgo.utils.validation import validate_atoms, validate_calculator_attached

if TYPE_CHECKING:
    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer


def _detach_calc(atoms: Atoms | None) -> None:
    """Remove calculator from structure when present."""
    if atoms is None:
        return
    with contextlib.suppress(AttributeError, TypeError):
        atoms.calc = None


def attach_singlepoint_from_relax_output(
    atoms: Atoms,
    energy: float,
    relaxed_atoms: Atoms,
    *,
    require_forces: bool = True,
) -> None:
    """Attach ``SinglePointCalculator`` to ``atoms`` from one ``relax_batch`` result."""
    forces = relaxed_atoms.arrays.get("forces")
    if forces is None and relaxed_atoms.calc is not None:
        with contextlib.suppress(AttributeError, NotImplementedError):
            forces = relaxed_atoms.get_forces()
    if forces is not None and getattr(forces, "size", 0) > 0:
        atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
        return
    if require_forces:
        raise RuntimeError(
            "TorchSim did not return forces. Ensure the model is loaded with compute_forces=True."
        )
    atoms.calc = SinglePointCalculator(atoms, energy=energy)


def _image_has_cached_forces(img: Atoms) -> bool:
    """True when ``img`` already carries PES forces (array or calculator cache)."""
    if img.arrays.get("forces") is not None:
        return True
    calc = img.calc
    if calc is None:
        return False
    with contextlib.suppress(AttributeError, NotImplementedError, RuntimeError):
        return calc.get_forces(img) is not None
    return False


def calculate_structure_similarity(
    atoms1: Atoms,
    atoms2: Atoms,
    tolerance: float = DEFAULT_COMPARATOR_TOL,
    pair_cor_max: float = DEFAULT_PAIR_COR_MAX,
    *,
    ignore_fixed_atoms: bool = True,
    use_mic: bool = False,
    n_slab: int | None = None,
) -> tuple[float, float, bool]:
    """Return (cum_diff, max_diff, are_similar) comparing two Atoms; raises ValueError if counts differ."""
    from scgo.utils.comparators import (
        PureInteratomicDistanceComparator,
        get_shared_mobile_atom_indices,
    )

    if len(atoms1) != len(atoms2):
        raise ValueError(
            f"Atoms objects have different lengths: {len(atoms1)} vs {len(atoms2)}"
        )

    if ignore_fixed_atoms:
        comparison_indices = get_shared_mobile_atom_indices(
            atoms1,
            atoms2,
            n_slab=n_slab,
        )
    else:
        comparison_indices = np.arange(len(atoms1), dtype=int)
    atoms1_cmp = atoms1[comparison_indices]
    atoms2_cmp = atoms2[comparison_indices]

    comparator = PureInteratomicDistanceComparator(
        n_top=len(atoms1_cmp),
        tol=tolerance,
        pair_cor_max=pair_cor_max,
        mic=use_mic,
    )

    cum_diff, max_diff = comparator.get_differences(atoms1_cmp, atoms2_cmp)
    are_similar = comparator.looks_like(atoms1_cmp, atoms2_cmp)

    return cum_diff, max_diff, are_similar


class TorchSimNEB(NEB):
    """NEB that batches PES evaluations via TorchSim for GPU efficiency."""

    def __init__(
        self,
        images: list[Atoms],
        relaxer: TorchSimBatchRelaxer,
        k: float | list[float] = 0.1,
        climb: bool = False,
        parallel: bool = False,
        remove_rotation_and_translation: bool = False,
        method: str = DEFAULT_NEB_TANGENT_METHOD,
    ):
        """Initialize NEB with images and a TorchSimBatchRelaxer."""
        super().__init__(
            images,
            k=k,
            climb=climb,
            parallel=parallel,
            remove_rotation_and_translation=remove_rotation_and_translation,
            method=method,
        )
        self.relaxer = relaxer
        self._force_calls = 0

    def get_forces(self) -> np.ndarray:
        """Batch-evaluate PES forces with TorchSim and return NEB forces.

        When images already carry PES forces (for example because
        ``ParallelNEBBatch`` just evaluated them in a single batched call),
        reuse the cached arrays instead of re-invoking TorchSim.
        """
        if all(_image_has_cached_forces(img) for img in self.images):
            return super().get_forces()

        self._force_calls += 1
        results = self.relaxer.relax_batch(self.images, steps=0)

        for atoms, (energy, relaxed_atoms) in zip(self.images, results, strict=True):
            attach_singlepoint_from_relax_output(
                atoms, energy, relaxed_atoms, require_forces=True
            )

        return super().get_forces()

    def get_force_calls(self) -> int:
        """Return the number of times forces have been evaluated."""
        return self._force_calls


def _local_distance_fingerprints(atoms: Atoms) -> np.ndarray:
    """Return per-atom sorted distance fingerprint (shape: n_atoms x (n_atoms-1)).

    The fingerprint is used only for robust endpoint atom matching and is
    intentionally simple and deterministic (no RNG).
    """
    pos = atoms.get_positions()
    n = len(atoms)
    fp = np.zeros((n, max(0, n - 1)), dtype=float)
    for i in range(n):
        d = np.linalg.norm(pos - pos[i], axis=1)
        d = np.delete(d, i)
        d.sort()
        if d.size > 0:
            fp[i, : d.size] = d
    return fp


def _local_distance_fingerprints_mic(
    atoms: Atoms,
    cell: np.ndarray,
    pbc: np.ndarray | list[bool],
) -> np.ndarray:
    """MIC-aware distance fingerprints for periodic endpoint matching."""
    pos = atoms.get_positions()
    n = len(atoms)
    fp = np.zeros((n, max(0, n - 1)), dtype=float)
    for i in range(n):
        disp = pos - pos[i]
        disp_mic, _ = find_mic(disp, cell=cell, pbc=pbc)
        d = np.linalg.norm(disp_mic, axis=1)
        d = np.delete(d, i)
        d.sort()
        if d.size > 0:
            fp[i, : d.size] = d
    return fp


def _mic_matching_context(
    reactant: Atoms,
    *,
    n_slab: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return (cell, pbc) for MIC-aware fingerprint matching, or (None, None)."""
    if not _requires_surface_pbc_alignment(reactant, n_slab=n_slab):
        return None, None
    return _cell_array(reactant.cell), _pbc_for_mic_alignment(reactant.pbc)


def _match_atoms_by_fingerprint(
    a1: Atoms,
    a2: Atoms,
    *,
    mic_cell: np.ndarray | None = None,
    mic_pbc: np.ndarray | list[bool] | None = None,
) -> list[int]:
    """Return mapping such that mapped_idx[i] is index in `a2` matching atom i in `a1`.

    Uses per-atom local-distance fingerprints and the Hungarian algorithm to
    obtain a permutation that is robust to rotations and permutations.
    When ``mic_cell`` and ``mic_pbc`` are set, fingerprints use minimum-image
    distances (required for slab endpoints near periodic boundaries).
    """
    if len(a1) != len(a2):
        raise ValueError("Atoms objects have different lengths")

    mapping = [-1] * len(a1)
    use_mic = mic_cell is not None and mic_pbc is not None
    if use_mic:
        fp1_all = _local_distance_fingerprints_mic(a1, mic_cell, mic_pbc)
        fp2_all = _local_distance_fingerprints_mic(a2, mic_cell, mic_pbc)
    else:
        fp1_all = _local_distance_fingerprints(a1)
        fp2_all = _local_distance_fingerprints(a2)
    # Match separately for each atomic number (handles mixed-species clusters)
    for z in set(a1.numbers):
        idx1 = [i for i, x in enumerate(a1.numbers) if x == z]
        idx2 = [i for i, x in enumerate(a2.numbers) if x == z]
        if len(idx1) != len(idx2):
            raise ValueError("Composition mismatch during endpoint matching")

        fp1 = fp1_all[idx1]
        fp2 = fp2_all[idx2]
        # Cost = L2 distance between fingerprints
        cost = np.linalg.norm(fp1[:, None, :] - fp2[None, :, :], axis=2)
        r, c = linear_sum_assignment(cost)
        for ri, ci in zip(r, c, strict=False):
            mapping[idx1[ri]] = idx2[ci]

    return mapping


def _reorder_block_positions_to_match(
    a1_block: Atoms,
    a2_block: Atoms,
    *,
    mic_cell: np.ndarray | None = None,
    mic_pbc: np.ndarray | list[bool] | None = None,
) -> np.ndarray:
    """Return positions (N,3) for a2_block reordered to match a1_block ordering."""
    m = _match_atoms_by_fingerprint(
        a1_block, a2_block, mic_cell=mic_cell, mic_pbc=mic_pbc
    )
    pos2 = a2_block.get_positions()
    return pos2[m]


def _permute_atoms_block_to_match(
    a1_block: Atoms,
    a2_block: Atoms,
    *,
    mic_cell: np.ndarray | None = None,
    mic_pbc: np.ndarray | list[bool] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (positions, atomic_numbers) for a2_block permuted to match a1_block."""
    mapping = _match_atoms_by_fingerprint(
        a1_block, a2_block, mic_cell=mic_cell, mic_pbc=mic_pbc
    )
    pos2 = a2_block.get_positions()
    nums2 = a2_block.numbers
    return pos2[mapping], nums2[mapping]


def _align_endpoints_blockwise(
    a1: Atoms,
    a2: Atoms,
    n_slab: int,
    n_core: int,
    n_ads: int,
    *,
    mic_cell: np.ndarray | None = None,
    mic_pbc: np.ndarray | list[bool] | None = None,
) -> None:
    """Match product to reactant per block (slab indices unchanged; core/ads via fingerprint)."""
    n = len(a1)
    if len(a2) != n:
        raise ValueError("align blockwise: endpoint length mismatch")
    if n_slab + n_core + n_ads != n:
        raise ValueError(
            f"align blockwise: n_slab+n_core+n_ads={n_slab + n_core + n_ads} != len={n}"
        )
    p2 = a2.get_positions().copy()
    n2 = a2.numbers.copy()
    if n_core > 0:
        s1, s2 = n_slab, n_slab + n_core
        p_blk, n_blk = _permute_atoms_block_to_match(
            a1[s1:s2], a2[s1:s2], mic_cell=mic_cell, mic_pbc=mic_pbc
        )
        p2[s1:s2] = p_blk
        n2[s1:s2] = n_blk
    if n_ads > 0:
        t1 = n_slab + n_core
        t2 = t1 + n_ads
        p_blk, n_blk = _permute_atoms_block_to_match(
            a1[t1:t2], a2[t1:t2], mic_cell=mic_cell, mic_pbc=mic_pbc
        )
        p2[t1:t2] = p_blk
        n2[t1:t2] = n_blk
    a2.set_positions(p2)
    a2.numbers = n2


def _kabsch_rotation(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Return rotation matrix R that minimizes ||P - Q @ R|| (P and Q are centered)."""
    dim = int(P.shape[1])
    U, _, Vt = np.linalg.svd(P.T @ Q)
    d = np.ones(dim, dtype=float)
    d[-1] = float(np.sign(np.linalg.det(U @ Vt)) or 1.0)
    return U @ np.diag(d) @ Vt


def _kabsch_rotation_in_plane(
    P: np.ndarray, Q: np.ndarray, *, surface_normal_axis: int = 2
) -> np.ndarray:
    """Return 3x3 rotation that aligns Q to P using only in-plane degrees of freedom."""
    if surface_normal_axis not in (0, 1, 2):
        raise ValueError("surface_normal_axis must be 0, 1, or 2")
    plane_axes = [i for i in range(3) if i != surface_normal_axis]
    r2 = _kabsch_rotation(P[:, plane_axes], Q[:, plane_axes])
    rot = np.eye(3)
    for i, ia in enumerate(plane_axes):
        for j, ja in enumerate(plane_axes):
            rot[ia, ja] = r2[i, j]
    return rot


def _infer_surface_normal_axis(pbc: np.ndarray | list[bool]) -> int:
    """Guess vacuum/normal axis as the sole non-periodic direction, else z."""
    pbc_arr = np.asarray(pbc, dtype=bool)
    open_axes = [i for i in range(3) if not pbc_arr[i]]
    if len(open_axes) == 1:
        return int(open_axes[0])
    return 2


def _fixed_atom_mask(atoms: Atoms) -> np.ndarray:
    """Return a boolean mask for atoms fixed by ``FixAtoms`` constraints."""
    mask = np.zeros(len(atoms), dtype=bool)
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            idx = np.asarray(constraint.get_indices(), dtype=int)
            mask[idx] = True
    return mask


def _anchor_mask(
    atoms: Atoms,
    *,
    n_slab: int,
    fixed_mask: np.ndarray,
) -> np.ndarray:
    """Mask of atoms used to anchor periodic endpoint alignment (slab frame)."""
    n = len(atoms)
    if np.any(fixed_mask):
        return fixed_mask
    if n_slab > 0:
        anchor = np.zeros(n, dtype=bool)
        anchor[: min(n_slab, n)] = True
        return anchor
    return np.zeros(n, dtype=bool)


def _mobile_alignment_mask(
    anchor_mask: np.ndarray,
    *,
    n_slab: int,
    n_atoms: int,
) -> np.ndarray:
    """Atoms that may receive rigid alignment (not slab prefix, not anchored)."""
    mobile = np.ones(n_atoms, dtype=bool)
    if n_slab > 0:
        mobile[: min(n_slab, n_atoms)] = False
    mobile &= ~anchor_mask
    return mobile


def _cell_array(cell: Any) -> np.ndarray:
    """Return a 3x3 cell matrix from ASE ``Cell`` or ndarray."""
    if hasattr(cell, "array"):
        return np.asarray(cell.array, dtype=float)
    return np.asarray(cell, dtype=float)


def _pbc_for_mic_alignment(pbc: np.ndarray | list[bool]) -> np.ndarray:
    """PBC mask for MIC: in-plane periodic, vacuum axis open (slab convention)."""
    pbc_arr = np.asarray(pbc, dtype=bool).copy()
    normal_axis = _infer_surface_normal_axis(pbc_arr)
    pbc_arr[normal_axis] = False
    return pbc_arr


def _inplane_periodic_axes(pbc: np.ndarray | list[bool]) -> tuple[int, int]:
    """Return the two in-plane periodic axis indices for a slab-like cell."""
    pbc_arr = np.asarray(pbc, dtype=bool)
    periodic = [i for i in range(3) if pbc_arr[i]]
    if len(periodic) == 2:
        return int(periodic[0]), int(periodic[1])
    return 0, 1


def _validate_lattice_compatible_rotation(
    rot: np.ndarray,
    normal_axis: int,
    *,
    tol: float = 1e-6,
) -> None:
    """Fail-fast when a rotation would alter the vacuum axis or handedness."""
    if normal_axis not in (0, 1, 2):
        raise ValueError("normal_axis must be 0, 1, or 2")
    if abs(float(rot[normal_axis, normal_axis]) - 1.0) > tol:
        raise ValueError(
            "Rotation must preserve the surface normal axis (energy-equivalent)."
        )
    for i in range(3):
        if i != normal_axis and abs(float(rot[normal_axis, i])) > tol:
            raise ValueError(
                "Rotation must not mix the surface normal with in-plane axes."
            )
    if abs(float(np.linalg.det(rot)) - 1.0) > tol:
        raise ValueError("Rotation determinant must be +1 for rigid alignment.")


def _inplane_rotation_matrix_3d(angle: float, normal_axis: int) -> np.ndarray:
    """Build a 3x3 rotation about the surface normal (right-handed, det=+1)."""
    plane_axes = [i for i in range(3) if i != normal_axis]
    c, s = float(np.cos(angle)), float(np.sin(angle))
    rot2 = np.array([[c, -s], [s, c]], dtype=float)
    rot = np.eye(3, dtype=float)
    for i, ia in enumerate(plane_axes):
        for j, ja in enumerate(plane_axes):
            rot[ia, ja] = rot2[i, j]
    _validate_lattice_compatible_rotation(rot, normal_axis)
    return rot


def _lattice_translation_candidates(
    cell: np.ndarray,
    axis_a: int,
    axis_b: int,
    *,
    max_shift: int = 1,
) -> list[np.ndarray]:
    """Integer in-plane lattice translations (Cartesian vectors)."""
    if max_shift < 0:
        raise ValueError("max_shift must be non-negative")
    candidates: list[np.ndarray] = []
    for nx in range(-max_shift, max_shift + 1):
        for ny in range(-max_shift, max_shift + 1):
            delta = nx * cell[axis_a] + ny * cell[axis_b]
            candidates.append(np.asarray(delta, dtype=float))
    return candidates


def _mic_displacements(
    ref_pos: np.ndarray,
    prod_pos: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray | list[bool],
) -> np.ndarray:
    """Minimum-image displacements from reactant to product positions."""
    disp = prod_pos - ref_pos
    disp_mic, _ = find_mic(disp, cell=cell, pbc=pbc)
    return disp_mic


def _snap_to_reactant_mic_frame(
    ref_pos: np.ndarray,
    pos: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray | list[bool],
    anchor_mask: np.ndarray,
) -> np.ndarray:
    """Express ``pos`` in the reactant periodic image (Cartesian, MIC-short)."""
    disp_mic = _mic_displacements(ref_pos, pos, cell, pbc)
    if np.any(anchor_mask):
        disp_mic = disp_mic - np.mean(disp_mic[anchor_mask], axis=0)
    snapped = ref_pos + disp_mic
    if np.any(anchor_mask):
        snapped[anchor_mask] = ref_pos[anchor_mask]
    return snapped


def _score_mobile_endpoint_displacement(
    ref_pos: np.ndarray,
    prod_pos: np.ndarray,
    mobile_mask: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray | list[bool],
) -> tuple[float, float]:
    """Return (max, rms) mobile-atom displacement norms in the reactant MIC frame."""
    if not np.any(mobile_mask):
        return 0.0, 0.0
    disp_mic = _mic_displacements(ref_pos, prod_pos, cell, pbc)
    norms = np.linalg.norm(disp_mic[mobile_mask], axis=1)
    return float(np.max(norms)), float(np.sqrt(np.mean(norms**2)))


def _collective_mobile_lattice_snap(
    ref_pos: np.ndarray,
    prod_pos: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray | list[bool],
    mobile_mask: np.ndarray,
    *,
    axis_a: int,
    axis_b: int,
    max_shift: int,
) -> np.ndarray:
    """Pick a uniform in-plane lattice image for mobile atoms before per-atom MIC snap."""
    if not np.any(mobile_mask):
        return prod_pos

    best_pos = prod_pos.copy()
    best_score, _ = _score_mobile_endpoint_displacement(
        ref_pos, best_pos, mobile_mask, cell, pbc
    )
    for shift in _lattice_translation_candidates(
        cell, axis_a, axis_b, max_shift=max_shift
    ):
        shifted = prod_pos.copy()
        shifted[mobile_mask] += shift
        score, _ = _score_mobile_endpoint_displacement(
            ref_pos, shifted, mobile_mask, cell, pbc
        )
        if score < best_score:
            best_score = score
            best_pos = shifted
    return best_pos


def _apply_global_inplane_kabsch(
    ref_pos: np.ndarray,
    prod_pos: np.ndarray,
    mobile_mask: np.ndarray,
    *,
    normal_axis: int,
    anchor_mask: np.ndarray,
) -> np.ndarray:
    """Apply one global in-plane rotation derived from mobile-atom Kabsch."""
    idx = np.where(mobile_mask)[0]
    if idx.size < 2:
        return prod_pos
    center = ref_pos[idx].mean(axis=0)
    p_ref_c = ref_pos[idx] - center
    p_prod_c = prod_pos[idx] - center
    rot = _kabsch_rotation_in_plane(p_ref_c, p_prod_c, surface_normal_axis=normal_axis)
    _validate_lattice_compatible_rotation(rot, normal_axis)
    out = (prod_pos - center) @ rot.T + center
    if np.any(anchor_mask):
        out[anchor_mask] = ref_pos[anchor_mask]
    return out


def _align_product_surface_pbc(
    reactant: Atoms,
    product_positions: np.ndarray,
    *,
    n_slab: int = 0,
    enable_cell_remap: bool = True,
    enable_lattice_rotation: bool = True,
    max_lattice_shift: int = 1,
) -> np.ndarray:
    """Align product to reactant using MIC, lattice shifts, and global in-plane rotation.

    **Single surface NEB alignment entry point.** Serial (:func:`find_transition_state`),
    parallel (:func:`run_parallel_neb_search`), and :func:`interpolate_path` all route
    slab/periodic endpoint prep through this helper (not mobile-only Kabsch).

    Only energy-equivalent transforms are considered:
    - collective uniform in-plane lattice image for mobile atoms,
    - per-atom minimum-image wrapping,
    - integer in-plane lattice translations up to ``max_lattice_shift`` cells,
    - global in-plane rigid rotation (same ``R`` for all atoms; evaluated jointly
      with each shift candidate; anchors reset to reactant afterward).

    Does **not** rotate mobile atoms independently of the lattice frame.
    """
    ref_pos = reactant.get_positions()
    cell = _cell_array(reactant.cell)
    pbc_mic = _pbc_for_mic_alignment(reactant.pbc)
    normal_axis = _infer_surface_normal_axis(reactant.pbc)
    axis_a, axis_b = _inplane_periodic_axes(pbc_mic)

    fixed_mask = _fixed_atom_mask(reactant)
    anchor_mask = _anchor_mask(reactant, n_slab=n_slab, fixed_mask=fixed_mask)
    mobile_mask = _mobile_alignment_mask(
        anchor_mask, n_slab=n_slab, n_atoms=len(reactant)
    )

    prod = np.asarray(product_positions, dtype=float).copy()
    if enable_cell_remap:
        prod = _collective_mobile_lattice_snap(
            ref_pos,
            prod,
            cell,
            pbc_mic,
            mobile_mask,
            axis_a=axis_a,
            axis_b=axis_b,
            max_shift=max_lattice_shift,
        )

    prod = _snap_to_reactant_mic_frame(ref_pos, prod, cell, pbc_mic, anchor_mask)

    best_pos = prod.copy()
    best_score, _ = _score_mobile_endpoint_displacement(
        ref_pos, best_pos, mobile_mask, cell, pbc_mic
    )

    shifts = _lattice_translation_candidates(
        cell, axis_a, axis_b, max_shift=max_lattice_shift
    )
    if not enable_cell_remap:
        shifts = [np.zeros(3, dtype=float)]

    for shift in shifts:
        prod_shifted = prod + shift
        prod_snapped = _snap_to_reactant_mic_frame(
            ref_pos, prod_shifted, cell, pbc_mic, anchor_mask
        )
        candidates: list[tuple[float, np.ndarray]] = []
        score, _ = _score_mobile_endpoint_displacement(
            ref_pos, prod_snapped, mobile_mask, cell, pbc_mic
        )
        candidates.append((score, prod_snapped))

        if enable_lattice_rotation:
            prod_rot = _apply_global_inplane_kabsch(
                ref_pos,
                prod_snapped,
                mobile_mask,
                normal_axis=normal_axis,
                anchor_mask=anchor_mask,
            )
            prod_rot_snapped = _snap_to_reactant_mic_frame(
                ref_pos, prod_rot, cell, pbc_mic, anchor_mask
            )
            score_rot, _ = _score_mobile_endpoint_displacement(
                ref_pos, prod_rot_snapped, mobile_mask, cell, pbc_mic
            )
            candidates.append((score_rot, prod_rot_snapped))

        for score_c, pos_c in candidates:
            if score_c < best_score:
                best_score = score_c
                best_pos = pos_c

    return _snap_to_reactant_mic_frame(ref_pos, best_pos, cell, pbc_mic, anchor_mask)


def _requires_surface_pbc_alignment(reactant: Atoms, *, n_slab: int) -> bool:
    """True when endpoint alignment must use lattice-compatible surface PBC logic."""
    return n_slab > 0 or bool(np.any(reactant.pbc))


def _align_product_for_neb(
    reactant: Atoms,
    product_positions: np.ndarray,
    *,
    n_slab: int = 0,
    surface_cell_remap: bool = True,
    surface_lattice_rotation: bool = True,
    surface_max_lattice_shift: int = 1,
) -> np.ndarray:
    """Single NEB endpoint rigid-alignment entry point (gas Kabsch or surface PBC)."""
    if _requires_surface_pbc_alignment(reactant, n_slab=n_slab):
        return _align_product_surface_pbc(
            reactant,
            product_positions,
            n_slab=n_slab,
            enable_cell_remap=surface_cell_remap,
            enable_lattice_rotation=surface_lattice_rotation,
            max_lattice_shift=surface_max_lattice_shift,
        )
    return _align_product_kabsch_to_reactant(
        reactant,
        product_positions,
        n_slab=n_slab,
        in_plane_only=False,
    )


def _align_product_kabsch_to_reactant(
    reactant: Atoms,
    product_positions: np.ndarray,
    *,
    n_slab: int = 0,
    in_plane_only: bool = False,
) -> np.ndarray:
    """Rigidly align product to reactant (gas-phase clusters without periodic endpoints)."""
    if n_slab > 0:
        raise RuntimeError(
            "Slab NEB endpoints must use _align_product_surface_pbc, not Kabsch-only alignment."
        )
    ref_pos = reactant.get_positions()
    fixed_mask = _fixed_atom_mask(reactant)
    anchor_mask = _anchor_mask(reactant, n_slab=n_slab, fixed_mask=fixed_mask)
    mobile_mask = _mobile_alignment_mask(
        anchor_mask, n_slab=n_slab, n_atoms=len(reactant)
    )

    if np.any(mobile_mask) and mobile_mask.size < len(reactant):
        out = product_positions.copy()
        p_ref = ref_pos[mobile_mask]
        p_prod = product_positions[mobile_mask]
        center = p_ref.mean(axis=0)
        p_ref_c = p_ref - center
        p_prod_c = p_prod - center
        if in_plane_only:
            rot = _kabsch_rotation_in_plane(
                p_ref_c,
                p_prod_c,
                surface_normal_axis=_infer_surface_normal_axis(reactant.pbc),
            )
        else:
            rot = _kabsch_rotation(p_ref_c, p_prod_c)
        out[mobile_mask] = (p_prod_c @ rot.T) + center
        if np.any(anchor_mask):
            out[anchor_mask] = ref_pos[anchor_mask]
        return out

    p_ref = ref_pos
    p_prod = product_positions
    center = p_ref.mean(axis=0)
    p_ref_c = p_ref - center
    p_prod_c = p_prod - center
    if in_plane_only:
        rot = _kabsch_rotation_in_plane(
            p_ref_c,
            p_prod_c,
            surface_normal_axis=_infer_surface_normal_axis(reactant.pbc),
        )
    else:
        rot = _kabsch_rotation(p_ref_c, p_prod_c)
    return (p_prod_c @ rot.T) + center


def _reorder_product_to_match_reactant(
    reactant: Atoms,
    product: Atoms,
    *,
    n_slab: int,
    n_core_mobile: int | None,
    n_adsorbate_mobile: int | None,
) -> np.ndarray:
    """Reorder product atoms (positions and species) to match reactant ordering."""
    n_atom = len(reactant)
    mic_cell, mic_pbc = _mic_matching_context(reactant, n_slab=n_slab)
    use_blocks = (
        n_core_mobile is not None
        and n_adsorbate_mobile is not None
        and n_slab + int(n_core_mobile) + int(n_adsorbate_mobile) == n_atom
    )
    if use_blocks:
        _align_endpoints_blockwise(
            reactant,
            product,
            n_slab,
            int(n_core_mobile),
            int(n_adsorbate_mobile),
            mic_cell=mic_cell,
            mic_pbc=mic_pbc,
        )
        return product.get_positions()
    if 0 < n_slab < n_atom:
        p_m, n_m = _permute_atoms_block_to_match(
            reactant[n_slab:],
            product[n_slab:],
            mic_cell=mic_cell,
            mic_pbc=mic_pbc,
        )
        pos = product.get_positions().copy()
        nums = product.numbers.copy()
        pos[n_slab:] = p_m
        nums[n_slab:] = n_m
        product.set_positions(pos)
        product.numbers = nums
        return pos
    mapping = _match_atoms_by_fingerprint(
        reactant, product, mic_cell=mic_cell, mic_pbc=mic_pbc
    )
    product.set_positions(product.get_positions()[mapping])
    product.numbers = product.numbers[mapping]
    return product.get_positions()


def interpolate_path(
    atoms1: Atoms,
    atoms2: Atoms,
    n_images: int = 5,
    method: str = "idpp",
    mic: bool = False,
    *,
    align_endpoints: bool = True,
    perturb_sigma: float = 0.0,
    rng: np.random.Generator | None = None,
    system_type: SystemType | None = None,
    n_slab: int = 0,
    n_core_mobile: int | None = None,
    n_adsorbate_mobile: int | None = None,
    neb_surface_cell_remap: bool = True,
    neb_surface_lattice_rotation: bool = True,
    neb_surface_max_lattice_shift: int = 1,
) -> list[Atoms]:
    """Interpolate between two structures and return images including endpoints.

    ``align_endpoints`` (default True): reorder endpoint atoms to match reactant.
    For slab/surface workflows (``n_slab > 0`` or periodic cell), alignment uses
    :func:`_align_product_surface_pbc`: MIC-aware matching, collective mobile
    lattice-image selection, per-atom MIC snapping, optional integer in-plane
    lattice shifts (``neb_surface_max_lattice_shift``), and global in-plane rotation
    evaluated jointly with each shift, with anchors reset to the reactant slab frame
    (no independent mobile-only rotation). Gas-phase clusters
    without a slab prefix use Kabsch (3D or in-plane when ``mic`` is active).
    ``perturb_sigma``: optional Gaussian displacement (Å) on interior images only.
    ``rng``: optional NumPy Generator when ``perturb_sigma`` > 0.

    If ``n_slab`` + ``n_core_mobile`` + ``n_adsorbate_mobile`` equals
    ``len(atoms)``, match endpoints per slab / core / adsorbate block instead
    of one global permutation.

    For constrained slab systems we always interpolate with
    ``apply_constraint=False``; constraints remain attached and are enforced
    during subsequent NEB optimization.
    """
    validate_atoms(atoms1)
    validate_atoms(atoms2)

    a1_copy = atoms1.copy()
    a2_copy = atoms2.copy()

    surface_cell_remap = neb_surface_cell_remap
    surface_lattice_rotation = neb_surface_lattice_rotation
    if align_endpoints and system_type is not None:
        system_policy = get_system_policy(system_type)
        if system_policy.neb_disable_alignment:
            raise ValueError(
                f"Endpoint alignment is not allowed for {system_type!r}; set align_endpoints=False."
            )
        surface_cell_remap = (
            system_policy.neb_surface_cell_remap and neb_surface_cell_remap
        )
        surface_lattice_rotation = (
            system_policy.neb_surface_lattice_rotation and neb_surface_lattice_rotation
        )

    if align_endpoints:
        new_pos = _reorder_product_to_match_reactant(
            a1_copy,
            a2_copy,
            n_slab=n_slab,
            n_core_mobile=n_core_mobile,
            n_adsorbate_mobile=n_adsorbate_mobile,
        )
        # Keep species order consistent with reactant for downstream NEB.
        a2_copy.numbers = a1_copy.numbers.copy()
        aligned = _align_product_for_neb(
            a1_copy,
            new_pos,
            n_slab=n_slab,
            surface_cell_remap=surface_cell_remap,
            surface_lattice_rotation=surface_lattice_rotation,
            surface_max_lattice_shift=neb_surface_max_lattice_shift,
        )
        a2_copy.set_positions(aligned)
        if _requires_surface_pbc_alignment(a1_copy, n_slab=n_slab):
            a2_copy.set_cell(a1_copy.cell)
            a2_copy.pbc = a1_copy.pbc

    # Build the band from aligned endpoints; ASE interpolation only fills interiors.
    images = [a1_copy] + [a1_copy.copy() for _ in range(n_images)] + [a2_copy]
    neb = NEB(images, method=DEFAULT_NEB_TANGENT_METHOD)
    # Interpolate unconstrained positions first; endpoint/image constraints
    # (e.g., fixed slab atoms) are enforced during subsequent optimization.
    neb.interpolate(method=method, mic=mic, apply_constraint=False)
    images = neb.images

    if perturb_sigma and perturb_sigma > 0.0:
        if rng is None:
            rng = np.random.default_rng()
        for img in images[1:-1]:
            disp = rng.normal(
                scale=float(perturb_sigma), size=img.get_positions().shape
            )
            img.set_positions(img.get_positions() + disp)

    return images


def _coerce_neb_steps(neb_steps: int | str | None) -> int | str | None:
    """Coerce numpy integer step counts to plain int (JSON-friendly)."""
    if isinstance(neb_steps, (int, np.integer)):
        return int(neb_steps)
    return neb_steps


def make_ts_result(
    *,
    pair_id: str,
    n_images: int,
    spring_constant: float,
    use_torchsim: bool,
    fmax: float,
    neb_steps: int | str | None,
    interpolation_method: str,
    climb: bool,
    align_endpoints: bool,
    perturb_sigma: float,
    neb_interpolation_mic: bool,
    neb_tangent_method: str,
    use_parallel_neb: bool = False,
    reactant_energy: float | None = None,
    product_energy: float | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Build a normalized TS-result dict (failure shape, success-promoted later)."""
    return {
        "status": "failed",
        "pair_id": pair_id,
        "neb_converged": False,
        "n_images": n_images,
        "spring_constant": spring_constant,
        "reactant_energy": float(reactant_energy)
        if reactant_energy is not None
        else None,
        "product_energy": float(product_energy) if product_energy is not None else None,
        "ts_energy": None,
        "ts_image_index": None,
        "barrier_height": None,
        "barrier_forward": None,
        "barrier_reverse": None,
        "transition_state": None,
        "error": error,
        "use_torchsim": bool(use_torchsim),
        "use_parallel_neb": bool(use_parallel_neb),
        "fmax": float(fmax),
        "neb_steps": _coerce_neb_steps(neb_steps),
        "interpolation_method": interpolation_method,
        "climb": bool(climb),
        "align_endpoints": bool(align_endpoints),
        "perturb_sigma": float(perturb_sigma),
        "neb_interpolation_mic": bool(neb_interpolation_mic),
        "neb_tangent_method": neb_tangent_method,
        "final_fmax": None,
        "steps_taken": None,
    }


def minima_provenance_dict(minima: list, idx: int) -> dict[str, Any]:
    """Extract per-minimum GO provenance for JSON serialization."""
    if not minima or idx < 0 or idx >= len(minima):
        get_logger(__name__).warning(
            "minima_provenance_dict: invalid index %s for %d minima",
            idx,
            len(minima) if minima else 0,
        )
        return {}

    energy, atoms = minima[idx]
    return {
        "run_id": get_metadata(atoms, "run_id"),
        "source_db": get_metadata(atoms, "source_db"),
        "source_db_relpath": get_metadata(atoms, "source_db_relpath"),
        "systems_row_id": get_metadata(atoms, "systems_row_id"),
        "confid": get_metadata(atoms, "confid"),
        "gaid": get_metadata(atoms, "gaid"),
        "unique_id": get_metadata(atoms, "unique_id"),
        "final_id": get_metadata(atoms, "final_id"),
        "energy": float(energy) if energy is not None else None,
    }


def attach_minima_traceability(
    result: dict[str, Any],
    minima: list[tuple[float, Any]],
    i: int,
    j: int,
) -> None:
    """Record minima list indices and endpoint provenance on one TS result."""
    result["minima_indices"] = [int(i), int(j)]
    result["minima_provenance"] = [
        minima_provenance_dict(minima, i),
        minima_provenance_dict(minima, j),
    ]


def _finalize_neb_result(
    result: dict[str, Any],
    images: list[Atoms],
    *,
    logger: Any | None = None,
) -> None:
    """Populate ``result`` with TS / endpoint geometry, energies, and barriers.

    Mutates ``result`` in place. Assumes ``reactant_energy`` and
    ``product_energy`` are already set; raises ``RuntimeError`` otherwise.
    Marks an endpoint-as-TS result as failed.
    """
    pair_id = result.get("pair_id")

    react = images[0].copy()
    prod = images[-1].copy()
    _detach_calc(react)
    _detach_calc(prod)
    result["reactant_structure"] = react
    result["product_structure"] = prod

    max_energy_idx = 0
    max_energy = -np.inf
    ts_atoms: Atoms | None = None
    for idx, atoms in enumerate(images):
        energy = float(atoms.get_potential_energy())
        if energy > max_energy:
            max_energy = energy
            max_energy_idx = idx
            ts_atoms = atoms

    if result.get("reactant_energy") is None or result.get("product_energy") is None:
        raise RuntimeError(
            f"Missing endpoint energies after NEB for pair {pair_id}: "
            f"reactant={result.get('reactant_energy')}, product={result.get('product_energy')}"
        )
    if ts_atoms is None:
        raise RuntimeError(f"No TS energy found after NEB for pair {pair_id}")

    reactant_energy = float(result["reactant_energy"])
    product_energy = float(result["product_energy"])
    ts_energy = float(max_energy)
    barrier_height = ts_energy - min(reactant_energy, product_energy)

    ts_copy = deepcopy(ts_atoms)
    _detach_calc(ts_copy)
    result["transition_state"] = ts_copy
    result["ts_energy"] = ts_energy
    result["ts_image_index"] = int(max_energy_idx)
    result["barrier_height"] = barrier_height
    result["barrier_forward"] = ts_energy - reactant_energy
    result["barrier_reverse"] = ts_energy - product_energy

    endpoint_ts = max_energy_idx == 0 or max_energy_idx == len(images) - 1
    if endpoint_ts:
        result["status"] = "failed"
        result["neb_converged"] = False
        result["error"] = (
            f"NEB returned endpoint as TS (image {max_energy_idx}); "
            "no interior saddle located"
        )
        if logger is not None:
            logger.warning(
                "NEB reported endpoint as TS for pair %s (image %d) — marking as non-converged",
                pair_id,
                max_energy_idx,
            )
    else:
        result["status"] = "success" if result.get("neb_converged") else "failed"


def find_transition_state(
    atoms1: Atoms,
    atoms2: Atoms,
    calculator: Calculator | None,
    output_dir: str,
    pair_id: str,
    rng: np.random.Generator | None = None,
    n_images: int = 3,
    spring_constant: float = 0.1,
    optimizer: type[Optimizer] = FIRE,
    fmax: float = 0.05,
    neb_steps: int = 500,
    trajectory: str | None = None,
    verbosity: int = 1,
    use_torchsim: bool = False,
    torchsim_params: dict[str, Any] | None = None,
    climb: bool = False,
    interpolation_method: str = "idpp",
    align_endpoints: bool = True,
    perturb_sigma: float = 0.0,
    neb_interpolation_mic: bool = False,
    neb_tangent_method: str = DEFAULT_NEB_TANGENT_METHOD,
    system_type: SystemType | None = None,
    write_timing_json: bool = False,
    n_slab: int = 0,
    n_core_mobile: int | None = None,
    n_adsorbate_mobile: int | None = None,
    neb_surface_cell_remap: bool = True,
    neb_surface_lattice_rotation: bool = True,
    neb_surface_max_lattice_shift: int = 1,
) -> dict[str, Any]:
    """Run NEB to locate a transition state between two structures.

    Args:
        neb_interpolation_mic: Forwarded to :func:`interpolate_path` as ``mic``.
            Use ``True`` for periodic cells (e.g. slabs); default ``False`` for
            isolated clusters.
        neb_tangent_method: ASE NEB tangent method (``ase.mep.neb.NEB`` ``method``
            argument). Default ``improvedtangent`` matches ASE recommendations.
        n_slab: Blockwise alignment: slab length (default 0).
        n_core_mobile: Mobile core count (with ``n_adsorbate_mobile`` for blockwise NEB).
        n_adsorbate_mobile: Mobile adsorbate fragment count.
        neb_surface_cell_remap: Enable in-plane lattice-image search (surface).
        neb_surface_lattice_rotation: Enable global in-plane rotation (surface).
        neb_surface_max_lattice_shift: Max integer cell index searched in-plane
            during remap (default ``1``).

    Returns:
        A summary dict with TS geometry, energies and convergence status.
    """
    logger = get_logger(__name__)

    validate_atoms(atoms1)
    validate_atoms(atoms2)

    if use_torchsim:
        if is_uma_like_calculator(calculator):
            _require_torchsim_fairchem()
        else:
            _require_torchsim()
    else:
        validate_calculator_attached(atoms1, "NEB reactant")
        validate_calculator_attached(atoms2, "NEB product")

    if len(atoms1) != len(atoms2):
        raise ValueError(
            f"Atoms objects have different lengths: {len(atoms1)} vs {len(atoms2)}"
        )

    if trajectory is None:
        trajectory = os.path.join(output_dir, f"neb_{pair_id}.traj")

    # Extract initial energies (safe for TorchSim where atoms have no calculator).
    reactant_energy = extract_energy_from_atoms(atoms1)
    product_energy = extract_energy_from_atoms(atoms2)

    # For ASE NEB we require explicit endpoint energies; for TorchSim the
    # relaxer computes them below.
    if not use_torchsim:
        if reactant_energy is None:
            raise ValueError(
                f"Cannot extract energy from reactant atoms for pair {pair_id}"
            )
        if product_energy is None:
            raise ValueError(
                f"Cannot extract energy from product atoms for pair {pair_id}"
            )

    if verbosity >= 1:
        logger.info(f"Finding transition state for pair {pair_id}")
        if reactant_energy is not None:
            logger.info(f"  Reactant energy: {reactant_energy:.6f} eV")
        if product_energy is not None:
            logger.info(f"  Product energy: {product_energy:.6f} eV")

    result = make_ts_result(
        pair_id=pair_id,
        n_images=n_images,
        spring_constant=spring_constant,
        use_torchsim=use_torchsim,
        fmax=fmax,
        neb_steps=neb_steps,
        interpolation_method=interpolation_method,
        climb=climb,
        align_endpoints=align_endpoints,
        perturb_sigma=perturb_sigma,
        neb_interpolation_mic=neb_interpolation_mic,
        neb_tangent_method=neb_tangent_method,
        reactant_energy=reactant_energy,
        product_energy=product_energy,
    )

    t_wall0: float | None = None
    neb_opt = 0.0
    try:
        t_wall0 = perf_counter()
        if np.allclose(atoms1.get_positions(), atoms2.get_positions(), atol=1e-8):
            raise ValueError(
                f"Endpoints are identical for pair {pair_id}; no interior TS"
            )

        if verbosity >= 2:
            logger.info(
                f"Generating initial path with {interpolation_method} interpolation"
            )
        # Keep interpolation unconstrained; constraints are applied during NEB.
        images = interpolate_path(
            atoms1,
            atoms2,
            n_images=n_images,
            method=interpolation_method,
            mic=neb_interpolation_mic,
            align_endpoints=align_endpoints,
            perturb_sigma=perturb_sigma,
            rng=rng,
            system_type=system_type,
            n_slab=n_slab,
            n_core_mobile=n_core_mobile,
            n_adsorbate_mobile=n_adsorbate_mobile,
            neb_surface_cell_remap=neb_surface_cell_remap,
            neb_surface_lattice_rotation=neb_surface_lattice_rotation,
            neb_surface_max_lattice_shift=neb_surface_max_lattice_shift,
        )

        if np.allclose(
            images[0].get_positions(), images[-1].get_positions(), atol=1e-8
        ):
            raise ValueError(
                f"Endpoints are identical for pair {pair_id}; no interior TS"
            )

        neb: NEB
        if use_torchsim:
            relaxer = _tsh.TorchSimBatchRelaxer(**(torchsim_params or {}))

            ep_results = relaxer.relax_batch([images[0], images[-1]], steps=0)
            result["reactant_energy"] = float(ep_results[0][0])
            result["product_energy"] = float(ep_results[1][0])

            if verbosity >= 2:
                logger.info(f"Using TorchSim batched NEB (climb={climb})")

            neb = TorchSimNEB(
                images,
                relaxer,
                k=spring_constant,
                climb=climb,
                method=neb_tangent_method,
            )
        else:
            if calculator is None:
                raise ValueError("Calculator required when use_torchsim=False")
            for img in images:
                try:
                    img.calc = deepcopy(calculator)
                except (TypeError, AttributeError):
                    img.calc = calculator

            neb = NEB(
                images,
                k=spring_constant,
                climb=climb,
                method=neb_tangent_method,
            )

        opt_logfile = None if verbosity <= 1 else sys.stdout
        dyn: Optimizer = optimizer(neb, trajectory=trajectory, logfile=opt_logfile)  # type: ignore[arg-type]

        if verbosity >= 2:
            logger.info(f"Starting NEB optimization with {optimizer.__name__}")

        t_neb0 = perf_counter()
        dyn.run(fmax=fmax, steps=neb_steps)
        neb_opt = perf_counter() - t_neb0

        try:
            neb_forces = neb.get_forces()
            final_fmax: float | None = float(np.max(np.abs(neb_forces)))
        except (AttributeError, RuntimeError, ValueError):
            final_fmax = None

        result["final_fmax"] = final_fmax
        result["neb_converged"] = final_fmax is not None and final_fmax < fmax
        result["steps_taken"] = int(dyn.nsteps)

        if not result["neb_converged"] and result.get("error") is None:
            result["error"] = (
                f"NEB did not converge (final_fmax={final_fmax}, fmax={fmax})"
            )

        if verbosity >= 1:
            fmax_str = f"{final_fmax:.6f}" if final_fmax is not None else "unknown"
            if result["neb_converged"]:
                logger.info(
                    "NEB converged in %d steps (final_fmax=%s < %.6f)",
                    result["steps_taken"],
                    fmax_str,
                    fmax,
                )
            else:
                logger.warning(
                    "NEB not converged after %d steps (final_fmax=%s, target_fmax=%.6f)",
                    result["steps_taken"],
                    fmax_str,
                    fmax,
                )

        _finalize_neb_result(result, neb.images, logger=logger)

        if use_torchsim and result["status"] == "success":
            result["force_calls"] = neb.get_force_calls()

        if verbosity >= 1 and result["status"] == "success":
            logger.info(
                f"TS found at image {result['ts_image_index']}/{len(neb.images) - 1}"
            )
            logger.info(f"  TS energy: {result['ts_energy']:.6f} eV")
            logger.info(f"  Barrier height: {result['barrier_height']:.6f} eV")
            if use_torchsim:
                logger.info(
                    "  GPU-batched force calls: %s",
                    result.get("force_calls"),
                )

    except KeyboardInterrupt:
        raise
    except (ValueError, RuntimeError, OSError) as e:
        result["error"] = str(e)
        if is_cuda_oom_error(e):
            cleanup_torch_cuda(logger=logger)
            if verbosity >= 1:
                logger.warning(
                    "Detected CUDA out-of-memory during NEB for pair %s — attempted GPU cleanup",
                    pair_id,
                )
        if verbosity >= 1:
            logger.error(
                f"Failed to find TS for pair {pair_id}: {type(e).__name__}: {e}"
            )

    if t_wall0 is not None:
        total_s = perf_counter() - t_wall0
        ts_timings: dict[str, float] = {
            "total_wall_s": total_s,
            "neb_optimization_s": neb_opt,
            "cpu_non_relax_s": max(0.0, total_s - neb_opt),
        }
        result["timings_s"] = ts_timings
        neb_backend = "neb_torchsim" if use_torchsim else "neb_ase"
        log_timing_summary(logger, neb_backend, ts_timings, verbosity=verbosity)
        if write_timing_json:
            write_timing_file(
                output_dir,
                build_timing_payload(
                    backend=neb_backend,
                    timings_s=ts_timings,
                    extra={"pair_id": pair_id},
                ),
                filename=f"timing_{pair_id}.json",
            )

    return result


_PROVENANCE_KEYS = (
    "system_type",
    "use_torchsim",
    "use_parallel_neb",
    "climb",
    "align_endpoints",
    "perturb_sigma",
    "neb_interpolation_mic",
    "interpolation_method",
    "fmax",
    "neb_steps",
    "minima_indices",
    "minima_provenance",
)


def save_neb_result(
    result: dict[str, Any],
    output_dir: str,
    pair_id: str,
) -> None:
    """Save NEB result: TS and endpoint XYZ (when present) plus metadata JSON.

    Writes:
    - ``ts_{pair_id}.xyz`` on success when a TS geometry is present
    - ``reactant_{pair_id}.xyz`` / ``product_{pair_id}.xyz`` when
      ``reactant_structure`` / ``product_structure`` are on the result dict
    - ``neb_{pair_id}_metadata.json`` (includes schema/version/time and NEB params)
    """
    logger = get_logger(__name__)

    os.makedirs(output_dir, exist_ok=True)

    if result["status"] == "success" and result["transition_state"] is not None:
        _detach_calc(result["transition_state"])
        ts_path = os.path.join(output_dir, f"ts_{pair_id}.xyz")
        write(ts_path, result["transition_state"])
        logger.info(f"Saved TS structure to {ts_path}")

    for label, key in (
        ("reactant", "reactant_structure"),
        ("product", "product_structure"),
    ):
        atoms = result.get(key)
        if atoms is not None:
            ep = atoms.copy()
            _detach_calc(ep)
            ep_path = os.path.join(output_dir, f"{label}_{pair_id}.xyz")
            write(ep_path, ep)
            logger.info(f"Saved {label} endpoint structure to {ep_path}")

    extra = {key: result[key] for key in _PROVENANCE_KEYS if key in result}
    extra["neb_backend"] = (
        "torchsim" if result.get("use_torchsim") else result.get("neb_backend", "ase")
    )
    metadata = ts_output_provenance(extra=extra)
    metadata.update(
        {
            "pair_id": result["pair_id"],
            "status": result["status"],
            "neb_converged": result["neb_converged"],
            "n_images": result["n_images"],
            "spring_constant": result["spring_constant"],
            "reactant_energy": result["reactant_energy"],
            "product_energy": result["product_energy"],
            "ts_energy": result["ts_energy"],
            "barrier_height": result["barrier_height"],
            "error": result["error"],
            "final_fmax": result.get("final_fmax"),
            "steps_taken": result.get("steps_taken"),
            "force_calls": result.get("force_calls"),
        }
    )

    if result["status"] == "success":
        metadata["ts_image_index"] = result.get("ts_image_index")

    metadata_path = os.path.join(output_dir, f"neb_{pair_id}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved NEB metadata to {metadata_path}")
