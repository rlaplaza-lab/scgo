"""Surface-NEB helpers: moiety unwrap, symmetry copies, variable springs, fidelity."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111
from ase.calculators.emt import EMT
from ase.mep import NEB

from scgo.exceptions import SCGOValidationError
from scgo.metadata.atoms import set_tags
from scgo.ts_search import neb_surface as ns
from scgo.ts_search.neb_surface import (
    SYMMETRY_DRIFT_EV,
    apply_inplane_symmetry,
    bond_set,
    classify_band,
    consistent_product_positions,
    count_imaginary_frequencies,
    count_imaginary_modes_from_forces,
    displaced_mobile_images,
    inplane_symmetry_matrices,
    symmetry_anchor_center,
    symmetry_product_rejected,
    variable_spring_constants,
)
from scgo.ts_search.parallel_neb import ParallelNEBBatch
from scgo.ts_search.transition_state import (
    TorchSimNEB,
    _align_product_surface_pbc,
    _finalize_neb_result,
    find_transition_state,
    interpolate_path,
)


def _oh_on_slab_split_by_mic():
    """O–H dimer where product H is shifted by 0.4 cell (MIC path stretches the bond)."""
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=8.0, orthogonal=True)
    slab.pbc = [True, True, False]
    z0 = float(slab.get_positions()[:, 2].max()) + 1.5
    n_slab = len(slab)
    ox = 0.5
    cellx = float(slab.cell[0, 0])
    a = slab.copy() + Atoms(
        "OH",
        positions=[[ox, 0.5, z0], [ox + 1.0, 0.5, z0]],
    )
    b = slab.copy() + Atoms(
        "OH",
        positions=[[ox, 0.5, z0], [ox + 1.0 + 0.4 * cellx, 0.5, z0]],
    )
    return a, b, n_slab


# --- Moieties -----------------------------------------------------------------


def test_consistent_product_positions_unwraps_split_dimer():
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=8.0, orthogonal=True)
    slab.pbc = [True, True, False]
    z0 = float(slab.get_positions()[:, 2].max()) + 1.5
    n_slab = len(slab)
    ox = 0.5
    a = slab.copy() + Atoms(
        "OH",
        positions=[[ox, 0.5, z0], [ox + 1.0, 0.5, z0]],
    )
    b = slab.copy() + Atoms(
        "OH",
        positions=[[ox, 0.5, z0], [ox + 1.0 + slab.cell[0, 0], 0.5, z0]],
    )
    unwrapped = consistent_product_positions(a, b.get_positions(), n_slab=n_slab)
    oh = float(np.linalg.norm(unwrapped[-1] - unwrapped[-2]))
    assert abs(oh - 1.0) < 0.05
    mid = 0.5 * (a.get_positions() + unwrapped)
    mid_oh = float(np.linalg.norm(mid[-1] - mid[-2]))
    assert abs(mid_oh - 1.0) < 0.05


def test_per_atom_mic_interpolate_splits_dimer_bond():
    a, b, _n_slab = _oh_on_slab_split_by_mic()
    images = [a.copy(), a.copy(), b.copy()]
    neb = NEB(images)
    neb.interpolate(method="linear", mic=True, apply_constraint=False)
    mid = neb.images[1]
    mid_oh = float(mid.get_distance(-2, -1, mic=False))
    assert mid_oh > 2.0


def test_interpolate_path_keeps_intact_oh_after_idpp():
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=8.0, orthogonal=True)
    slab.pbc = [True, True, False]
    z0 = float(slab.get_positions()[:, 2].max()) + 1.5
    n_slab = len(slab)
    ox = 0.5
    a = slab.copy() + Atoms(
        "OH",
        positions=[[ox, 0.5, z0], [ox + 1.0, 0.5, z0]],
    )
    b = slab.copy() + Atoms(
        "OH",
        positions=[[ox, 0.5, z0], [ox + 1.0 + slab.cell[0, 0], 0.5, z0]],
    )
    images = interpolate_path(
        a,
        b,
        n_images=3,
        method="linear",
        mic=True,
        align_endpoints=True,
        n_slab=n_slab,
        system_type="surface_cluster_adsorbate",
    )
    for img in images:
        d = float(img.get_distance(-2, -1, mic=True))
        assert abs(d - 1.0) < 0.15
        assert np.isfinite(d)


def test_consistent_product_positions_noop_for_gas_h2():
    h2 = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]], pbc=False)
    prod = h2.get_positions().copy()
    prod[1, 0] += 0.1
    out = consistent_product_positions(h2, prod, n_slab=0)
    np.testing.assert_allclose(out, prod)


# --- Symmetry -----------------------------------------------------------------


def test_hexagonal_symmetry_recovers_120_deg_dimer():
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=8.0, orthogonal=False)
    slab.pbc = [True, True, False]
    n_slab = len(slab)
    mats = inplane_symmetry_matrices(slab, n_slab=n_slab)
    assert len(mats) > 1

    z0 = float(slab.get_positions()[:, 2].max()) + 1.5
    a = slab.copy() + Atoms(
        "PtPt",
        positions=[[1.0, 1.0, z0], [2.5, 1.0, z0]],
    )
    center = symmetry_anchor_center(a, n_slab=n_slab)
    assert center is not None
    rot120 = mats[2] if len(mats) > 2 else mats[1]
    mobile_mask = np.zeros(len(a), dtype=bool)
    mobile_mask[n_slab:] = True
    anchor_mask = np.zeros(len(a), dtype=bool)
    anchor_mask[:n_slab] = True
    prod_pos = apply_inplane_symmetry(
        a.get_positions(),
        rot120,
        center=center,
        mobile_mask=mobile_mask,
        anchor_mask=anchor_mask,
        ref_pos=a.get_positions(),
    )
    aligned, used = _align_product_surface_pbc(
        a,
        prod_pos,
        n_slab=n_slab,
        enable_lattice_rotation=False,
        allow_symmetry_copies=True,
    )
    disp = np.linalg.norm(aligned[n_slab:] - a.get_positions()[n_slab:], axis=1)
    assert float(np.max(disp)) < 0.05
    assert used is True

    aligned_off, used_off = _align_product_surface_pbc(
        a,
        prod_pos,
        n_slab=n_slab,
        enable_lattice_rotation=False,
        allow_symmetry_copies=False,
    )
    disp_off = np.linalg.norm(aligned_off[n_slab:] - a.get_positions()[n_slab:], axis=1)
    assert float(np.max(disp_off)) > 0.5
    assert used_off is False


def test_square_monolayer_90_deg_symmetry():
    cell = 2.8
    slab = Atoms(
        "Pt4",
        positions=[
            [0.0, 0.0, 0.0],
            [cell, 0.0, 0.0],
            [0.0, cell, 0.0],
            [cell, cell, 0.0],
        ],
        cell=[cell, cell, 12.0],
        pbc=[True, True, False],
    )
    n_slab = len(slab)
    mats = inplane_symmetry_matrices(slab, n_slab=n_slab)
    assert len(mats) > 1

    z0 = 2.0
    a = slab.copy() + Atoms(
        "PtPt",
        positions=[[0.7, 0.7, z0], [1.5, 0.7, z0]],
    )
    center = symmetry_anchor_center(a, n_slab=n_slab)
    assert center is not None
    rot90 = None
    for m in mats[1:]:
        if abs(float(m[0, 1]) + 1.0) < 0.1 and abs(float(m[1, 0]) - 1.0) < 0.1:
            rot90 = m
            break
    assert rot90 is not None
    mobile_mask = np.zeros(len(a), dtype=bool)
    mobile_mask[n_slab:] = True
    anchor_mask = np.zeros(len(a), dtype=bool)
    anchor_mask[:n_slab] = True
    prod_pos = apply_inplane_symmetry(
        a.get_positions(),
        rot90,
        center=center,
        mobile_mask=mobile_mask,
        anchor_mask=anchor_mask,
        ref_pos=a.get_positions(),
    )
    aligned, used = _align_product_surface_pbc(
        a,
        prod_pos,
        n_slab=n_slab,
        enable_lattice_rotation=False,
        allow_symmetry_copies=True,
    )
    disp = np.linalg.norm(aligned[n_slab:] - a.get_positions()[n_slab:], axis=1)
    assert float(np.max(disp)) < 0.05
    assert used is True


def test_broken_slab_symmetry_falls_back_to_identity():
    # Asymmetric in-plane lattice: orthogonal angle but no C2 map.
    slab = Atoms(
        "Pt4",
        positions=[[0, 0, 0], [3.0, 0, 0], [0.4, 4.5, 0], [2.2, 3.8, 0]],
        cell=[3.0, 4.5, 12.0],
        pbc=[True, True, False],
    )
    n_slab = len(slab)
    mats = inplane_symmetry_matrices(slab, n_slab=n_slab)
    assert len(mats) == 1
    z0 = 2.0
    a = slab.copy() + Atoms("Pt", positions=[[0.5, 0.5, z0]])
    b_pos = a.get_positions().copy()
    b_pos[-1, 0] += 0.3
    on, _ = _align_product_surface_pbc(
        a, b_pos, n_slab=n_slab, allow_symmetry_copies=True
    )
    off, _ = _align_product_surface_pbc(
        a, b_pos, n_slab=n_slab, allow_symmetry_copies=False
    )
    np.testing.assert_allclose(on, off, atol=1e-8)


def test_orthogonal_fcc111_proposes_c2():
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=8.0, orthogonal=True)
    slab.pbc = [True, True, False]
    n_slab = len(slab)
    mats = inplane_symmetry_matrices(slab, n_slab=n_slab)
    assert len(mats) == 2  # identity + C2

    z0 = float(slab.get_positions()[:, 2].max()) + 1.5
    a = slab.copy() + Atoms(
        "PtPt",
        positions=[[1.0, 1.0, z0], [2.5, 1.0, z0]],
    )
    center = symmetry_anchor_center(a, n_slab=n_slab)
    assert center is not None
    mobile_mask = np.zeros(len(a), dtype=bool)
    mobile_mask[n_slab:] = True
    anchor_mask = np.zeros(len(a), dtype=bool)
    anchor_mask[:n_slab] = True
    prod_pos = apply_inplane_symmetry(
        a.get_positions(),
        mats[1],
        center=center,
        mobile_mask=mobile_mask,
        anchor_mask=anchor_mask,
        ref_pos=a.get_positions(),
    )
    aligned, used = _align_product_surface_pbc(
        a,
        prod_pos,
        n_slab=n_slab,
        enable_lattice_rotation=False,
        allow_symmetry_copies=True,
    )
    disp = np.linalg.norm(aligned[n_slab:] - a.get_positions()[n_slab:], axis=1)
    assert float(np.max(disp)) < 0.05
    assert used is True


def test_hex_symmetry_matrices_exclude_reflections():
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=8.0, orthogonal=False)
    slab.pbc = [True, True, False]
    mats = inplane_symmetry_matrices(slab, n_slab=len(slab))
    assert len(mats) > 1
    for m in mats:
        assert abs(float(np.linalg.det(m)) - 1.0) < 1e-8


def test_hex_kabsch_competes_with_discrete_when_rotation_on():
    """Continuous Kabsch must beat discrete ops for a non-lattice rotation."""
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=8.0, orthogonal=False)
    slab.pbc = [True, True, False]
    n_slab = len(slab)
    assert len(inplane_symmetry_matrices(slab, n_slab=n_slab)) > 1

    z0 = float(slab.get_positions()[:, 2].max()) + 1.5
    a = slab.copy() + Atoms(
        "PtPt",
        positions=[[1.0, 1.0, z0], [2.5, 1.0, z0]],
    )
    rot20 = ns._rotation_about_normal(20.0, 2)
    prod_pos = a.get_positions().copy()
    mobile = prod_pos[n_slab:]
    com = mobile.mean(axis=0)
    prod_pos[n_slab:] = (mobile - com) @ rot20.T + com

    aligned, used = _align_product_surface_pbc(
        a,
        prod_pos,
        n_slab=n_slab,
        enable_lattice_rotation=True,
        allow_symmetry_copies=True,
    )
    disp = float(
        np.max(np.linalg.norm(aligned[n_slab:] - a.get_positions()[n_slab:], axis=1))
    )
    assert disp < 0.05
    assert used is False

    aligned_no_kabsch, _ = _align_product_surface_pbc(
        a,
        prod_pos,
        n_slab=n_slab,
        enable_lattice_rotation=False,
        allow_symmetry_copies=True,
    )
    disp_no = float(
        np.max(
            np.linalg.norm(
                aligned_no_kabsch[n_slab:] - a.get_positions()[n_slab:], axis=1
            )
        )
    )
    assert disp_no > 10.0 * disp


def test_symmetry_product_rejected_thresholds():
    assert symmetry_product_rejected(0.2, 0.0, used_symmetry_copy=True) is True
    assert symmetry_product_rejected(0.05, 0.0, used_symmetry_copy=True) is False
    assert symmetry_product_rejected(0.2, 0.0, used_symmetry_copy=False) is False
    assert symmetry_product_rejected(0.2, None, used_symmetry_copy=True) is False
    assert SYMMETRY_DRIFT_EV == 0.1


def test_find_transition_state_symmetry_drift_skips(monkeypatch, tmp_path):
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=8.0, orthogonal=False)
    slab.pbc = [True, True, False]
    n_slab = len(slab)
    z0 = float(slab.get_positions()[:, 2].max()) + 1.5
    a = slab.copy() + Atoms("Pt", positions=[[0.5, 0.5, z0]])
    a.calc = EMT()
    e_ref = float(a.get_potential_energy())
    b = slab.copy() + Atoms("Pt", positions=[[1.2, 0.8, z0]])
    b.calc = EMT()
    set_tags(a, potential_energy=e_ref)
    set_tags(b, potential_energy=float(b.get_potential_energy()))

    seen: list[bool] = []
    orig = interpolate_path

    def _spy(*args, **kwargs):
        imgs = orig(*args, **kwargs)
        seen.append(bool(kwargs.get("allow_symmetry_copies", True)))
        if kwargs.get("allow_symmetry_copies", True):
            imgs[-1].info["scgo_symmetry_copy"] = True
        return imgs

    def _reject(*_args, **_kwargs):
        raise SCGOValidationError("symmetry copy drifted")

    monkeypatch.setattr("scgo.ts_search.transition_state.interpolate_path", _spy)
    monkeypatch.setattr(
        "scgo.ts_search.transition_state.ensure_symmetry_copy_energy",
        _reject,
    )

    result = find_transition_state(
        a,
        b,
        calculator=EMT(),
        output_dir=str(tmp_path / "sym"),
        pair_id="sym_retry",
        n_images=1,
        fmax=5.0,
        neb_steps=2,
        verbosity=0,
        n_slab=n_slab,
        neb_interpolation_mic=True,
        system_type="surface_cluster",
    )
    assert seen == [True]
    assert result["status"] == "skipped"


def test_parallel_neb_symmetry_veto_without_mismatch_screen(tmp_path):
    """Symmetry drift is rejected even when max_endpoint_mismatch is None."""
    from unittest.mock import patch

    from scgo.ts_search.parallel_neb import run_parallel_neb_search
    from scgo.utils.ts_runner_kwargs import NebRunConfig

    slab = fcc111("Pt", size=(2, 2, 1), vacuum=8.0, orthogonal=False)
    slab.pbc = [True, True, False]
    n_slab = len(slab)
    z0 = float(slab.get_positions()[:, 2].max()) + 1.5
    a = slab.copy() + Atoms("Pt", positions=[[0.5, 0.5, z0]])
    b = slab.copy() + Atoms("Pt", positions=[[1.2, 0.8, z0]])
    set_tags(a, potential_energy=0.0)
    set_tags(b, potential_energy=0.05)

    cfg = NebRunConfig(
        neb_n_images=1,
        neb_spring_constant=0.1,
        neb_fmax=5.0,
        neb_steps=2,
        neb_climb=False,
        neb_interpolation_method="linear",
        neb_align_endpoints=True,
        neb_perturb_sigma=0.0,
        neb_interpolation_mic=True,
        neb_tangent_method="aseneb",
        neb_surface_cell_remap=True,
        neb_surface_lattice_rotation=True,
        neb_surface_max_lattice_shift=1,
        n_slab=n_slab,
        n_core_mobile=None,
        n_adsorbate_mobile=None,
        adsorbate_fragment_lengths=None,
        max_endpoint_mismatch=None,
        adsorbate_definition=None,
        connectivity_factor=None,
        allow_cluster_fragmentation=True,
        allow_adsorbate_surface_detachment=False,
        enforce_adsorbate_subgraph_integrity=True,
        system_type="surface_cluster",
        surface_config=None,
        torchsim_params={},
        neb_prescreen_clash_distance=0.7,
        min_saddle_prominence=0.40,
        neb_max_spurious_barrier=8.0,
        layer_cluster_threshold_ang=0.4,
        neb_interpolation_bond_tolerance_a=0.5,
    )

    def _fake_interpolate(a1, a2, **_kwargs):
        imgs = [a1.copy(), a2.copy()]
        imgs[-1].info["scgo_symmetry_copy"] = True
        return imgs

    class _FakeRelaxer:
        def relax_batch(self, atoms_list, steps=0):
            return [
                (0.5, atoms.copy())  # drifted vs GO product 0.05 eV
                for atoms in atoms_list
            ]

    with (
        patch(
            "scgo.ts_search.parallel_neb.prepare_neb_endpoints",
            side_effect=lambda x, y, _cfg: (x.copy(), y.copy()),
        ),
        patch(
            "scgo.ts_search.parallel_neb.interpolate_path",
            side_effect=_fake_interpolate,
        ),
        patch("scgo.ts_search.parallel_neb.validate_initial_neb_path"),
        patch("scgo.ts_search.parallel_neb.save_neb_result"),
    ):
        results, _meta = run_parallel_neb_search(
            [(0, 1)],
            [(0.0, a), (0.05, b)],
            neb_cfg=cfg,
            run_dir=tmp_path / "par_sym",
            rng=None,
            verbosity=0,
            relaxer=_FakeRelaxer(),
        )

    assert len(results) == 1
    assert results[0]["status"] == "skipped"
    assert "symmetry copy" in str(results[0].get("error", "")).lower()


# --- Springs ------------------------------------------------------------------


def test_variable_spring_constants_peak_and_flat():
    k = variable_spring_constants(np.array([0.0, 0.1, 1.0, 0.1, 0.0]), 0.1)
    assert k.shape == (4,)
    assert k[1] == pytest.approx(4.0)
    assert k[0] == pytest.approx(4.0 - 3.9 * 0.9)
    flat = variable_spring_constants(np.array([1.0, 1.0, 1.0, 1.0]), 0.1)
    np.testing.assert_allclose(flat, 0.1)


def test_variable_spring_constants_clamp_when_k_min_above_k_max():
    k = variable_spring_constants(np.array([0.0, 0.1, 1.0, 0.1, 0.0]), 5.0)
    assert k.shape == (4,)
    assert float(np.min(k)) >= 5.0 - 1e-12
    assert float(np.max(k)) == pytest.approx(5.0)


def test_neb_accepts_variable_spring_array():
    a = Atoms("Cu2", positions=[[0, 0, 0], [2.5, 0, 0]], cell=[10, 10, 10], pbc=False)
    b = Atoms("Cu2", positions=[[0, 0, 0], [2.7, 0.2, 0]], cell=[10, 10, 10], pbc=False)
    for atoms in (a, b):
        atoms.calc = EMT()
    images = [a.copy(), a.copy(), b.copy()]
    for img in images:
        img.calc = EMT()
    k = variable_spring_constants(np.array([0.0, 1.0, 0.0]), 0.1)
    neb = NEB(images, k=k.tolist())
    forces = neb.get_forces()
    assert np.all(np.isfinite(forces))


def test_parallel_neb_updates_k_on_surface(monkeypatch):
    pytest.importorskip("torch")

    a = Atoms(
        "Cu3",
        positions=[[0, 0, 0], [2.5, 0, 0], [1.25, 2.1, 0]],
        cell=[8, 8, 12],
        pbc=[True, True, False],
    )
    b = a.copy()
    b.positions[2, 0] += 0.3
    for atoms in (a, b):
        atoms.calc = EMT()

    class _FakeRelaxer:
        def relax_batch(self, images, steps=0):
            out = []
            for img in images:
                img = img.copy()
                img.calc = EMT()
                e = float(img.get_potential_energy())
                f = img.get_forces()
                from ase.calculators.singlepoint import SinglePointCalculator

                img.calc = SinglePointCalculator(img, energy=e, forces=f)
                out.append((e, img))
            return out

    images1 = [a.copy(), a.copy(), b.copy()]
    images2 = [a.copy(), a.copy(), b.copy()]
    for imgs in (images1, images2):
        for img in imgs:
            img.calc = EMT()
            e = img.get_potential_energy()
            f = img.get_forces()
            from ase.calculators.singlepoint import SinglePointCalculator

            img.calc = SinglePointCalculator(img, energy=e, forces=f)

    relaxer = _FakeRelaxer()
    monkeypatch.setattr(ns, "K_UPDATE_EVERY", 1)
    neb1 = TorchSimNEB(images1, relaxer, k=0.1, climb=False)
    neb1._scgo_n_slab = 0
    neb1._scgo_k_min = 0.1
    neb2 = TorchSimNEB(images2, relaxer, k=0.1, climb=False)
    neb2._scgo_n_slab = 0
    neb2._scgo_k_min = 0.1
    batch = ParallelNEBBatch([neb1, neb2], relaxer, max_total_steps=2)
    batch.run_optimization(fmax=10.0, max_steps=2)
    assert isinstance(neb1.k, (list, np.ndarray))
    assert len(np.asarray(neb1.k)) == len(images1) - 1

    monkeypatch.setattr(ns, "K_UPDATE_EVERY", 40)
    neb3 = TorchSimNEB(images1, relaxer, k=0.1, climb=False)
    neb3._scgo_n_slab = 0
    neb3._scgo_k_min = 0.1
    batch2 = ParallelNEBBatch([neb3], relaxer, max_total_steps=1)
    batch2.run_optimization(fmax=10.0, max_steps=1)
    # No refresh within one step when interval is 40; k stays scalar (or length-1).
    k3 = np.asarray(neb3.k, dtype=float).ravel()
    assert k3.size >= 1
    assert float(k3[0]) == pytest.approx(0.1)

    # Gas (no 2D PBC): k unchanged even with interval=1.
    monkeypatch.setattr(ns, "K_UPDATE_EVERY", 1)
    gas = Atoms(
        "Cu3",
        positions=[[0, 0, 0], [2.5, 0, 0], [1.25, 2.1, 0]],
        cell=[12, 12, 12],
        pbc=False,
    )
    gas_b = gas.copy()
    gas_b.positions[2, 0] += 0.2
    imgs = [gas.copy(), gas.copy(), gas_b.copy()]
    for img in imgs:
        img.calc = EMT()
        e = img.get_potential_energy()
        f = img.get_forces()
        from ase.calculators.singlepoint import SinglePointCalculator

        img.calc = SinglePointCalculator(img, energy=e, forces=f)
    neb_g = TorchSimNEB(imgs, relaxer, k=0.1, climb=False)
    neb_g._scgo_n_slab = 0
    neb_g._scgo_k_min = 0.1
    batch_g = ParallelNEBBatch([neb_g], relaxer, max_total_steps=2)
    batch_g.run_optimization(fmax=10.0, max_steps=2)
    assert float(np.asarray(neb_g.k).ravel()[0]) == pytest.approx(0.1)


# --- Fidelity -----------------------------------------------------------------


def test_bond_set_mobile_only():
    dimer = Atoms("Pt2", positions=[[0, 0, 0], [2.4, 0, 0]], pbc=False)
    assert (0, 1) in bond_set(dimer, n_slab=0)
    far = Atoms("Pt2", positions=[[0, 0, 0], [5.0, 0, 0]], pbc=False)
    assert (0, 1) not in bond_set(far, n_slab=0)
    # Slab atom bonded to mobile is excluded.
    trio = Atoms(
        "Pt3",
        positions=[[0, 0, 0], [2.4, 0, 0], [4.8, 0, 0]],
        pbc=False,
    )
    bs = bond_set(trio, n_slab=1)
    assert all(i >= 1 and j >= 1 for i, j in bs)


def _synthetic_band_labels():
    """Seven images: R,R,T,T,T,P,P with distinct bond graphs via distances."""
    # Reactant: only (0,1) bonded. Product: only (1,2). Transition: none.
    positions = [
        [[0, 0, 0], [2.4, 0, 0], [10.0, 0, 0]],  # R
        [[0, 0, 0], [2.4, 0, 0], [10.0, 0, 0]],  # R
        [[0, 0, 0], [5.0, 0, 0], [10.0, 0, 0]],  # T
        [[0, 0, 0], [5.0, 0, 0], [10.0, 0, 0]],  # T
        [[0, 0, 0], [5.0, 0, 0], [10.0, 0, 0]],  # T
        [[0, 0, 0], [7.6, 0, 0], [10.0, 0, 0]],  # P (1-2 at 2.4)
        [[0, 0, 0], [7.6, 0, 0], [10.0, 0, 0]],  # P
    ]
    images = [Atoms("Pt3", positions=p, cell=[20, 20, 20], pbc=True) for p in positions]
    return images


def test_classify_band_unimodal_success():
    images = _synthetic_band_labels()
    energies = [0.0, 0.1, 0.5, 1.0, 0.6, 0.2, 0.0]
    single, at_bond = classify_band(images, energies, n_slab=0)
    assert single is True
    assert at_bond is True


def test_classify_band_local_min_fails_single_step():
    images = _synthetic_band_labels()
    energies = [0.0, 0.1, 0.8, 0.3, 0.9, 0.2, 0.0]
    single, at_bond = classify_band(images, energies, n_slab=0)
    assert single is False
    assert at_bond is True


def test_classify_band_max_on_reactant_like():
    images = _synthetic_band_labels()
    energies = [1.0, 0.9, 0.5, 0.4, 0.3, 0.2, 0.0]
    single, at_bond = classify_band(images, energies, n_slab=0)
    assert at_bond is False
    assert single is False


def test_classify_band_identical_endpoints():
    images = [Atoms("Pt2", positions=[[0, 0, 0], [2.4, 0, 0]]) for _ in range(4)]
    energies = [0.0, 0.5, 0.5, 0.0]
    assert classify_band(images, energies, n_slab=0) == (None, None)


def test_classify_band_no_transition_images():
    r = Atoms("Pt3", positions=[[0, 0, 0], [2.4, 0, 0], [10.0, 0, 0]])
    p = Atoms("Pt3", positions=[[0, 0, 0], [7.6, 0, 0], [10.0, 0, 0]])
    images = [r, r.copy(), p.copy(), p.copy()]
    energies = [0.0, 0.1, 0.2, 0.0]
    single, at_bond = classify_band(images, energies, n_slab=0)
    assert single is False
    assert at_bond is False


def test_finalize_writes_fidelity_without_clearing_status():
    images = [
        Atoms("Pt2", positions=[[0, 0, 0], [2.4, 0, 0]]),
        Atoms("Pt2", positions=[[0, 0, 0], [3.2, 0, 0]]),
        Atoms("Pt2", positions=[[0, 0, 0], [3.5, 0, 0]]),
        Atoms("Pt2", positions=[[0, 0, 0], [5.0, 0, 0]]),
    ]
    for e, img in zip([0.0, 0.5, 1.0, 0.1], images, strict=True):
        from ase.calculators.singlepoint import SinglePointCalculator

        img.calc = SinglePointCalculator(img, energy=e, forces=np.zeros((2, 3)))
    result = {
        "pair_id": "fid",
        "neb_converged": True,
        "reactant_energy": 0.0,
        "product_energy": 0.1,
        "status": "failed",
    }
    _finalize_neb_result(result, images, n_slab=0)
    assert "fidelity_single_step" in result
    assert "fidelity_energy_at_bond_change" in result
    assert result["status"] == "success"
    assert result["neb_converged"] is True


def test_count_imaginary_frequencies_cutoff():
    from ase import units

    s = ns._HBAR_TO_ENERGY
    invcm = units.invcm
    e100 = 1j * 100.0 * invcm
    e10 = 1j * 10.0 * invcm
    o100 = complex(e100 / s) ** 2
    o10 = complex(e10 / s) ** 2
    omega2 = np.array([o100.real, o10.real, 1.0], dtype=float)
    assert count_imaginary_frequencies(omega2) == 1


def test_displaced_mobile_images_shape():
    atoms = Atoms("Pt2", positions=[[0, 0, 0], [2.4, 0, 0]])
    imgs = displaced_mobile_images(atoms, n_slab=1, delta=0.01)
    assert len(imgs) == 6
    base = atoms.get_positions()
    for img in imgs:
        d = img.get_positions() - base
        assert np.allclose(d[0], 0.0)
        assert abs(float(np.linalg.norm(d[1])) - 0.01) < 1e-12


def test_imaginary_modes_from_negative_curvature_force_fn():
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    # F = +c * x  (negative curvature). All 3 Cartesian modes are imaginary.
    c = 50.0

    def force_fn(displaced: list[Atoms]) -> list[np.ndarray]:
        out = []
        for img in displaced:
            x = img.get_positions()
            out.append(c * x)
        return out

    n_im = count_imaginary_modes_from_forces(atoms, force_fn, n_slab=0)
    assert n_im == 3

    images = [atoms.copy() for _ in range(3)]
    for e, img in zip([0.0, 1.0, 0.0], images, strict=True):
        from ase.calculators.singlepoint import SinglePointCalculator

        img.calc = SinglePointCalculator(img, energy=e, forces=np.zeros((1, 3)))
    result = {
        "pair_id": "im",
        "neb_converged": True,
        "reactant_energy": 0.0,
        "product_energy": 0.0,
    }
    _finalize_neb_result(
        result,
        images,
        n_slab=0,
        force_fn=force_fn,
    )
    assert result["status"] == "success"
    assert result["n_imaginary_modes"] == 3

    def boom(_imgs):
        raise RuntimeError("no hessian")

    result2 = {
        "pair_id": "im2",
        "neb_converged": True,
        "reactant_energy": 0.0,
        "product_energy": 0.0,
    }
    _finalize_neb_result(result2, images, n_slab=0, force_fn=boom)
    assert result2["status"] == "success"
    assert result2["n_imaginary_modes"] is None
