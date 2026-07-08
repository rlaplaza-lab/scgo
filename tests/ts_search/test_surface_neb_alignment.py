"""Surface NEB endpoint alignment: lattice remap and compatible rotation."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111
from ase.constraints import FixAtoms

from scgo.exceptions import SCGORuntimeError, SCGOValidationError
from scgo.surface.composition import full_adsorbate_slab_composition
from scgo.system_types import get_system_policy
from scgo.ts_search import transition_state as ts_mod
from scgo.ts_search.transition_state import (
    _align_product_for_neb,
    _align_product_kabsch_to_reactant,
    _align_product_surface_pbc,
    _inplane_rotation_matrix_3d,
    _lattice_translation_candidates,
    _validate_lattice_compatible_rotation,
    interpolate_path,
)
from scgo.utils.helpers import get_cluster_formula


def test_system_policy_surface_enables_remap_and_rotation():
    for st in ("surface_cluster", "surface_cluster_adsorbate"):
        policy = get_system_policy(st)
        assert policy.neb_surface_cell_remap is True
        assert policy.neb_surface_lattice_rotation is True


def test_validate_lattice_compatible_rotation_rejects_out_of_plane():
    # 90° rotation about x tilts the slab normal (z) into the plane.
    rot_bad = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        dtype=float,
    )
    with pytest.raises(SCGOValidationError, match="surface normal"):
        _validate_lattice_compatible_rotation(rot_bad, normal_axis=2)


def test_inplane_rotation_matrix_preserves_normal_axis():
    rot = _inplane_rotation_matrix_3d(np.deg2rad(40.0), normal_axis=2)
    _validate_lattice_compatible_rotation(rot, normal_axis=2)


def test_surface_alignment_cell_remap_shortens_periodic_jump():
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    slab.pbc = [True, True, False]
    z0 = slab.get_positions()[:, 2].max() + 1.5
    n_slab = len(slab)

    a = slab.copy() + Atoms("Pt", positions=[[0.1, 0.1, z0]])
    b = slab.copy() + Atoms("Pt", positions=[[slab.cell[0, 0] - 0.1, 0.1, z0]])

    raw = b.get_positions().copy()
    aligned = _align_product_surface_pbc(
        a, raw, n_slab=n_slab, enable_cell_remap=True, enable_lattice_rotation=False
    )
    disp = aligned - a.get_positions()
    assert abs(float(disp[-1, 0])) < 0.5


def test_surface_alignment_rotation_reduces_mobile_rms():
    slab = fcc111("Pt", size=(2, 2, 2), vacuum=6.0, orthogonal=True)
    slab.pbc = [True, True, False]
    n_slab = len(slab)
    z0 = slab.get_positions()[:, 2].max() + 1.8

    mobile = np.array(
        [
            [0.0, 0.0, z0],
            [1.2, 0.0, z0],
            [0.6, 1.0, z0 + 0.3],
        ]
    )
    theta = np.deg2rad(35.0)
    rot2 = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    mobile_rot = mobile.copy()
    mobile_rot[:, :2] = (mobile[:, :2] - mobile[:, :2].mean(axis=0)) @ rot2.T
    mobile_rot[:, :2] += mobile[:, :2].mean(axis=0)
    mobile_rot[:, 0] += slab.cell[0, 0]

    a = slab.copy() + Atoms("Pt3", positions=mobile)
    b = slab.copy() + Atoms("Pt3", positions=mobile_rot)

    aligned = _align_product_surface_pbc(
        a,
        b.get_positions(),
        n_slab=n_slab,
        enable_cell_remap=True,
        enable_lattice_rotation=True,
    )
    rms = float(np.sqrt(np.mean((aligned[n_slab:] - a.get_positions()[n_slab:]) ** 2)))
    assert rms < 0.15
    slab_disp = np.linalg.norm(aligned[:n_slab] - a.get_positions()[:n_slab], axis=1)
    assert float(np.max(slab_disp)) < 1e-6


def test_interpolate_path_surface_uses_pbc_align_entrypoint(monkeypatch):
    """Slab NEB routes endpoint alignment through ``_align_product_for_neb``."""
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    slab.pbc = [True, True, False]
    z0 = slab.get_positions()[:, 2].max() + 1.5
    n_slab = len(slab)
    a = slab.copy() + Atoms("Pt", positions=[[0.1, 0.1, z0]])
    b = slab.copy() + Atoms("Pt", positions=[[slab.cell[0, 0] - 0.1, 0.1, z0]])

    called = {"for_neb": 0}
    orig_for_neb = ts_mod._align_product_for_neb

    def _track_for_neb(*args, **kwargs):
        called["for_neb"] += 1
        return orig_for_neb(*args, **kwargs)

    monkeypatch.setattr(ts_mod, "_align_product_for_neb", _track_for_neb)

    interpolate_path(
        a,
        b,
        n_images=2,
        method="linear",
        mic=True,
        align_endpoints=True,
        n_slab=n_slab,
        system_type="surface_cluster",
    )
    assert called["for_neb"] == 1


def test_interpolate_path_surface_unifies_product_cell():
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    slab.pbc = [True, True, False]
    z0 = slab.get_positions()[:, 2].max() + 1.5
    n_slab = len(slab)
    a = slab.copy() + Atoms("Pt", positions=[[0.1, 0.1, z0]])
    b = slab.copy() + Atoms("Pt", positions=[[slab.cell[0, 0] - 0.1, 0.1, z0]])
    b.cell[0, 0] += 0.01

    images = interpolate_path(
        a,
        b,
        n_images=2,
        mic=True,
        align_endpoints=True,
        n_slab=n_slab,
        system_type="surface_cluster",
    )
    assert np.allclose(images[-1].cell, images[0].cell)
    assert list(images[-1].pbc) == list(images[0].pbc)


def test_kabsch_align_rejects_slab_systems():
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    slab.pbc = [True, True, False]
    a = slab.copy() + Atoms("Pt", positions=[[0.0, 0.0, 8.0]])
    with pytest.raises(SCGORuntimeError, match="Slab NEB endpoints must use"):
        _align_product_kabsch_to_reactant(a, a.get_positions(), n_slab=len(slab))


def test_align_product_for_neb_routes_mic_alias_to_surface():
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    slab.pbc = [True, True, False]
    z0 = slab.get_positions()[:, 2].max() + 1.5
    a = slab.copy() + Atoms("Pt", positions=[[0.1, 0.1, z0]])
    b = slab.copy() + Atoms("Pt", positions=[[slab.cell[0, 0] - 0.1, 0.1, z0]])
    via_for_neb = _align_product_for_neb(a, b.get_positions(), n_slab=len(slab))
    via_surface = _align_product_surface_pbc(a, b.get_positions(), n_slab=len(slab))
    np.testing.assert_allclose(via_for_neb, via_surface, atol=1e-8)


def test_interpolate_path_endpoints_unchanged_by_ase_interpolate(monkeypatch):
    """ASE ``NEB.interpolate`` must not move aligned endpoint images before optimization."""
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    slab.pbc = [True, True, False]
    z0 = slab.get_positions()[:, 2].max() + 1.5
    n_slab = len(slab)
    a = slab.copy() + Atoms("Pt", positions=[[0.1, 0.1, z0]])
    b = slab.copy() + Atoms("Pt", positions=[[slab.cell[0, 0] - 0.1, 0.1, z0]])

    captured: dict[str, np.ndarray] = {}
    from ase.mep import NEB as AseNEB

    _orig = AseNEB.interpolate

    def _record_endpoints(self, *args, **kwargs):
        captured["reactant"] = self.images[0].get_positions().copy()
        captured["product"] = self.images[-1].get_positions().copy()
        return _orig(self, *args, **kwargs)

    monkeypatch.setattr(AseNEB, "interpolate", _record_endpoints)

    images = interpolate_path(
        a,
        b,
        n_images=2,
        method="linear",
        mic=True,
        align_endpoints=True,
        n_slab=n_slab,
        system_type="surface_cluster",
    )
    np.testing.assert_allclose(captured["reactant"], images[0].get_positions())
    np.testing.assert_allclose(captured["product"], images[-1].get_positions())


def test_run_transition_state_search_forwards_alignment_kwargs(monkeypatch, tmp_path):
    """Runner should pass slab/block dims and max lattice shift into interpolation."""
    from scgo.surface.config import SurfaceSystemConfig
    from scgo.ts_search import transition_state_run as ts_run_mod

    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    slab.pbc = [True, True, False]
    n_slab = len(slab)
    z0 = slab.get_positions()[:, 2].max() + 1.5
    react = slab.copy() + Atoms("Pt", positions=[[0.1, 0.1, z0]])
    prod = slab.copy() + Atoms("Pt", positions=[[slab.cell[0, 0] - 0.1, 0.1, z0]])
    cfg = SurfaceSystemConfig(slab=slab, fix_all_slab_atoms=True)
    captured: dict[str, object] = {}

    def _fake_find_transition_state(reactant, product, calculator, **kwargs):
        captured.update(kwargs)
        return {
            "status": "failed",
            "pair_id": kwargs.get("pair_id", "stub"),
            "error": "stub",
            "neb_converged": False,
        }

    monkeypatch.setattr(
        ts_run_mod, "find_transition_state", _fake_find_transition_state
    )
    monkeypatch.setattr(ts_run_mod, "save_neb_result", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        ts_run_mod, "save_transition_state_results", lambda *args, **kwargs: None
    )
    full_comp = full_adsorbate_slab_composition(["Pt"], cfg)
    formula = get_cluster_formula(full_comp)
    monkeypatch.setattr(
        ts_run_mod,
        "load_minima_by_composition",
        lambda *_a, **_k: {
            formula: [
                (0.0, react),
                (0.1, prod),
            ]
        },
    )
    monkeypatch.setattr(
        ts_run_mod,
        "select_structure_pairs",
        lambda *_a, **_k: [(0, 1)],
    )
    monkeypatch.setattr(ts_run_mod, "get_calculator_class", lambda _n: object)
    monkeypatch.setattr(ts_run_mod, "auto_niter_ts", lambda _c: 10)

    ts_run_mod.run_transition_state_search(
        composition=["Pt"],
        system_type="surface_cluster",
        output_dir=str(tmp_path),
        params={"calculator": "EMT", "calculator_kwargs": {}},
        surface_config=cfg,
        verbosity=0,
        neb_surface_max_lattice_shift=3,
    )
    assert captured["n_slab"] == n_slab
    assert captured["neb_surface_max_lattice_shift"] == 3


def test_get_ts_search_params_surface_keeps_alignment_defaults():
    from scgo.param_presets import get_ts_search_params
    from scgo.surface.config import SurfaceSystemConfig

    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    cfg = SurfaceSystemConfig(slab=slab, fix_all_slab_atoms=True)
    ts = get_ts_search_params(system_type="surface_cluster", surface_config=cfg)
    assert ts["neb_align_endpoints"] is True
    assert ts["neb_interpolation_mic"] is True
    assert ts["neb_surface_cell_remap"] is True
    assert ts["neb_surface_lattice_rotation"] is True
    assert ts["neb_surface_max_lattice_shift"] == 1


def _slab_with_mobile_pt(*, size=(2, 2, 1), mobile_xy=(0.1, 0.1)):
    slab = fcc111("Pt", size=size, vacuum=6.0, orthogonal=True)
    slab.pbc = [True, True, False]
    z0 = slab.get_positions()[:, 2].max() + 1.5
    n_slab = len(slab)
    a = slab.copy() + Atoms("Pt", positions=[[mobile_xy[0], mobile_xy[1], z0]])
    return slab, a, n_slab, z0


def test_surface_alignment_y_axis_periodic_jump():
    slab, a, n_slab, z0 = _slab_with_mobile_pt()
    b = slab.copy() + Atoms("Pt", positions=[[0.1, slab.cell[1, 1] - 0.1, z0]])
    aligned = _align_product_surface_pbc(
        a, b.get_positions(), n_slab=n_slab, enable_lattice_rotation=False
    )
    disp = aligned - a.get_positions()
    assert abs(float(disp[-1, 1])) < 0.5


def test_surface_alignment_diagonal_two_cell_wrap():
    slab, a, n_slab, z0 = _slab_with_mobile_pt()
    shift = slab.cell[0] + slab.cell[1]
    b = slab.copy() + Atoms("Pt", positions=[[0.1 + shift[0], 0.1 + shift[1], z0]])
    aligned = _align_product_surface_pbc(
        a,
        b.get_positions(),
        n_slab=n_slab,
        max_lattice_shift=2,
        enable_lattice_rotation=False,
    )
    disp = aligned - a.get_positions()
    assert float(np.linalg.norm(disp[-1])) < 0.5


def test_lattice_translation_candidates_span_grows_with_max_shift():
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    cell = np.asarray(slab.cell.array, dtype=float)
    small = _lattice_translation_candidates(cell, 0, 1, max_shift=1)
    large = _lattice_translation_candidates(cell, 0, 1, max_shift=2)
    assert len(large) > len(small)
    assert len(small) == 9
    assert len(large) == 25


def test_surface_alignment_split_periodic_images_multi_atom():
    """Mobile atoms stored in inconsistent periodic images should still align."""
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    slab.pbc = [True, True, False]
    n_slab = len(slab)
    z0 = slab.get_positions()[:, 2].max() + 1.8
    mobile = np.array(
        [
            [0.0, 0.0, z0],
            [1.2, 0.0, z0],
            [0.6, 1.0, z0 + 0.3],
        ]
    )
    a = slab.copy() + Atoms("Pt3", positions=mobile)
    mobile_split = mobile.copy()
    mobile_split[1, 0] += slab.cell[0, 0]
    mobile_split[2, 1] += slab.cell[1, 1]
    b = slab.copy() + Atoms("Pt3", positions=mobile_split)

    aligned = _align_product_surface_pbc(
        a, b.get_positions(), n_slab=n_slab, enable_lattice_rotation=False
    )
    rms = float(np.sqrt(np.mean((aligned[n_slab:] - a.get_positions()[n_slab:]) ** 2)))
    assert rms < 0.2


def test_surface_alignment_remap_only_shortens_x_wrap():
    slab, a, n_slab, _z0 = _slab_with_mobile_pt()
    b = slab.copy() + Atoms(
        "Pt", positions=[[slab.cell[0, 0] - 0.1, 0.1, a.get_positions()[-1, 2]]]
    )
    aligned = _align_product_surface_pbc(
        a,
        b.get_positions(),
        n_slab=n_slab,
        enable_cell_remap=True,
        enable_lattice_rotation=False,
    )
    disp = aligned - a.get_positions()
    assert abs(float(disp[-1, 0])) < 0.5


def test_surface_alignment_remap_disabled_leaves_rotated_cluster_misaligned():
    slab = fcc111("Pt", size=(2, 2, 2), vacuum=6.0, orthogonal=True)
    slab.pbc = [True, True, False]
    n_slab = len(slab)
    z0 = slab.get_positions()[:, 2].max() + 1.8
    mobile = np.array(
        [
            [0.0, 0.0, z0],
            [1.2, 0.0, z0],
            [0.6, 1.0, z0 + 0.3],
        ]
    )
    theta = np.deg2rad(35.0)
    rot2 = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    mobile_rot = mobile.copy()
    mobile_rot[:, :2] = (mobile[:, :2] - mobile[:, :2].mean(axis=0)) @ rot2.T
    mobile_rot[:, :2] += mobile[:, :2].mean(axis=0)
    mobile_rot[:, 0] += slab.cell[0, 0]
    a = slab.copy() + Atoms("Pt3", positions=mobile)
    b = slab.copy() + Atoms("Pt3", positions=mobile_rot)

    aligned = _align_product_surface_pbc(
        a,
        b.get_positions(),
        n_slab=n_slab,
        enable_cell_remap=False,
        enable_lattice_rotation=False,
    )
    rms = float(np.sqrt(np.mean((aligned[n_slab:] - a.get_positions()[n_slab:]) ** 2)))
    assert rms > 0.15


def test_surface_alignment_rotation_reduces_rms_on_rotated_cluster():
    slab = fcc111("Pt", size=(2, 2, 2), vacuum=6.0, orthogonal=True)
    slab.pbc = [True, True, False]
    n_slab = len(slab)
    z0 = slab.get_positions()[:, 2].max() + 1.8
    mobile = np.array(
        [
            [0.0, 0.0, z0],
            [1.2, 0.0, z0],
            [0.6, 1.0, z0 + 0.3],
        ]
    )
    theta = np.deg2rad(35.0)
    rot2 = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    mobile_rot = mobile.copy()
    mobile_rot[:, :2] = (mobile[:, :2] - mobile[:, :2].mean(axis=0)) @ rot2.T
    mobile_rot[:, :2] += mobile[:, :2].mean(axis=0)
    a = slab.copy() + Atoms("Pt3", positions=mobile)
    b = slab.copy() + Atoms("Pt3", positions=mobile_rot)

    no_rot = _align_product_surface_pbc(
        a,
        b.get_positions(),
        n_slab=n_slab,
        enable_cell_remap=False,
        enable_lattice_rotation=False,
    )
    with_rot = _align_product_surface_pbc(
        a,
        b.get_positions(),
        n_slab=n_slab,
        enable_cell_remap=False,
        enable_lattice_rotation=True,
    )
    rms_no = float(
        np.sqrt(np.mean((no_rot[n_slab:] - a.get_positions()[n_slab:]) ** 2))
    )
    rms_yes = float(
        np.sqrt(np.mean((with_rot[n_slab:] - a.get_positions()[n_slab:]) ** 2))
    )
    assert rms_yes < rms_no
    assert rms_yes < 0.15


def test_interpolate_path_forwards_max_lattice_shift(monkeypatch):
    slab, a, n_slab, z0 = _slab_with_mobile_pt()
    b = slab.copy() + Atoms("Pt", positions=[[0.1 + 2.0 * slab.cell[0, 0], 0.1, z0]])
    captured: dict[str, int] = {}

    def _spy_align(reactant, product_positions, **kwargs):
        captured["max_shift"] = kwargs.get("max_lattice_shift", -1)
        return _align_product_surface_pbc(reactant, product_positions, **kwargs)

    monkeypatch.setattr(ts_mod, "_align_product_surface_pbc", _spy_align)
    interpolate_path(
        a,
        b,
        n_images=2,
        mic=True,
        align_endpoints=True,
        n_slab=n_slab,
        system_type="surface_cluster",
        neb_surface_max_lattice_shift=2,
    )
    assert captured["max_shift"] == 2


def test_get_ts_search_params_surface_max_lattice_shift_default():
    from scgo.param_presets import get_ts_search_params
    from scgo.surface.config import SurfaceSystemConfig

    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    cfg = SurfaceSystemConfig(slab=slab, fix_all_slab_atoms=True)
    ts = get_ts_search_params(system_type="surface_cluster", surface_config=cfg)
    assert ts["neb_surface_max_lattice_shift"] == 1


def test_interpolate_path_fixed_slab_anchors_under_surface_align():
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    slab.pbc = [True, True, False]
    z0 = slab.get_positions()[:, 2].max() + 1.5
    fixed_idx = list(range(len(slab)))

    a = slab.copy() + Atoms("Pt", positions=[[0.1, 0.1, z0]])
    b = slab.copy() + Atoms("Pt", positions=[[slab.cell[0, 0] - 0.1, 0.1, z0]])
    a.set_constraint(FixAtoms(indices=fixed_idx))
    b.set_constraint(FixAtoms(indices=fixed_idx))

    images = interpolate_path(
        a,
        b,
        n_images=2,
        mic=True,
        align_endpoints=True,
        system_type="surface_cluster",
    )
    disp = images[-1].get_positions() - images[0].get_positions()
    assert float(np.max(np.linalg.norm(disp[fixed_idx], axis=1))) < 1e-2
