"""Surface NEB endpoint alignment: lattice remap and compatible rotation."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111
from ase.constraints import FixAtoms

from scgo.system_types import get_system_policy
from scgo.ts_search import transition_state as ts_mod
from scgo.ts_search.transition_state import (
    _align_product_for_neb,
    _align_product_kabsch_to_reactant,
    _align_product_surface_pbc,
    _inplane_rotation_matrix_3d,
    _validate_lattice_compatible_rotation,
    interpolate_path,
)


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
    with pytest.raises(ValueError, match="surface normal"):
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


def test_interpolate_path_surface_uses_pbc_align_not_mobile_kabsch(monkeypatch):
    """Slab NEB must not call mobile-only Kabsch (energy-inequivalent rotation)."""
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    slab.pbc = [True, True, False]
    z0 = slab.get_positions()[:, 2].max() + 1.5
    n_slab = len(slab)
    a = slab.copy() + Atoms("Pt", positions=[[0.1, 0.1, z0]])
    b = slab.copy() + Atoms("Pt", positions=[[slab.cell[0, 0] - 0.1, 0.1, z0]])

    called = {"mobile_kabsch": 0, "for_neb": 0}
    orig_for_neb = ts_mod._align_product_for_neb

    def _track_mobile(*args, **kwargs):
        called["mobile_kabsch"] += 1
        return ts_mod._apply_inplane_mobile_kabsch(*args, **kwargs)

    def _track_for_neb(*args, **kwargs):
        called["for_neb"] += 1
        return orig_for_neb(*args, **kwargs)

    monkeypatch.setattr(ts_mod, "_apply_inplane_mobile_kabsch", _track_mobile)
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
    assert called["mobile_kabsch"] == 0


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
    with pytest.raises(RuntimeError, match="Slab NEB endpoints must use"):
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
