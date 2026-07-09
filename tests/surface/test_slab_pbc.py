"""Slab periodic boundary normalization."""

from ase import Atoms
from ase.build import fcc111

from scgo.surface import make_surface_config
from scgo.surface.config import SurfaceSystemConfig
from scgo.surface.pbc import normalize_slab_pbc
from scgo.surface.presets import build_graphite_slab


def test_normalize_slab_pbc_clears_vacuum_axis():
    slab = Atoms("Pt", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
    normalize_slab_pbc(slab)
    assert list(slab.pbc) == [True, True, False]


def test_fcc111_slab_unchanged_by_normalize():
    slab = fcc111("Pt", size=(2, 2, 2), vacuum=6.0, orthogonal=True)
    expected = list(slab.pbc)
    normalize_slab_pbc(slab)
    assert list(slab.pbc) == expected


def test_surface_system_config_rejects_all_open_pbc():
    from scgo.exceptions import SCGOValidationError

    slab = Atoms("Pt", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=False)
    try:
        SurfaceSystemConfig(slab=slab)
    except SCGOValidationError as exc:
        assert "periodic" in str(exc).lower()
    else:
        raise AssertionError("expected SCGOValidationError")


def test_surface_system_config_normalizes_3d_pbc():
    slab = Atoms("Pt", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
    cfg = SurfaceSystemConfig(slab=slab)
    assert list(cfg.slab.pbc) == [True, True, False]


def test_make_surface_config_normalizes_before_config():
    slab = Atoms("Pt", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
    cfg = make_surface_config(slab)
    assert list(cfg.slab.pbc) == [True, True, False]


def test_graphite_preset_is_slab_periodic():
    slab = build_graphite_slab(layers=2, vacuum=8.0, repeat_xy=2)
    assert list(slab.pbc) == [True, True, False]
