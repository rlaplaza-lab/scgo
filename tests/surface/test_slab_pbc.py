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


def test_graphite_preset_uses_ab_bernal_stacking():
    """Odd layers must be laterally shifted by (a1+a2)/3 (not AA-only)."""
    import numpy as np
    from ase.geometry import find_mic

    slab = build_graphite_slab(layers=2, vacuum=8.0, repeat_xy=1)
    n_per_layer = len(slab) // 2
    pos = slab.get_positions()
    layer0 = pos[:n_per_layer]
    layer1 = pos[n_per_layer:]
    cell = np.asarray(slab.get_cell(), dtype=float)
    expected_shift = (cell[0] + cell[1]) / 3.0
    pbc_xy = [True, True, False]

    # AA stacking would place layers on top of each other in xy.
    for p0 in layer0:
        dists = [
            float(np.linalg.norm(find_mic(p1 - p0, cell, pbc=pbc_xy)[0][:2]))
            for p1 in layer1
        ]
        assert min(dists) > 0.5

    # Undoing the Bernal shift restores an AA overlay.
    for p0 in layer0:
        dists = [
            float(
                np.linalg.norm(
                    find_mic((p1 - expected_shift) - p0, cell, pbc=pbc_xy)[0][:2]
                )
            )
            for p1 in layer1
        ]
        assert min(dists) < 1e-6
