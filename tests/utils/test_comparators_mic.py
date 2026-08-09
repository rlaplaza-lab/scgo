"""Pure comparator MIC flag is honored literally under PBC."""

from __future__ import annotations

from ase import Atoms

from scgo.utils.comparators import (
    PureInteratomicDistanceComparator,
    get_sorted_dist_list,
)


def test_pure_comparator_mic_false_does_not_fold_under_pbc() -> None:
    """Periodic near-images must differ when mic=False and match when mic=True."""
    cell = [8.0, 8.0, 12.0]
    a1 = Atoms(
        "Pt2",
        positions=[[0.10, 0.0, 0.0], [7.90, 0.0, 0.0]],
        cell=cell,
        pbc=[True, True, False],
    )
    a2 = Atoms(
        "Pt2",
        positions=[[0.10, 0.0, 0.0], [-0.10, 0.0, 0.0]],
        cell=cell,
        pbc=[True, True, False],
    )
    no_mic = PureInteratomicDistanceComparator(n_top=2, mic=False)
    with_mic = PureInteratomicDistanceComparator(n_top=2, mic=True)
    assert bool(no_mic.looks_like(a1, a2)) is False
    assert bool(with_mic.looks_like(a1, a2)) is True

    fp_no = get_sorted_dist_list(a1, mic=False)
    fp_yes = get_sorted_dist_list(a1, mic=True)
    # Without MIC the long in-cell Pt–Pt distance remains ~7.8 Å.
    assert float(fp_no[78][0]) > 7.0
    assert float(fp_yes[78][0]) < 1.0


def test_resolve_structure_mic_gas_vs_surface() -> None:
    from ase.build import fcc111

    from scgo.surface.config import SurfaceSystemConfig
    from scgo.system_types import resolve_neb_mic, resolve_structure_mic

    assert resolve_structure_mic("gas_cluster") is False
    assert resolve_structure_mic("gas_cluster_adsorbate") is False
    assert resolve_neb_mic("gas_cluster") is False
    assert resolve_neb_mic("gas_cluster_adsorbate") is False

    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    cfg_on = SurfaceSystemConfig(slab=slab)
    assert cfg_on.comparator_use_mic is True
    assert resolve_structure_mic("surface_cluster", cfg_on) is True
    assert resolve_neb_mic("surface_cluster") is True
    assert resolve_neb_mic("surface_cluster_adsorbate") is True

    cfg_off = SurfaceSystemConfig(slab=slab, comparator_use_mic=False)
    assert resolve_structure_mic("surface_cluster", cfg_off) is False
