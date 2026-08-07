"""Slab GA stamps n_slab_atoms / system_type for downstream consumers."""

from __future__ import annotations

from ase import Atoms

from scgo.algorithms.ga_common import slab_ga_metadata_extras
from scgo.metadata.atoms import get_tag, set_tags
from scgo.surface.config import SurfaceSystemConfig


def test_slab_ga_metadata_extras_empty_without_surface() -> None:
    assert slab_ga_metadata_extras(None, 4, "gas_cluster") == {
        "system_type": "gas_cluster"
    }
    slab = Atoms("Pt2", positions=[[0, 0, 0], [2.0, 0, 0]], cell=[20, 20, 20], pbc=True)
    cfg = SurfaceSystemConfig(slab=slab)
    assert slab_ga_metadata_extras(cfg, 0, "surface_cluster") == {
        "system_type": "surface_cluster"
    }


def test_slab_ga_metadata_extras_and_set_tags(pt_slab_small) -> None:
    slab = pt_slab_small
    cfg = SurfaceSystemConfig(slab=slab)
    n_slab = len(slab)
    ads = Atoms(
        "Pt",
        positions=[[0, 0, 3.0]],
        cell=slab.get_cell(),
        pbc=slab.get_pbc(),
    )
    combined = slab + ads
    extra = slab_ga_metadata_extras(cfg, n_slab, "surface_cluster_adsorbate")
    set_tags(combined, run_id="run_test", **extra)
    assert get_tag(combined, "n_slab_atoms") == n_slab
    assert get_tag(combined, "system_type") == "surface_cluster_adsorbate"
    assert (
        get_tag(combined, "slab_chemical_symbols_json") == slab.get_chemical_symbols()
    )
    assert get_tag(combined, "run_id") == "run_test"
