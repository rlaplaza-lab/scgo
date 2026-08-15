"""Surface hull-site diversity for adsorbate placement on slab-supported clusters."""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.build import fcc111
from ase_ga.utilities import closest_distances_generator, get_all_atom_types
from numpy.random import default_rng

from scgo.metadata.atoms import get_tag
from scgo.surface.config import SurfaceSystemConfig
from scgo.surface.deposition import create_deposited_cluster_batch
from scgo.system_types import AdsorbateDefinition

_SITE_COUNTER_KEYS = ("vertex", "edge", "facet", "directional_fallback")


def _o_template() -> Atoms:
    return Atoms("O", positions=[[0.0, 0.0, 0.0]], pbc=False)


def _oh_template() -> Atoms:
    return Atoms(symbols=["O", "H"], positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0]])


def _pt111_slab() -> Atoms:
    return fcc111("Pt", size=(2, 2, 2), vacuum=6.0, orthogonal=True)


def _blmin_for(slab: Atoms, composition: list[str]) -> dict:
    n_slab = len(slab)
    template = Atoms(
        symbols=list(slab.get_chemical_symbols()) + composition,
        positions=np.vstack([slab.get_positions(), np.zeros((len(composition), 3))]),
        cell=slab.cell,
        pbc=slab.pbc,
    )
    return closest_distances_generator(
        get_all_atom_types(template, range(n_slab, n_slab + len(composition))),
        ratio_of_covalent_radii=0.7,
    )


def _deposit_batch_with_counters(
    composition: list[str],
    adsorbate_definition: dict,
    *,
    n_structures: int,
    seed: int,
) -> tuple[list[Atoms], dict[str, int]]:
    slab = _pt111_slab()
    cfg = SurfaceSystemConfig(
        slab=slab,
        adsorption_height_min=1.0,
        adsorption_height_max=2.5,
        max_placement_attempts=200,
    )
    counts = dict.fromkeys(_SITE_COUNTER_KEYS, 0)
    batch = create_deposited_cluster_batch(
        composition,
        slab,
        _blmin_for(slab, composition),
        n_structures,
        default_rng(seed),
        cfg,
        adsorbate_definition=adsorbate_definition,
        adsorbate_fragment_template=[_oh_template()],
        batch_site_counts=counts,
        n_jobs=1,
    )
    return batch, counts


def test_deposited_batch_stamps_site_type_for_cluster_seed_path() -> None:
    """The bare ``Atoms`` rebuild must not drop the seed's site metadata."""
    composition = ["Pt", "Pt", "Pt", "Pt", "O", "H"]
    ads_def = AdsorbateDefinition(
        core_symbols=["Pt", "Pt", "Pt", "Pt"],
        adsorbate_symbols=["O", "H"],
        adsorbate_fragment_lengths=[2],
    )
    batch, counts = _deposit_batch_with_counters(
        composition, ads_def, n_structures=4, seed=3
    )
    assert len(batch) == 4
    for atoms in batch:
        site_type = get_tag(atoms, "adsorbate_site_type")
        assert isinstance(site_type, str) and site_type
        assert get_tag(atoms, "adsorbate_site_types_json") is not None
    assert sum(counts.values()) == len(batch)


def test_deposited_batch_stamps_site_type_for_fragments_on_slab_path() -> None:
    """Adsorbate-only deposition (no metal core) must also stamp site metadata."""
    composition = ["O", "H"]
    ads_def = AdsorbateDefinition(
        core_symbols=[],
        adsorbate_symbols=["O", "H"],
        adsorbate_fragment_lengths=[2],
    )
    batch, counts = _deposit_batch_with_counters(
        composition, ads_def, n_structures=4, seed=5
    )
    assert len(batch) == 4
    for atoms in batch:
        site_type = get_tag(atoms, "adsorbate_site_type")
        assert isinstance(site_type, str) and site_type
    assert sum(counts.values()) == len(batch)
