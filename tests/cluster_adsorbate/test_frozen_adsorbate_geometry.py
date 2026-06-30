"""Frozen adsorbate internal geometry: restore, operators, and constraints."""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase_ga.utilities import closest_distances_generator, get_all_atom_types
from numpy.random import default_rng

from scgo.algorithms.ga_common import create_mutation_operators
from scgo.ase_ga_patches.standardmutations import MirrorMutation, OverlapReliefMutation
from scgo.cluster_adsorbate.rigid import restore_rigid_adsorbate_fragments


def _oh_on_pt2() -> tuple[Atoms, Atoms]:
    core = Atoms("Pt2", positions=[[0, 0, 0], [2.4, 0, 0]], pbc=False)
    oh = Atoms("OH", positions=[[1.2, 0, 1.5], [1.2, 0, 2.46]], pbc=False)
    combined = core + oh
    combined.set_cell([20, 20, 20])
    combined.set_pbc(False)
    combined.set_tags([0, 0, 1, 1])
    return combined, oh


def test_restore_rigid_adsorbate_fragments_resets_bond_length() -> None:
    combined, oh_template = _oh_on_pt2()
    ads_def = {
        "core_symbols": ["Pt", "Pt"],
        "adsorbate_symbols": ["O", "H"],
        "adsorbate_fragment_lengths": [2],
    }
    pos = combined.get_positions()
    pos[3] += np.array([0.4, 0.0, 0.0])
    combined.set_positions(pos)
    distorted = float(np.linalg.norm(pos[3] - pos[2]))

    restore_rigid_adsorbate_fragments(
        combined,
        n_slab=0,
        adsorbate_definition=ads_def,
        fragment_templates=[oh_template],
    )
    restored = combined.get_positions()
    bond = float(np.linalg.norm(restored[3] - restored[2]))
    template_bond = float(
        np.linalg.norm(oh_template.get_positions()[1] - oh_template.get_positions()[0])
    )
    assert abs(bond - template_bond) < 1e-6
    assert abs(distorted - template_bond) > 0.05


def test_overlap_relief_use_tags_preserves_intra_fragment_geometry() -> None:
    combined, _ = _oh_on_pt2()
    blmin = closest_distances_generator([78, 8, 1], ratio_of_covalent_radii=0.7)
    op = OverlapReliefMutation(
        blmin,
        len(combined),
        system_type="gas_cluster_adsorbate",
        use_tags=True,
        rng=default_rng(0),
    )
    before = combined.get_positions()[2:].copy()
    out = op.mutate(combined)
    assert out is not None
    after = out.get_positions()[2:]
    assert np.allclose(after[1] - after[0], before[1] - before[0], atol=1e-9)


def test_mirror_target_tags_only_mutates_core() -> None:
    combined, oh_template = _oh_on_pt2()
    blmin = closest_distances_generator([78, 8, 1], ratio_of_covalent_radii=0.7)
    ads_pos_before = combined.get_positions()[2:].copy()
    op = MirrorMutation(
        blmin,
        len(combined),
        system_type="gas_cluster_adsorbate",
        target_tags=[0],
        rng=default_rng(1),
        max_tries=24,
    )
    out = op.mutate(combined)
    assert out is not None
    assert np.allclose(out.get_positions()[2:], ads_pos_before)


def test_freeze_omits_overlap_relief_operator() -> None:
    comp = ["Pt", "Pt", "O", "H"]
    ads = {
        "core_symbols": ["Pt", "Pt"],
        "adsorbate_symbols": ["O", "H"],
        "adsorbate_fragment_lengths": [2],
    }
    tmpl = Atoms(symbols=comp, positions=np.zeros((4, 3)), pbc=False)
    blmin = closest_distances_generator(
        get_all_atom_types(tmpl, [0, 1, 2, 3]), ratio_of_covalent_radii=0.7
    )
    _ops, name_map = create_mutation_operators(
        composition=comp,
        n_to_optimize=4,
        blmin=blmin,
        rng=default_rng(2),
        use_adaptive=True,
        system_type="gas_cluster_adsorbate",
        adsorbate_definition=ads,
        freeze_adsorbate_internal_geometry=True,
    )
    assert "overlap_relief" not in name_map
    assert "flattening_ads" not in name_map
    assert "breathing_ads" not in name_map
