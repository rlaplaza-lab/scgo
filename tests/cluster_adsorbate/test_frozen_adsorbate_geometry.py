"""Frozen adsorbate internal geometry: restore, operators, and constraints."""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms, FixBondLengths
from ase.optimize import LBFGS
from ase_ga.utilities import closest_distances_generator, get_all_atom_types
from numpy.random import default_rng

from scgo.algorithms.ga_common import create_mutation_operators
from scgo.ase_ga_patches.mutations import MirrorMutation, OverlapReliefMutation
from scgo.cluster_adsorbate.constraints import attach_fix_bond_lengths
from scgo.cluster_adsorbate.rigid import (
    enforce_frozen_adsorbate_geometry,
    restore_rigid_adsorbate_fragments,
)
from scgo.system_types import AdsorbateDefinition


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
    ads_def = AdsorbateDefinition(
        core_symbols=["Pt", "Pt"],
        adsorbate_symbols=["O", "H"],
        adsorbate_fragment_lengths=[2],
    )
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


def _slab_core_oh() -> tuple[Atoms, Atoms, int, dict]:
    """Slab (4 Pt) + mobile Pt2 core + rigid OH, plus the adsorbate definition."""
    slab = Atoms(
        "Pt4",
        positions=[
            [0.0, 0.0, 0.0],
            [2.8, 0.0, 0.0],
            [0.0, 2.8, 0.0],
            [2.8, 2.8, 0.0],
        ],
        cell=[5.6, 5.6, 20.0],
        pbc=[True, True, False],
    )
    core = Atoms("Pt2", positions=[[1.4, 1.4, 2.3], [3.8, 1.4, 2.3]])
    oh_template = Atoms("OH", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.96]])
    oh = Atoms("OH", positions=[[2.6, 1.4, 4.0], [2.6, 1.4, 4.96]])
    combined = slab + core + oh
    combined.set_cell(slab.get_cell())
    combined.set_pbc(slab.get_pbc())
    ads_def = AdsorbateDefinition(
        core_symbols=["Pt", "Pt"],
        adsorbate_symbols=["O", "H"],
        adsorbate_fragment_lengths=[2],
    )
    return combined, oh_template, len(slab), ads_def


def test_enforce_frozen_adsorbate_geometry_keeps_slab_fixatoms() -> None:
    """Re-attaching adsorbate constraints must not wipe the slab ``FixAtoms``."""
    combined, oh_template, n_slab, ads_def = _slab_core_oh()
    combined.set_constraint(FixAtoms(indices=list(range(n_slab))))

    enforce_frozen_adsorbate_geometry(
        combined,
        n_slab=n_slab,
        adsorbate_definition=ads_def,
        fragment_templates=[oh_template],
        reattach_constraints=True,
    )

    fixatoms = [c for c in combined.constraints if isinstance(c, FixAtoms)]
    assert len(fixatoms) == 1
    assert {int(i) for i in fixatoms[0].index} == set(range(n_slab))
    bonds = [c for c in combined.constraints if isinstance(c, FixBondLengths)]
    assert len(bonds) == 1
    pairs = {tuple(sorted(int(x) for x in pair)) for pair in bonds[0].pairs}
    assert pairs == {(6, 7)}


def test_enforce_frozen_adsorbate_geometry_does_not_duplicate_bond_constraints() -> (
    None
):
    """Repeated re-attachment keeps exactly one bond-length constraint."""
    combined, oh_template, n_slab, ads_def = _slab_core_oh()
    combined.set_constraint(FixAtoms(indices=list(range(n_slab))))
    for _ in range(3):
        enforce_frozen_adsorbate_geometry(
            combined,
            n_slab=n_slab,
            adsorbate_definition=ads_def,
            fragment_templates=[oh_template],
            reattach_constraints=True,
        )
    assert len([c for c in combined.constraints if isinstance(c, FixAtoms)]) == 1
    assert len([c for c in combined.constraints if isinstance(c, FixBondLengths)]) == 1


def test_rigid_three_atom_fragment_keeps_all_internal_distances() -> None:
    """A single multi-pair constraint holds a 3-atom fragment rigid during relaxation."""
    core = Atoms(
        "Pt3",
        positions=[[0.0, 0.0, 0.0], [2.6, 0.0, 0.0], [1.3, 2.25, 0.0]],
        cell=[20.0, 20.0, 20.0],
        pbc=False,
    )
    water = Atoms(
        "OH2",
        positions=[
            [1.3, 0.75, 1.9],
            [1.83, 1.53, 2.15],
            [0.55, 0.95, 2.47],
        ],
    )
    combined = core + water
    combined.set_cell(core.get_cell())
    combined.set_pbc(False)
    frag_pairs = [(3, 4), (3, 5), (4, 5)]
    reference = {pair: combined.get_distance(*pair) for pair in frag_pairs}

    attach_fix_bond_lengths(combined, frag_pairs)
    combined.set_constraint(
        [*combined.constraints, FixAtoms(indices=list(range(len(core))))]
    )
    combined.calc = EMT()
    LBFGS(combined, logfile=None).run(fmax=0.01, steps=8)

    for pair in frag_pairs:
        assert abs(combined.get_distance(*pair) - reference[pair]) < 1e-6


def test_freeze_omits_overlap_relief_operator() -> None:
    comp = ["Pt", "Pt", "O", "H"]
    ads = AdsorbateDefinition(
        core_symbols=["Pt", "Pt"],
        adsorbate_symbols=["O", "H"],
        adsorbate_fragment_lengths=[2],
    )
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
