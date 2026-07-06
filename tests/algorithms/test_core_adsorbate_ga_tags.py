"""Core vs adsorbate partition: tags and operator wiring for two-block mobile GA."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase_ga.utilities import closest_distances_generator, get_all_atom_types

from scgo.algorithms.ga_common import (
    apply_mobile_core_ads_tags,
    core_adsorbate_partition_counts,
    core_adsorbate_partition_details,
    create_ga_pairing,
    create_mutation_operators,
    maybe_apply_mobile_core_ads_tags,
)
from scgo.ase_ga_patches.cutandsplicepairing import CutAndSplicePairing
from scgo.system_types import AdsorbateDefinition


def test_apply_mobile_core_ads_tags() -> None:
    a = Atoms(
        symbols=["Pt", "Pt", "Pt", "O", "H", "O", "H"],
        positions=np.zeros((7, 3)),
        pbc=False,
    )
    apply_mobile_core_ads_tags(a, n_slab=2, n_core=1, ads_fragment_lengths=[2, 2])
    assert list(a.get_tags()) == [0, 0, 0, 1, 1, 2, 2]


@pytest.mark.parametrize(
    ("ads", "composition", "expected"),
    [
        (
            {"core_symbols": ["Pt", "Pt"], "adsorbate_symbols": ["O", "H"]},
            ["Pt", "Pt", "O", "H"],
            (2, 2),
        ),
        (
            {"core_symbols": ["Pt", "Pt", "O", "H"], "adsorbate_symbols": []},
            ["Pt", "Pt", "O", "H"],
            None,
        ),
    ],
)
def test_core_adsorbate_partition_counts(
    ads: AdsorbateDefinition,
    composition: list[str],
    expected: tuple[int, int] | None,
) -> None:
    assert (
        core_adsorbate_partition_counts("gas_cluster_adsorbate", composition, ads)
        == expected
    )


def test_core_adsorbate_partition_details_fragment_lengths() -> None:
    ads: AdsorbateDefinition = {
        "core_symbols": ["Pt", "Pt"],
        "adsorbate_symbols": ["O", "H", "O", "H"],
        "adsorbate_fragment_lengths": [2, 2],
    }
    details = core_adsorbate_partition_details(
        "gas_cluster_adsorbate",
        ["Pt", "Pt", "O", "H", "O", "H"],
        ads,
    )
    assert details == (2, [2, 2])


def test_create_ga_pairing_use_tags_for_two_block() -> None:
    comp = ["Pt", "Pt", "O", "H"]
    ads: AdsorbateDefinition = {
        "core_symbols": ["Pt", "Pt"],
        "adsorbate_symbols": ["O", "H"],
    }
    at = Atoms(symbols=comp, positions=np.zeros((4, 3)), cell=[20, 20, 20], pbc=False)
    p = create_ga_pairing(
        at,
        4,
        np.random.default_rng(0),
        system_type="gas_cluster_adsorbate",
        composition=comp,
        adsorbate_definition=ads,
        exploratory_crossover_probability=0.0,
    )
    assert isinstance(p, CutAndSplicePairing)
    assert p.use_tags is True


def test_create_mutation_operators_two_block_tags_omit_distort() -> None:
    comp = ["Pt", "Pt", "O", "H"]
    ads: AdsorbateDefinition = {
        "core_symbols": ["Pt", "Pt"],
        "adsorbate_symbols": ["O", "H"],
        "adsorbate_fragment_lengths": [2],
    }
    tmpl = Atoms(symbols=comp, positions=np.zeros((4, 3)), pbc=False)
    blmin = closest_distances_generator(
        get_all_atom_types(tmpl, [0, 1, 2, 3]), ratio_of_covalent_radii=0.7
    )
    ops, name_map = create_mutation_operators(
        composition=comp,
        n_to_optimize=4,
        blmin=blmin,
        rng=np.random.default_rng(0),
        use_adaptive=True,
        system_type="gas_cluster_adsorbate",
        adsorbate_definition=ads,
    )
    assert "flattening" not in name_map and "breathing" not in name_map
    assert ops[name_map["rattle"]].use_tags is True
    assert ops[name_map["anisotropic_rattle"]].use_tags is True
    assert ops[name_map["rotational"]].target_tags == [0]


def test_create_mutation_operators_freeze_omits_ads_distort_ops() -> None:
    comp = ["Pt", "Pt", "O", "H", "O", "H"]
    ads: AdsorbateDefinition = {
        "core_symbols": ["Pt", "Pt"],
        "adsorbate_symbols": ["O", "H", "O", "H"],
        "adsorbate_fragment_lengths": [2, 2],
    }
    tmpl = Atoms(symbols=comp, positions=np.zeros((6, 3)), pbc=False)
    blmin = closest_distances_generator(
        get_all_atom_types(tmpl, [0, 1, 2, 3, 4, 5]), ratio_of_covalent_radii=0.7
    )
    _ops, name_map = create_mutation_operators(
        composition=comp,
        n_to_optimize=6,
        blmin=blmin,
        rng=np.random.default_rng(1),
        use_adaptive=True,
        system_type="gas_cluster_adsorbate",
        adsorbate_definition=ads,
        freeze_adsorbate_internal_geometry=True,
    )
    assert "flattening_ads" not in name_map
    assert "breathing_ads" not in name_map


def test_maybe_apply_skips_monolithic_ads_def() -> None:
    a = Atoms("H2", [[0, 0, 0], [0, 0, 0.78]], pbc=False)
    maybe_apply_mobile_core_ads_tags(
        a,
        0,
        ["H", "H"],
        {"core_symbols": ["H", "H"], "adsorbate_symbols": []},
        "gas_cluster_adsorbate",
    )
    assert np.all(a.get_tags() == 0)
