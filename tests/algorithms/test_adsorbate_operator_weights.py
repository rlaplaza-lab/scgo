"""Partitioned adsorbate GA operators receive non-zero selection weights."""

from __future__ import annotations

import numpy as np
from ase import Atoms
from numpy.random import default_rng

from scgo.algorithms.ga_common import (
    _effective_operator_weight,
    create_mutation_operators,
    update_mutation_weights,
)
from scgo.initialization.atomic_radii import build_blmin_from_zs
from scgo.utils.mutation_weights import (
    _apply_stagnation_boost,
    get_adaptive_mutation_config,
)


def test_partitioned_operators_inherit_base_weights() -> None:
    name_map = {
        "rattle": 0,
        "flattening_core": 1,
        "flattening_ads": 2,
        "in_plane_slide_core": 3,
        "in_plane_slide_ads": 4,
        "fragment_reposition": 5,
    }
    base = {"flattening": 0.2, "in_plane_slide": 0.1, "rotational": 0.15}
    assert _effective_operator_weight("flattening_core", base, name_map) > 0.0
    assert _effective_operator_weight("in_plane_slide_ads", base, name_map) > 0.0
    assert _effective_operator_weight("fragment_reposition", base, name_map) > 0.0
    assert (
        _effective_operator_weight(
            "fragment_reposition",
            base,
            {**name_map, "rotational": 4},
        )
        > 0.0
    )


def test_adsorbate_operator_selector_assigns_weight_to_partitioned_ops() -> None:
    comp = ["Pt", "Pt", "Pt", "O", "H"]
    ads = {
        "core_symbols": ["Pt", "Pt", "Pt"],
        "adsorbate_symbols": ["O", "H"],
        "adsorbate_fragment_lengths": [2],
    }
    tmpl = Atoms(symbols=comp, positions=np.zeros((5, 3)), pbc=False)
    blmin = build_blmin_from_zs(tmpl.numbers, ratio=0.7)
    ops, name_map = create_mutation_operators(
        composition=comp,
        n_to_optimize=5,
        blmin=blmin,
        rng=default_rng(0),
        use_adaptive=True,
        system_type="gas_cluster_adsorbate",
        adsorbate_definition=ads,
        adsorbate_fragment_template=[tmpl[-2:]],
    )
    adaptive = get_adaptive_mutation_config(comp, use_adaptive=True)
    selector = update_mutation_weights(ops, name_map, adaptive, rng=default_rng(0))
    assert selector.rho[name_map["flattening_core"]] > 0.0
    assert selector.rho[name_map["fragment_reposition"]] > 0.0


def test_rotational_operator_targets_core_only_for_partition() -> None:
    comp = ["Pt", "Pt", "O", "H"]
    ads = {
        "core_symbols": ["Pt", "Pt"],
        "adsorbate_symbols": ["O", "H"],
        "adsorbate_fragment_lengths": [2],
    }
    tmpl = Atoms(symbols=comp, positions=np.zeros((4, 3)), pbc=False)
    blmin = build_blmin_from_zs(tmpl.numbers, ratio=0.7)
    ops, name_map = create_mutation_operators(
        composition=comp,
        n_to_optimize=4,
        blmin=blmin,
        rng=default_rng(2),
        use_adaptive=True,
        system_type="gas_cluster_adsorbate",
        adsorbate_definition=ads,
        adsorbate_fragment_template=[tmpl[-2:]],
    )
    rot = ops[name_map["rotational"]]
    assert rot.target_tags == [0]
    assert rot.use_tags is True


def test_stagnation_boost_propagates_to_partitioned_flattening_weight() -> None:
    name_map = {
        "rattle": 0,
        "flattening_core": 1,
        "flattening_ads": 2,
        "rotational": 3,
        "fragment_reposition": 4,
    }
    base_weights = {
        "rattle": 0.2,
        "flattening": 0.2,
        "rotational": 0.15,
        "anisotropic_rattle": 0.14,
    }
    boosted = _apply_stagnation_boost(base_weights, 1.0, 1.8)
    w_base = _effective_operator_weight("flattening_core", base_weights, name_map)
    w_boost = _effective_operator_weight("flattening_core", boosted, name_map)
    assert w_boost > w_base
    assert boosted["rotational"] > base_weights["rotational"]
