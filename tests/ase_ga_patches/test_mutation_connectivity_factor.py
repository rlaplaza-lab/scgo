"""Mutation connectivity gate must honor the run-resolved factor."""

from __future__ import annotations

import numpy as np
from ase import Atoms
from numpy.random import default_rng

from scgo.algorithms.ga_common import create_mutation_operators
from scgo.ase_ga_patches.mutations._common import _mobile_is_connected
from scgo.cluster_adsorbate import reposition as reposition_mod
from scgo.cluster_adsorbate.reposition import FragmentRepositionMutation
from scgo.initialization.atomic_radii import build_blmin_from_zs
from scgo.system_types import AdsorbateDefinition


def test_mutation_operators_use_run_connectivity_factor() -> None:
    blmin = build_blmin_from_zs([78, 79], ratio=0.7)
    ops, _ = create_mutation_operators(
        composition=["Pt", "Au", "Pt"],
        n_to_optimize=3,
        blmin=blmin,
        rng=np.random.default_rng(0),
        connectivity_factor=1.1,
    )
    assert ops and all(op.connectivity_factor == 1.1 for op in ops)

    # Pt–Pt at 3.5 Å: connected at default 1.4 (~3.81 Å), not at 1.1 (~2.99 Å).
    dimer = Atoms("Pt2", positions=[[0.0, 0.0, 0.0], [3.5, 0.0, 0.0]])
    assert _mobile_is_connected(dimer, use_mic=False, connectivity_factor=1.4)
    assert not _mobile_is_connected(dimer, use_mic=False, connectivity_factor=1.1)


def test_fragment_reposition_uses_run_connectivity_factor(monkeypatch) -> None:
    comp = ["Pt", "Pt", "Pt", "O", "H"]
    ads = AdsorbateDefinition(
        core_symbols=["Pt", "Pt", "Pt"],
        adsorbate_symbols=["O", "H"],
        adsorbate_fragment_lengths=[2],
    )
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
        connectivity_factor=1.8,
    )
    assert ops[name_map["fragment_reposition"]].connectivity_factor == 1.8

    combined = Atoms(
        symbols=comp,
        positions=[
            [0.0, 0.0, 0.0],
            [2.5, 0.0, 0.0],
            [1.25, 2.165, 0.0],
            [1.25, 1.0, 2.0],
            [1.25, 1.0, 2.96],
        ],
        pbc=False,
    )
    combined.set_tags([0, 0, 0, 1, 1])
    oh = Atoms("OH", positions=[[1.25, 1.0, 3.5], [1.25, 1.0, 4.46]], pbc=False)
    factor = {"Pt-C": 1.8}
    op = FragmentRepositionMutation(
        blmin,
        len(combined),
        system_type="gas_cluster_adsorbate",
        adsorbate_definition=ads,
        fragment_templates=[oh],
        rng=default_rng(0),
    )
    op.connectivity_factor = factor
    seen: dict[str, object] = {}

    def _spy_preserves(parent, mutant, **kwargs):
        seen.update(kwargs)
        return True

    monkeypatch.setattr(
        reposition_mod,
        "_preserves_mobile_connectivity",
        _spy_preserves,
    )
    monkeypatch.setattr(
        reposition_mod,
        "place_fragment_on_cluster",
        lambda *args, **kwargs: oh.copy(),
    )
    assert op.mutate(combined) is not None
    assert seen.get("connectivity_factor") == factor
