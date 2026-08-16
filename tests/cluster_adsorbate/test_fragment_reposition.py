"""Fragment reposition mutation preserves internal geometry."""

from __future__ import annotations

import numpy as np
from ase import Atoms
from numpy.random import default_rng

import scgo.cluster_adsorbate.reposition as reposition_mod
from scgo.algorithms.ga_common import apply_mobile_core_ads_tags
from scgo.cluster_adsorbate.config import ClusterAdsorbateConfig
from scgo.cluster_adsorbate.reposition import FragmentRepositionMutation
from scgo.initialization.atomic_radii import build_blmin_from_zs
from scgo.system_types import AdsorbateDefinition


def _pt3_oh_system() -> tuple[Atoms, Atoms]:
    core = Atoms(
        "Pt3",
        positions=[
            [0.0, 0.0, 0.0],
            [2.5, 0.0, 0.0],
            [1.25, 2.165, 0.0],
        ],
        pbc=False,
    )
    oh = Atoms("OH", positions=[[1.25, 1.0, 2.0], [1.25, 1.0, 2.96]], pbc=False)
    combined = core + oh
    combined.set_cell([20, 20, 20])
    combined.set_pbc(False)
    apply_mobile_core_ads_tags(combined, n_slab=0, n_core=3, ads_fragment_lengths=[2])
    return combined, oh


def test_fragment_reposition_preserves_bond_length() -> None:
    combined, oh_template = _pt3_oh_system()
    template_bond = float(
        np.linalg.norm(oh_template.positions[1] - oh_template.positions[0])
    )
    ads_def = AdsorbateDefinition(
        core_symbols=["Pt", "Pt", "Pt"],
        adsorbate_symbols=["O", "H"],
        adsorbate_fragment_lengths=[2],
    )
    blmin = build_blmin_from_zs(combined.numbers, ratio=0.7)
    op = FragmentRepositionMutation(
        blmin,
        len(combined),
        system_type="gas_cluster_adsorbate",
        adsorbate_definition=ads_def,
        fragment_templates=[oh_template],
        rng=default_rng(3),
    )
    for seed in range(30):
        op.rng = default_rng(seed)
        out = op.mutate(combined)
        if out is None:
            continue
        o_pos = out.positions[3]
        h_pos = out.positions[4]
        bond = float(np.linalg.norm(h_pos - o_pos))
        assert abs(bond - template_bond) < 1e-6
        assert not np.allclose(out.positions, combined.positions)
        return
    raise AssertionError("fragment_reposition did not succeed within 30 seeds")


def test_fragment_reposition_changes_relative_pose(monkeypatch) -> None:
    combined, oh_template = _pt3_oh_system()
    ads_def = AdsorbateDefinition(
        core_symbols=["Pt", "Pt", "Pt"],
        adsorbate_symbols=["O", "H"],
        adsorbate_fragment_lengths=[2],
    )
    blmin = build_blmin_from_zs(combined.numbers, ratio=0.7)
    op = FragmentRepositionMutation(
        blmin,
        len(combined),
        system_type="gas_cluster_adsorbate",
        adsorbate_definition=ads_def,
        fragment_templates=[oh_template],
        rng=default_rng(11),
    )
    out = op.mutate(combined)
    if out is not None:
        assert not np.allclose(out.positions[3:5], combined.positions[3:5])

    # Failed mutate uses at most two place_fragment calls with a capped budget.
    calls: list[int] = []

    def _fake_place(*args, **kwargs):
        cfg = kwargs.get("config", args[3] if len(args) > 3 else None)
        calls.append(int(cfg.max_placement_attempts))
        return None

    monkeypatch.setattr(reposition_mod, "place_fragment_on_cluster", _fake_place)
    ca = ClusterAdsorbateConfig(max_placement_attempts=80)
    fail_op = FragmentRepositionMutation(
        blmin,
        len(combined),
        system_type="gas_cluster_adsorbate",
        adsorbate_definition=ads_def,
        fragment_templates=[oh_template],
        cluster_adsorbate_config=ca,
        rng=default_rng(0),
    )
    assert fail_op.mutate(combined) is None
    assert calls == [80, 240]
