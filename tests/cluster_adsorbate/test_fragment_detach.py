"""Fragment detachment mutation: one intact adsorbate moves onto the slab."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111
from numpy.random import default_rng

import scgo.cluster_adsorbate.detach as detach_mod
from scgo.algorithms.ga_common import (
    apply_mobile_core_ads_tags,
    create_mutation_operators,
    unmatched_operator_weight_keys,
)
from scgo.ase_ga_patches.mutations._common import _resolve_op_connectivity_factor
from scgo.cluster_adsorbate.detach import FragmentDetachMutation
from scgo.initialization.atomic_radii import build_blmin_from_zs
from scgo.surface.validation import validate_supported_cluster_deposit
from scgo.system_types import AdsorbateDefinition
from scgo.utils.mutation_weights import (
    _apply_stagnation_boost,
    calculate_system_type_weights,
    get_adaptive_mutation_config,
)


def _ads() -> AdsorbateDefinition:
    return AdsorbateDefinition(
        core_symbols=["Pt", "Pt", "Pt"],
        adsorbate_symbols=["O", "H"],
        adsorbate_fragment_lengths=[2],
    )


def _oh_template() -> Atoms:
    return Atoms("OH", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.96]], pbc=False)


def _supported_oh(z_offset: float = 1.5) -> tuple[Atoms, int]:
    """Pt3 + OH on a small slab. ``z_offset`` is the O height above the cluster."""
    slab = fcc111("Pt", size=(6, 6, 2), vacuum=8.0, orthogonal=True)
    n_slab = len(slab)
    z_top = float(np.max(slab.positions[:, 2]))
    xy = np.mean(slab.positions[:, :2], axis=0)
    cluster_z = z_top + 2.2
    mobile = Atoms(
        symbols=["Pt", "Pt", "Pt", "O", "H"],
        positions=[
            [xy[0], xy[1], cluster_z],
            [xy[0] + 2.6, xy[1], cluster_z],
            [xy[0] + 1.3, xy[1] + 2.25, cluster_z],
            [xy[0] + 1.3, xy[1] + 0.75, cluster_z + z_offset],
            [xy[0] + 1.3, xy[1] + 0.75, cluster_z + z_offset + 0.96],
        ],
        cell=slab.get_cell(),
        pbc=slab.get_pbc(),
    )
    apply_mobile_core_ads_tags(mobile, n_slab=0, n_core=3, ads_fragment_lengths=[2])
    full = slab + mobile
    full.info["confid"] = 1
    return full, n_slab


def _operator(full: Atoms, n_slab: int, seed: int = 0) -> FragmentDetachMutation:
    return FragmentDetachMutation(
        build_blmin_from_zs(full.numbers, ratio=0.7),
        n_top=len(full) - n_slab,
        system_type="surface_cluster_adsorbate",
        adsorbate_definition=_ads(),
        fragment_templates=[_oh_template()],
        rng=default_rng(seed),
    )


def test_fragment_detach_moves_bound_fragment_onto_slab() -> None:
    full, n_slab = _supported_oh()
    template_bond = 0.96
    core_parent = full.positions[n_slab : n_slab + 3].copy()
    op = _operator(full, n_slab)
    for seed in range(40):
        op.rng = default_rng(seed)
        out = op.mutate(full)
        if out is None:
            continue
        np.testing.assert_allclose(
            out.positions[n_slab : n_slab + 3], core_parent, atol=1e-12
        )
        bond = float(
            np.linalg.norm(out.positions[n_slab + 4] - out.positions[n_slab + 3])
        )
        assert abs(bond - template_bond) < 1e-6
        ok, msg = validate_supported_cluster_deposit(
            out,
            n_slab,
            surface_normal_axis=2,
            use_mic=True,
            n_core_mobile=3,
            adsorbate_fragment_lengths=[2],
            allow_adsorbate_surface_detachment=True,
        )
        assert ok, msg
        rejected, _reject_msg = validate_supported_cluster_deposit(
            out,
            n_slab,
            surface_normal_axis=2,
            use_mic=True,
            n_core_mobile=3,
            adsorbate_fragment_lengths=[2],
            allow_adsorbate_surface_detachment=False,
        )
        assert not rejected
        op.rng = default_rng(seed)
        child, _desc = op.get_new_individual([full])
        assert child is not None
        return
    raise AssertionError("fragment_detach did not succeed within 40 seeds")


def test_fragment_detach_rejects_non_touching_and_clashing_placements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full, n_slab = _supported_oh()
    op = _operator(full, n_slab, seed=1)
    z_top = float(np.max(full.positions[:n_slab, 2]))
    xy = np.mean(full.positions[n_slab : n_slab + 3, :2], axis=0)

    def _place_at(height: float):
        def _fake(_core, fragment_template, _rng, _config, **_kwargs):
            placed = fragment_template.copy()
            placed.positions -= placed.positions[0]
            placed.positions += [xy[0] + 5.0, xy[1], z_top + height]
            return placed

        return _fake

    calls = {"n": 0}

    def _counting_high(*args, **kwargs):
        calls["n"] += 1
        return _place_at(15.0)(*args, **kwargs)

    monkeypatch.setattr(detach_mod, "place_fragment_on_cluster", _counting_high)
    assert op.mutate(full) is None
    assert calls["n"] > 0

    monkeypatch.setattr(detach_mod, "place_fragment_on_cluster", _place_at(0.05))
    assert op.mutate(full) is None


def test_fragment_detach_rejects_penetration_and_a_floating_core() -> None:
    """Below the slab top, or a core left off the slab, is not a child."""
    full, n_slab = _supported_oh()
    op = _operator(full, n_slab, seed=1)
    slab = full[:n_slab]
    z_top = float(np.max(slab.positions[:, 2]))
    surface = slab.positions
    top_layer = surface[:, 2] >= float(np.median(surface[:, 2]))
    edge = surface[top_layer][int(np.argmax(surface[top_layer, 0]))]
    oxygen_xy = edge[:2] + np.array([1.6, 0.0])
    connectivity = _resolve_op_connectivity_factor(op)
    core_z = float(full.positions[n_slab, 2])

    def _child(core_height: float, oxygen_z: float) -> Atoms:
        mobile = full[n_slab:].copy()
        mobile.positions[:3, 2] = core_height
        mobile.positions[3] = [oxygen_xy[0], oxygen_xy[1], oxygen_z]
        mobile.positions[4] = [oxygen_xy[0], oxygen_xy[1], oxygen_z + 0.96]
        return mobile

    def _accepts(mobile: Atoms) -> bool:
        return op._accepts(slab, mobile, 1, connectivity, True)

    assert _accepts(_child(core_z, z_top + 1.8))
    assert not _accepts(_child(core_z, z_top - 0.4))
    assert not _accepts(_child(z_top + 6.0, z_top + 1.8))


def test_fragment_detach_returns_none_when_fragment_is_already_off() -> None:
    full, n_slab = _supported_oh()
    full.positions[n_slab + 3] = [
        full.positions[n_slab, 0] + 8.0,
        full.positions[n_slab, 1],
        float(np.max(full.positions[:n_slab, 2])) + 1.8,
    ]
    full.positions[n_slab + 4] = full.positions[n_slab + 3] + [0.0, 0.0, 0.96]
    op = _operator(full, n_slab, seed=2)
    assert op.mutate(full) is None


def test_fragment_detach_is_unregistered_without_the_flag() -> None:
    comp = ["Pt", "Pt", "Pt", "O", "H"]
    ads = _ads()
    slab = fcc111("Pt", size=(2, 2, 2), vacuum=6.0, orthogonal=True)
    blmin = build_blmin_from_zs([78, 78, 8, 1], ratio=0.7)
    _ops, names = create_mutation_operators(
        comp,
        len(comp),
        blmin,
        rng=default_rng(0),
        system_type="surface_cluster_adsorbate",
        n_slab=len(slab),
        adsorbate_definition=ads,
        adsorbate_fragment_template=[_oh_template()],
        allow_adsorbate_surface_detachment=False,
    )
    assert "fragment_detach" not in names
    weights, _perm = calculate_system_type_weights(
        "surface_cluster_adsorbate",
        comp,
        adsorbate_definition=ads,
    )
    assert "fragment_detach" not in weights
    assert weights["fragment_reposition"] >= 0.18
    static = get_adaptive_mutation_config(
        comp,
        use_adaptive=False,
        system_type="surface_cluster_adsorbate",
        adsorbate_definition=ads,
    )
    assert "fragment_detach" not in static["operator_weights"]


def test_fragment_detach_is_registered_and_weighted_when_flag_is_on() -> None:
    comp = ["Pt", "Pt", "Pt", "O", "H"]
    ads = _ads()
    slab = fcc111("Pt", size=(2, 2, 2), vacuum=6.0, orthogonal=True)
    blmin = build_blmin_from_zs([78, 78, 8, 1], ratio=0.7)
    _ops, names = create_mutation_operators(
        comp,
        len(comp),
        blmin,
        rng=default_rng(0),
        system_type="surface_cluster_adsorbate",
        n_slab=len(slab),
        adsorbate_definition=ads,
        adsorbate_fragment_template=[_oh_template()],
        allow_adsorbate_surface_detachment=True,
    )
    assert "fragment_detach" in names
    for use_adaptive in (True, False):
        config = get_adaptive_mutation_config(
            comp,
            use_adaptive=use_adaptive,
            system_type="surface_cluster_adsorbate",
            adsorbate_definition=ads,
            allow_adsorbate_surface_detachment=True,
        )
        weights = config["operator_weights"]
        assert unmatched_operator_weight_keys(weights, names) == []
        assert weights["fragment_reposition"] > weights["fragment_detach"] > 0.0

    _gas_ops, gas_names = create_mutation_operators(
        comp,
        len(comp),
        blmin,
        rng=default_rng(1),
        system_type="gas_cluster_adsorbate",
        adsorbate_definition=ads,
        adsorbate_fragment_template=[_oh_template()],
        allow_adsorbate_surface_detachment=True,
    )
    assert "fragment_detach" not in gas_names


def test_stagnation_boosts_fragment_detach() -> None:
    base = {"rattle": 0.5, "fragment_detach": 0.5}
    boosted = _apply_stagnation_boost(base, level=1.0, burst_multiplier=1.8)
    assert boosted["fragment_detach"] > base["fragment_detach"]
