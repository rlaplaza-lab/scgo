"""Tests for ASE FixAtoms -> TorchSim FixAtoms batch mapping."""

from __future__ import annotations

import logging

import pytest
from ase import Atoms
from ase.constraints import FixAtoms as ASEFixAtoms

from scgo.calculators.torchsim_constraints import (
    build_torchsim_fixbondlengths_from_ase_batch,
    collect_ase_fixbondlengths,
)
from scgo.calculators.torchsim_helpers import (
    build_torchsim_fixatoms_from_ase_batch,
    collect_ase_fixatoms_indices,
)


def test_collect_ase_fixatoms_indices_empty() -> None:
    a = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    assert collect_ase_fixatoms_indices(a) == []


def test_collect_ase_fixatoms_indices_one_constraint() -> None:
    a = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    a.set_constraint(ASEFixAtoms(indices=[0]))
    assert collect_ase_fixatoms_indices(a) == [0]


def test_build_torchsim_fixatoms_batch_global_indices() -> None:
    torch = pytest.importorskip("torch")

    s1 = Atoms("Cu2", positions=[[0, 0, 0], [1.8, 0, 0]])
    s1.set_constraint(ASEFixAtoms(indices=[0]))
    s2 = Atoms("Cu2", positions=[[0, 0, 0], [1.8, 0, 0]])
    s2.set_constraint(ASEFixAtoms(indices=[1]))

    c = build_torchsim_fixatoms_from_ase_batch([s1, s2], device=torch.device("cpu"))
    assert c is not None
    assert c.atom_idx.tolist() == [0, 3]


def test_build_torchsim_fixatoms_batch_none_when_unconstrained() -> None:
    torch = pytest.importorskip("torch")

    a = Atoms("H", positions=[[0, 0, 0]])
    assert (
        build_torchsim_fixatoms_from_ase_batch([a], device=torch.device("cpu")) is None
    )


def test_collect_ase_fixatoms_indices_normalizes_negative_index() -> None:
    """K8: ASE allows negative indices; TorchSim needs positive ones."""
    a = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    a.set_constraint(ASEFixAtoms(indices=[-1]))
    assert collect_ase_fixatoms_indices(a) == [1]


def test_build_torchsim_fixatoms_batch_handles_negative_indices() -> None:
    """K8: negative indices must map to the right *global* batch index."""
    torch = pytest.importorskip("torch")

    s1 = Atoms("Cu2", positions=[[0, 0, 0], [1.8, 0, 0]])
    s1.set_constraint(ASEFixAtoms(indices=[-1]))  # -> local 1, global 1
    s2 = Atoms("Cu2", positions=[[0, 0, 0], [1.8, 0, 0]])
    s2.set_constraint(ASEFixAtoms(indices=[-2]))  # -> local 0, global 2

    c = build_torchsim_fixatoms_from_ase_batch([s1, s2], device=torch.device("cpu"))
    assert c is not None
    assert c.atom_idx.tolist() == [1, 2]


def test_fixbondlengths_no_longer_dropped(caplog, monkeypatch) -> None:
    """P3: FixBondLengths is now mapped to a TorchSim constraint (not dropped)."""
    from ase.constraints import FixBondLengths

    from scgo.calculators import torchsim_helpers as th

    monkeypatch.setattr(th, "_WARNED_DROPPED_CONSTRAINTS", set())

    a = Atoms("H3", positions=[[0, 0, 0], [0, 0, 0.74], [0, 0, 1.5]])
    a.set_constraint(FixBondLengths([(0, 1)]))

    with caplog.at_level(logging.WARNING, logger="scgo.calculators.torchsim_helpers"):
        assert collect_ase_fixatoms_indices(a) == []
        assert collect_ase_fixatoms_indices(a) == []

    warnings_seen = [
        rec for rec in caplog.records if "FixBondLengths" in rec.getMessage()
    ]
    assert warnings_seen == []


def test_unsupported_constraint_warns_once(caplog, monkeypatch) -> None:
    """K8: dropped (non-FixAtoms/non-FixBondLengths) constraints reported once."""
    from ase.constraints import FixCartesian

    from scgo.calculators import torchsim_helpers as th

    monkeypatch.setattr(th, "_WARNED_DROPPED_CONSTRAINTS", set())

    a = Atoms("H3", positions=[[0, 0, 0], [0, 0, 0.74], [0, 0, 1.5]])
    a.set_constraint(FixCartesian([0, 1, 2], mask=(True, True, True)))

    with caplog.at_level(logging.WARNING, logger="scgo.calculators.torchsim_helpers"):
        assert collect_ase_fixatoms_indices(a) == []
        assert collect_ase_fixatoms_indices(a) == []

    warnings_seen = [
        rec for rec in caplog.records if "FixCartesian" in rec.getMessage()
    ]
    assert len(warnings_seen) == 1


def test_fixatoms_still_collected_alongside_dropped_constraints(monkeypatch) -> None:
    """A dropped constraint must not hide the FixAtoms indices."""
    from ase.constraints import FixBondLengths

    from scgo.calculators import torchsim_helpers as th

    monkeypatch.setattr(th, "_WARNED_DROPPED_CONSTRAINTS", set())

    a = Atoms("H3", positions=[[0, 0, 0], [0, 0, 0.74], [0, 0, 1.5]])
    a.set_constraint([ASEFixAtoms(indices=[-1]), FixBondLengths([(0, 1)])])
    assert collect_ase_fixatoms_indices(a) == [2]


def test_collect_ase_fixbondlengths_measures_target_lengths() -> None:
    """P3: target bond lengths are read from the initial geometry."""
    from ase.constraints import FixBondLengths

    a = Atoms("H3", positions=[[0, 0, 0], [0, 0, 0.74], [0, 0, 1.5]])
    a.set_constraint(FixBondLengths([(0, 1), (1, 2)]))

    collected = collect_ase_fixbondlengths(a)
    assert len(collected) == 1
    pairs, lengths, sys_id = collected[0]
    assert sys_id == 0
    assert pairs.tolist() == [[0, 1], [1, 2]]
    assert lengths.tolist() == pytest.approx([0.74, 0.76], abs=1e-6)


def test_collect_ase_fixbondlengths_with_offset() -> None:
    """P3: batch offsets shift atom indices to global coordinates."""
    from ase.constraints import FixBondLengths

    a = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.9]])
    a.set_constraint(FixBondLengths([(0, 1)]))

    pairs, _lengths, _sys = collect_ase_fixbondlengths(a, offset=4)[0]
    assert pairs.tolist() == [[4, 5]]


def test_build_torchsim_fixbondlengths_batch_global_indices() -> None:
    """P3: per-structure bonds are merged into one global TorchSim constraint."""
    torch = pytest.importorskip("torch")
    from ase.constraints import FixBondLengths

    from scgo.calculators.torchsim_constraints import TorchSimFixBondLengths

    s1 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.8]])
    s1.set_constraint(FixBondLengths([(0, 1)]))
    s2 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.2]])
    s2.set_constraint(FixBondLengths([(0, 1)]))

    c = build_torchsim_fixbondlengths_from_ase_batch(
        [s1, s2], device=torch.device("cpu")
    )
    assert isinstance(c, TorchSimFixBondLengths)
    assert c.pairs.tolist() == [[0, 1], [2, 3]]
    assert c.system_idx.tolist() == [0, 1]
    assert c.bond_lengths.tolist() == pytest.approx([0.8, 1.2], abs=1e-6)


def test_build_torchsim_fixbondlengths_none_when_unconstrained() -> None:
    """P3: no FixBondLengths -> no TorchSim bond constraint built."""
    torch = pytest.importorskip("torch")

    a = Atoms("H", positions=[[0, 0, 0]])
    assert (
        build_torchsim_fixbondlengths_from_ase_batch([a], device=torch.device("cpu"))
        is None
    )
