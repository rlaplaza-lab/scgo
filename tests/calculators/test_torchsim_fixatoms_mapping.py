"""Tests for ASE FixAtoms -> TorchSim FixAtoms batch mapping."""

from __future__ import annotations

import logging

import pytest
from ase import Atoms
from ase.constraints import FixAtoms as ASEFixAtoms

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


def test_unsupported_constraint_warns_once(caplog, monkeypatch) -> None:
    """K8: dropped (non-FixAtoms) constraints must be reported once per process."""
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
    assert len(warnings_seen) == 1


def test_fixatoms_still_collected_alongside_dropped_constraints(monkeypatch) -> None:
    """A dropped constraint must not hide the FixAtoms indices."""
    from ase.constraints import FixBondLengths

    from scgo.calculators import torchsim_helpers as th

    monkeypatch.setattr(th, "_WARNED_DROPPED_CONSTRAINTS", set())

    a = Atoms("H3", positions=[[0, 0, 0], [0, 0, 0.74], [0, 0, 1.5]])
    a.set_constraint([ASEFixAtoms(indices=[-1]), FixBondLengths([(0, 1)])])
    assert collect_ase_fixatoms_indices(a) == [2]
