"""Tests for ASE FixAtoms -> TorchSim FixAtoms batch mapping."""

from __future__ import annotations

import logging

import numpy as np
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


def test_select_constraint_packs_pairs_when_first_system_dropped(
    monkeypatch,
) -> None:
    """InFlight pop of system 0 must renumber remaining bonds to local [0, 1].

    Also guards the adsorbate CUDA crash: ``select_constraint`` / ``Constraint.to``
    reconstruct via ``__init__`` with device tensors; that must not call
    ``np.asarray`` (CUDA tensors raise ``TypeError`` there).
    """
    torch = pytest.importorskip("torch")

    from scgo.calculators.torchsim_constraints import TorchSimFixBondLengths

    _patch_asarray_reject_tensors(monkeypatch, torch)

    constraint = TorchSimFixBondLengths(
        [[0, 1], [2, 3]],
        [0.8, 1.2],
        [0, 1],
        device=torch.device("cpu"),
    )
    # ``to`` is what torch-sim ``initialize_state`` / ``SimState.to`` invoke.
    constraint = constraint.to(device=torch.device("cpu"), dtype=torch.float64)
    packed = constraint.select_constraint(
        torch.tensor([False, False, True, True]),
        torch.tensor([False, True]),
    )
    assert packed is not None
    assert packed.pairs.tolist() == [[0, 1]]
    assert packed.system_idx.tolist() == [0]
    assert packed.bond_lengths.tolist() == pytest.approx([1.2], abs=1e-6)


def _patch_asarray_reject_tensors(
    monkeypatch: pytest.MonkeyPatch, torch_mod: object
) -> None:
    """Make ``np.asarray(tensor)`` fail like CUDA ``Tensor.numpy()`` does."""
    real_asarray = np.asarray

    def _asarray_guard(a, *args, **kwargs):
        if isinstance(a, torch_mod.Tensor):
            raise TypeError(
                "can't convert cuda:0 device type tensor to numpy. "
                "Use Tensor.cpu() to copy the tensor to host memory first."
            )
        return real_asarray(a, *args, **kwargs)

    monkeypatch.setattr(
        "scgo.calculators.torchsim_constraints.np.asarray",
        _asarray_guard,
    )


def _run_fixbondlengths_simstate_pop(*, device: object) -> None:
    """Shared SimState attach → to(device) → pop → enforce bond length."""
    torch = pytest.importorskip("torch")
    torch_sim = pytest.importorskip("torch_sim")
    from ase.constraints import FixBondLengths

    from scgo.calculators.torchsim_constraints import (
        TorchSimFixBondLengths,
        build_torchsim_fixbondlengths_from_ase_batch,
    )

    s0 = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.8]], cell=[20, 20, 20])
    s0.set_constraint(FixBondLengths([(0, 1)]))
    s1 = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.2]], cell=[20, 20, 20])
    s1.set_constraint(FixBondLengths([(0, 1)]))

    # Match TorchSimBatchRelaxer: initialize on device, attach SCGO bond
    # constraints, then let torch-sim move/clone the state (Constraint.to).
    state = torch_sim.initialize_state([s0, s1], device, torch.float64)
    built = build_torchsim_fixbondlengths_from_ase_batch([s0, s1], device=device)
    assert built is not None
    state.constraints = [built]
    state = state.to(device, torch.float64)

    state.pop(0)
    assert state.n_systems == 1
    assert state.n_atoms == 2
    assert len(state.constraints) == 1
    remaining = state.constraints[0]
    assert isinstance(remaining, TorchSimFixBondLengths)
    assert remaining.pairs.device == torch.device(device)
    assert remaining.pairs.tolist() == [[0, 1]]
    assert remaining.system_idx.tolist() == [0]

    stretched = state.positions.clone()
    stretched[1, 2] = 5.0
    state.set_constrained_positions(stretched)
    dist = torch.linalg.norm(state.positions[1] - state.positions[0]).item()
    assert abs(dist - 1.2) <= 1e-5


def test_fixbondlengths_survives_simstate_pop_of_system_zero(monkeypatch) -> None:
    """torch-sim InFlight pop must leave a valid packed bond constraint.

    CPU CI cannot exercise real CUDA tensors, so ``np.asarray`` is patched to
    reject tensors the way CUDA ``Tensor.numpy()`` does — covering the ORR /
    adsorbate ``initialize_state`` → ``Constraint.to(cuda)`` failure mode.
    """
    torch = pytest.importorskip("torch")
    _patch_asarray_reject_tensors(monkeypatch, torch)
    _run_fixbondlengths_simstate_pop(device=torch.device("cpu"))


@pytest.mark.requires_cuda
def test_fixbondlengths_survives_simstate_to_cuda_and_pop() -> None:
    """Same SimState path on a real CUDA device (user's ORR traceback)."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    _run_fixbondlengths_simstate_pop(device=torch.device("cuda"))


def test_fixbondlengths_adjust_positions_uses_minimum_image() -> None:
    """Periodic wrap: restore the MIC bond length, not the unwrapped 9+ A image."""
    torch = pytest.importorskip("torch")
    torch_sim = pytest.importorskip("torch_sim")
    from ase.constraints import FixBondLengths
    from torch_sim.transforms import minimum_image_displacement

    from scgo.calculators.torchsim_constraints import (
        build_torchsim_fixbondlengths_from_ase_batch,
    )

    atoms = Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [9.5, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    atoms.set_constraint(FixBondLengths([(0, 1)]))
    assert atoms.get_distance(0, 1, mic=True) == pytest.approx(0.5, abs=1e-6)

    device = torch.device("cpu")
    state = torch_sim.initialize_state([atoms], device, torch.float64)
    built = build_torchsim_fixbondlengths_from_ase_batch([atoms], device=device)
    assert built is not None
    state.constraints = [built]

    stretched = state.positions.clone()
    stretched[1, 0] = 9.7
    state.set_constrained_positions(stretched)
    delta = minimum_image_displacement(
        dr=(state.positions[1] - state.positions[0]).unsqueeze(0),
        cell=state.cell[0],
        pbc=state.pbc,
    )
    assert torch.linalg.norm(delta).item() == pytest.approx(0.5, abs=1e-5)
