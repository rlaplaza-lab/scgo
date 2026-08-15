"""TorchSim-side constraint implementations not provided by upstream torch-sim.

torch-sim ships ``FixAtoms``, ``FixCom`` and ``FixSymmetry`` but no bond-length
constraint. SCGO needs to honor ASE ``FixBondLengths`` during batched TorchSim
relaxation, so this module provides a TorchSim ``Constraint`` that restores
fixed bond lengths after every optimizer step, mirroring the ASE
``FixBondLengths`` semantics (project positions and forces so the constrained
relative distances are preserved).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from ase import Atoms
from ase.constraints import FixBondLengths as ASEFixBondLengths
from torch_sim.constraints import Constraint

__all__ = [
    "TorchSimFixBondLengths",
    "build_torchsim_fixbondlengths_from_ase_batch",
    "collect_ase_fixbondlengths",
]


class TorchSimFixBondLengths(Constraint):
    """TorchSim constraint that keeps selected interatomic distances fixed.

    Unlike :class:`torch_sim.constraints.FixAtoms`, which pins fixed atoms to
    their current positions, this constraint restores each listed bond to its
    target length after every proposed position update. Both atoms of a bond are
    displaced by half the required correction along the bond axis, which keeps
    the center of mass of the diatomic pair stationary (no spurious translation
    is introduced). The corresponding force component along the bond axis is
    removed from both atoms so the optimizer does not keep fighting the
    constraint.
    """

    def __init__(
        self,
        pairs: torch.Tensor | list[list[int]],
        bond_lengths: torch.Tensor | list[float],
        system_idx: torch.Tensor | list[int] | None = None,
        *,
        device: torch.device | None = None,
    ) -> None:
        pairs_t = torch.as_tensor(np.asarray(pairs, dtype=np.int64))
        if pairs_t.ndim != 2 or pairs_t.shape[1] != 2:
            raise ValueError(
                "FixBondLengths pairs must have shape (n_bonds, 2), "
                f"got {tuple(pairs_t.shape)}"
            )
        # ``index_put``-style integer tensors must be long; float lengths float64.
        self.pairs = pairs_t.long()
        self.bond_lengths = torch.as_tensor(np.asarray(bond_lengths, dtype=np.float64))
        if self.bond_lengths.shape[0] != self.pairs.shape[0]:
            raise ValueError(
                "FixBondLengths bond_lengths must have one entry per pair "
                f"({self.pairs.shape[0]}), got {self.bond_lengths.shape[0]}"
            )
        if system_idx is None:
            system_idx = torch.zeros(self.pairs.shape[0], dtype=torch.long)
        else:
            system_idx = torch.as_tensor(np.asarray(system_idx), dtype=torch.long)
        self.system_idx = system_idx
        if device is not None:
            self.pairs = self.pairs.to(device=device)
            self.bond_lengths = self.bond_lengths.to(device=device)
            self.system_idx = self.system_idx.to(device=device)

    def get_removed_dof(self, state: object) -> torch.Tensor:
        """One degree of freedom removed per constrained bond, per system."""
        counts = torch.zeros(state.n_systems, dtype=torch.long, device=state.device)
        if self.pairs.shape[0] == 0:
            return counts
        counts.index_add_(
            0, self.system_idx.to(device=state.device), torch.ones_like(self.system_idx)
        )
        return counts

    def adjust_positions(self, state: object, new_positions: torch.Tensor) -> None:
        """Pull each constrained bond back to its target length."""
        for k in range(self.pairs.shape[0]):
            i = int(self.pairs[k, 0])
            j = int(self.pairs[k, 1])
            target = float(self.bond_lengths[k])
            pi = new_positions[i]
            pj = new_positions[j]
            d = pj - pi
            dist = torch.linalg.norm(d)
            if dist <= 1e-12:
                continue
            direction = d / dist
            correction = 0.5 * (dist - target) * direction
            new_positions[i] = pi + correction
            new_positions[j] = pj - correction

    def adjust_forces(self, state: object, forces: torch.Tensor) -> None:
        """Remove the bond-stretching component of the relative force."""
        for k in range(self.pairs.shape[0]):
            i = int(self.pairs[k, 0])
            j = int(self.pairs[k, 1])
            pi = state.positions[i]
            pj = state.positions[j]
            d = pj - pi
            dist = torch.linalg.norm(d)
            if dist <= 1e-12:
                continue
            direction = d / dist
            rel_force = forces[j] - forces[i]
            parallel = torch.dot(rel_force, direction) * direction
            forces[i] = forces[i] + 0.5 * parallel
            forces[j] = forces[j] - 0.5 * parallel

    def select_constraint(
        self,
        atom_mask: torch.Tensor,
        system_mask: torch.Tensor,  # noqa: ARG002
    ) -> Constraint | None:
        """Keep only bonds whose both atoms survive the atom mask."""
        if self.pairs.shape[0] == 0:
            return None
        in_mask = atom_mask[self.pairs]
        keep = in_mask.all(dim=1)
        if not keep.any():
            return None
        return type(self)(
            self.pairs[keep],
            self.bond_lengths[keep],
            self.system_idx[keep],
            device=self.pairs.device,
        )

    def select_sub_constraint(
        self, atom_idx: torch.Tensor, sys_idx: int
    ) -> Constraint | None:
        """Keep bonds for ``sys_idx`` and renumber atoms to local indices."""
        if self.pairs.shape[0] == 0:
            return None
        if atom_idx.device != self.pairs.device:
            atom_idx = atom_idx.to(self.pairs.device)
        sys_mask = self.system_idx == sys_idx
        pairs = self.pairs[sys_mask]
        if pairs.shape[0] == 0:
            return None
        present = torch.isin(pairs, atom_idx).all(dim=1)
        pairs = pairs[present]
        if pairs.shape[0] == 0:
            return None
        local_index = {int(a): k for k, a in enumerate(atom_idx.tolist())}
        local_pairs = torch.tensor(
            [[local_index[int(a)], local_index[int(b)]] for a, b in pairs.tolist()],
            dtype=torch.long,
            device=self.pairs.device,
        )
        return type(self)(
            local_pairs,
            self.bond_lengths[sys_mask][present],
            torch.zeros(pairs.shape[0], dtype=torch.long),
            device=self.pairs.device,
        )

    def reindex(self, atom_offset: int, system_offset: int) -> TorchSimFixBondLengths:
        """Return a copy with global atom/system indices shifted."""
        return type(self)(
            self.pairs + atom_offset,
            self.bond_lengths,
            self.system_idx + system_offset,
            device=self.pairs.device,
        )

    @classmethod
    def merge(cls, constraints: list[Constraint]) -> TorchSimFixBondLengths:
        """Merge already-reindexed bond constraints into one."""
        bond_constraints = [c for c in constraints if isinstance(c, cls)]
        if not bond_constraints:
            raise ValueError(
                f"{cls.__name__}.merge requires at least one {cls.__name__}."
            )
        device = bond_constraints[0].pairs.device
        pairs = torch.cat([c.pairs for c in bond_constraints])
        lengths = torch.cat([c.bond_lengths for c in bond_constraints])
        systems = torch.cat([c.system_idx for c in bond_constraints])
        return cls(pairs, lengths, systems, device=device)

    def to(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> TorchSimFixBondLengths:
        """Return a copy with all internal tensors moved to *device*/*dtype*."""
        return type(self)(
            self.pairs.to(device=device),
            self.bond_lengths.to(device=device, dtype=dtype),
            self.system_idx.to(device=device),
            device=device,
        )


def collect_ase_fixbondlengths(
    atoms: Atoms, offset: int = 0
) -> list[tuple[torch.Tensor, torch.Tensor, int]]:
    """Extract ASE ``FixBondLengths`` pairs/lengths with global atom indices.

    Args:
        atoms: ASE structure (single system).
        offset: Atom-index offset for batching into a global index space.

    Returns:
        A list of ``(pairs, lengths, system_index)`` tuples, one per
        ``FixBondLengths`` constraint found on ``atoms``. ``pairs`` is a
        ``(n_bonds, 2)`` long tensor of global atom indices and ``lengths`` is a
        ``(n_bonds,)`` float64 tensor of target bond lengths (taken from the
        constraint's cached ``bondlengths`` when present, else measured from the
        current geometry).
    """
    out: list[tuple[torch.Tensor, torch.Tensor, int]] = []
    for sys_id, constraint in enumerate(atoms.constraints):
        if not isinstance(constraint, ASEFixBondLengths):
            continue
        pairs = np.asarray(constraint.pairs, dtype=np.int64) + offset
        if getattr(constraint, "bondlengths", None) is not None:
            lengths = np.asarray(constraint.bondlengths, dtype=np.float64)
        else:
            lengths = np.array(
                [atoms.get_distance(a, b, mic=True) for a, b in pairs - offset],
                dtype=np.float64,
            )
        out.append(
            (
                torch.as_tensor(pairs),
                torch.as_tensor(lengths),
                sys_id,
            )
        )
    return out


def build_torchsim_fixbondlengths_from_ase_batch(
    atoms_list: Sequence[Atoms],
    device: object,
) -> TorchSimFixBondLengths | None:
    """Map per-structure ASE ``FixBondLengths`` to one global TorchSim constraint.

    Mirrors :func:`build_torchsim_fixatoms_from_ase_batch`: torch-sim does not
    read ``atoms.constraints`` so SCGO builds the TorchSim constraint explicitly
    and attaches it to the ``SimState`` before calling ``ts.optimize``.

    Args:
        atoms_list: One or more ASE systems in batch order.
        device: ``torch.device`` for the index/length tensors.

    Returns:
        A :class:`TorchSimFixBondLengths` instance, or ``None`` if no
        ``FixBondLengths`` constraints are present.
    """
    pairs_all: list[torch.Tensor] = []
    lengths_all: list[torch.Tensor] = []
    systems_all: list[torch.Tensor] = []
    offset = 0
    for batch_idx, atoms in enumerate(atoms_list):
        for pairs, lengths, _sys_id in collect_ase_fixbondlengths(atoms, offset=offset):
            pairs_all.append(pairs)
            lengths_all.append(lengths)
            systems_all.append(
                torch.full((pairs.shape[0],), batch_idx, dtype=torch.long)
            )
        offset += len(atoms)
    if not pairs_all:
        return None
    return TorchSimFixBondLengths(
        torch.cat(pairs_all),
        torch.cat(lengths_all),
        torch.cat(systems_all),
        device=torch.device(device) if device is not None else None,
    )
