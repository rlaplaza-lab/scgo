"""ASE sequential batch relaxer for GA when TorchSim MLIP relaxer is not used."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ase import Atoms
from ase.optimize import FIRE
from ase.optimize.optimize import Optimizer

from scgo.utils.helpers import perform_local_relaxation


class AseBatchRelaxer:
    """Relax structures one-by-one with ASE optimizers.

    Used for EMT and other plain ASE calculators, i.e. whenever no batched
    MLIP relaxer is available.

    Args:
        calculator: ASE calculator used for every structure in the batch.
        optimizer: ASE optimizer class (default :class:`ase.optimize.FIRE`).
        force_tol: Force convergence criterion in eV/Å.
        max_steps: Default step cap when ``relax_batch`` is called without
            ``steps``.
        surface_mode: When True, relaxed structures are canonicalized with the
            slab-aware storage frame instead of the gas-phase
            ``wrap()``/``center()`` (which would translate the fixed slab).
        n_slab: Number of leading slab atoms; required by the slab-aware
            canonicalize when ``surface_mode`` is True.
    """

    def __init__(
        self,
        calculator: Any,
        *,
        optimizer: type[Optimizer] = FIRE,
        force_tol: float = 0.05,
        max_steps: int = 250,
        surface_mode: bool = False,
        n_slab: int = 0,
    ) -> None:
        self.calculator = calculator
        self.optimizer = optimizer
        self.force_tol = force_tol
        self.max_steps = max_steps
        self.surface_mode = bool(surface_mode)
        self.n_slab = int(n_slab)

    def relax_batch(
        self,
        batch: Sequence[Atoms],
        *,
        steps: int | None = None,
    ) -> list[tuple[float, Atoms]]:
        n_steps = self.max_steps if steps is None else steps
        results: list[tuple[float, Atoms]] = []
        for atoms in batch:
            relaxed = atoms.copy()
            relaxed.calc = self.calculator
            energy = perform_local_relaxation(
                relaxed,
                self.calculator,
                self.optimizer,
                fmax=self.force_tol,
                steps=n_steps,
                center_after_relax=not self.surface_mode,
                surface_mode=self.surface_mode,
                n_slab=self.n_slab,
            )
            results.append((float(energy), relaxed))
        return results
