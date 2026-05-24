"""ASE sequential batch relaxer for GA when TorchSim MLIP relaxer is not used."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ase import Atoms
from ase.optimize import FIRE
from ase.optimize.optimize import Optimizer

from scgo.utils.helpers import perform_local_relaxation


class AseBatchRelaxer:
    """Relax structures one-by-one with ASE optimizers (EMT and other ASE calculators)."""

    def __init__(
        self,
        calculator: Any,
        *,
        optimizer: type[Optimizer] = FIRE,
        force_tol: float = 0.05,
        max_steps: int = 250,
    ) -> None:
        self.calculator = calculator
        self.optimizer = optimizer
        self.force_tol = force_tol
        self.max_steps = max_steps

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
            )
            results.append((float(energy), relaxed))
        return results
