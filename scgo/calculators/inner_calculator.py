"""ASE calculator that forwards ``calculate`` to a wrapped inner calculator."""

from __future__ import annotations

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes


class InnerDelegatingCalculator(Calculator):
    def _install_inner(self, inner: Calculator) -> None:
        self._inner = inner
        self.implemented_properties = list(inner.implemented_properties)

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        if properties is None:
            properties = self.implemented_properties
        super().calculate(atoms, properties, system_changes)
        self._inner.calculate(
            atoms=self.atoms,
            properties=properties,
            system_changes=system_changes,
        )
        self.results = self._inner.results
