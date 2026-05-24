"""Detect MLIP calculators suitable for TorchSim batch relaxation."""

from __future__ import annotations

from ase.calculators.calculator import Calculator

_MLIP_CALCULATOR_CLASS_NAMES = frozenset(
    {"MACECalculator", "MACE", "UMA", "FAIRChemCalculator"}
)


def is_ml_calculator(calculator: Calculator) -> bool:
    """Return True when ``calculator`` is a known MLIP ASE calculator class."""
    return calculator.__class__.__name__ in _MLIP_CALCULATOR_CLASS_NAMES
