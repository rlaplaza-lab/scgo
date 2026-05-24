"""Detect MLIP calculators suitable for TorchSim batch relaxation."""

from __future__ import annotations

from ase.calculators.calculator import Calculator


def is_ml_calculator(calculator: Calculator) -> bool:
    """Return True when ``calculator`` looks like an MLIP (MACE/UMA/FairChem)."""
    calculator_class_name = calculator.__class__.__name__
    model = getattr(calculator, "model", None)
    return hasattr(model, "forward") or calculator_class_name in (
        "MACECalculator",
        "MACE",
        "UMA",
        "FAIRChemCalculator",
    )
