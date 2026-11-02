"""Lock CONNECTIVITY_FACTOR to the bulk-shell argument."""

from __future__ import annotations

import numpy as np
import pytest
from ase.build import bulk

from scgo.initialization.atomic_radii import get_covalent_radius
from scgo.initialization.initialization_config import CONNECTIVITY_FACTOR

_FCC = ("Pt", "Cu", "Pd", "Ir", "Rh")


def _shells(symbol: str) -> tuple[float, float]:
    atoms = bulk(symbol, cubic=True).repeat((2, 2, 2))
    distances = atoms.get_all_distances(mic=True)[0]
    shells = np.unique(np.round(distances[distances > 1e-3], 5))
    return float(shells[0]), float(shells[1])


def _cutoff(symbol_i: str, symbol_j: str) -> float:
    return (get_covalent_radius(symbol_i) + get_covalent_radius(symbol_j)) * (
        CONNECTIVITY_FACTOR
    )


@pytest.mark.parametrize("symbol", _FCC)
def test_fcc_nearest_neighbor_bonded_and_second_shell_outside(symbol: str) -> None:
    nearest, second = _shells(symbol)
    cutoff = _cutoff(symbol, symbol)
    assert nearest <= cutoff
    assert second > cutoff


def test_aluminium_nearest_neighbor_is_bonded() -> None:
    nearest, _second = _shells("Al")
    assert nearest <= _cutoff("Al", "Al")


@pytest.mark.parametrize(
    ("symbol_i", "symbol_j", "reference"),
    [("O", "H", 0.97), ("H", "H", 0.74)],
)
def test_reference_bond_is_inside_cutoff(
    symbol_i: str, symbol_j: str, reference: float
) -> None:
    assert reference <= _cutoff(symbol_i, symbol_j)
