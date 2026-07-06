"""Minimal benchmark sanity checks (fast EMT-only)."""

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT
from ase.optimize import LBFGS

from scgo.initialization import create_initial_cluster
from scgo.utils.helpers import perform_local_relaxation
from tests.constants import EMT_PT2_BOND_ANG, EMT_PT2_BOND_TOL_ANG


def test_pt2_analytical_verification():
    atoms = Atoms("Pt2", positions=[[0, 0, 0], [EMT_PT2_BOND_ANG, 0, 0]])
    atoms.calc = EMT()

    energy = perform_local_relaxation(atoms, EMT(), LBFGS, fmax=0.001, steps=20)

    final_bond_length = atoms.get_distance(0, 1)
    assert final_bond_length == pytest.approx(
        EMT_PT2_BOND_ANG, abs=EMT_PT2_BOND_TOL_ANG
    )
    assert np.isfinite(energy)


def test_pt3_analytical_verification():
    side_length = 2.5
    atoms = Atoms(
        "Pt3",
        positions=[
            [0, 0, 0],
            [side_length, 0, 0],
            [side_length / 2, side_length * np.sqrt(3) / 2, 0],
        ],
    )
    atoms.calc = EMT()

    energy = perform_local_relaxation(atoms, EMT(), LBFGS, fmax=0.001, steps=3)

    distances = [
        atoms.get_distance(0, 1),
        atoms.get_distance(1, 2),
        atoms.get_distance(2, 0),
    ]

    assert np.std(distances) < 0.5
    assert np.isfinite(energy)


def test_initial_structure_safety(rng):
    comp = ["Pt", "Pt", "Pt"]

    atoms = create_initial_cluster(comp, rng=rng)
    atoms.calc = EMT()

    distances = [atoms.get_distance(i, j) for i in range(3) for j in range(i + 1, 3)]
    min_dist = min(distances)

    assert min_dist >= 1.0, (
        f"Atoms too close: minimum distance {min_dist:.4f} Å < 1.0 Å"
    )

    forces = atoms.get_forces()
    max_force = np.max(np.linalg.norm(forces, axis=1))

    assert max_force < 100.0, (
        f"Initial structure has dangerously high forces ({max_force:.2f} eV/Å). "
        f"This suggests atoms are too close (min distance: {min_dist:.4f} Å). "
        f"Consider tightening min_distance_factor or adding force-based validation."
    )

    if max_force > 20.0:
        import warnings

        warnings.warn(
            f"Initial structure has high forces ({max_force:.2f} eV/Å) "
            f"with min distance {min_dist:.4f} Å. "
            f"This may cause energy to increase during relaxation.",
            stacklevel=2,
        )
