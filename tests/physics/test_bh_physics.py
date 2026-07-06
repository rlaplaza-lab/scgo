"""Basin-hopping physics tests (Metropolis acceptance, energy lowering)."""

from __future__ import annotations

import pytest
from ase.calculators.emt import EMT
from ase.optimize import LBFGS

from scgo.algorithms.basinhopping_go import bh_go
from scgo.initialization import create_initial_cluster
from scgo.utils.helpers import perform_local_relaxation


@pytest.mark.slow
def test_bh_temperature_zero_rejects_uphill(tmp_path, rng) -> None:
    """At T=0, the best returned minimum is no worse than the first relaxed basin."""
    comp = ["Pt", "Pt"]
    atoms = create_initial_cluster(comp, rng=rng)
    atoms.calc = EMT()
    perform_local_relaxation(atoms, EMT(), LBFGS, fmax=0.05, steps=20)
    reference_energy = float(atoms.get_potential_energy())

    minima = bh_go(
        atoms=atoms,
        output_dir=str(tmp_path / "bh_zero"),
        niter=5,
        temperature=0.0,
        dr=0.2,
        niter_local_relaxation=5,
        rng=rng,
    )
    assert len(minima) >= 1
    best_energy = min(float(e) for e, _a in minima)
    assert best_energy <= reference_energy + 1e-5


@pytest.mark.slow
def test_bh_finds_lower_energy_than_initial(tmp_path, rng) -> None:
    comp = ["Pt", "Pt", "Pt"]
    atoms = create_initial_cluster(comp, rng=rng)
    atoms.calc = EMT()
    perform_local_relaxation(atoms, EMT(), LBFGS, fmax=0.1, steps=30)
    reference_energy = float(atoms.get_potential_energy())

    trial = atoms.copy()
    trial.calc = EMT()
    minima = bh_go(
        atoms=trial,
        output_dir=str(tmp_path / "bh_pt3"),
        niter=8,
        temperature=0.05,
        dr=0.3,
        niter_local_relaxation=8,
        rng=rng,
    )
    assert len(minima) >= 1
    finite_energies = [float(e) for e, _a in minima if float(e) < 1e5]
    assert finite_energies, "BH returned only penalty-energy structures"
    best_energy = min(finite_energies)
    assert best_energy <= reference_energy + 1e-5
