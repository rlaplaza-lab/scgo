"""Unit tests for fitness strategy implementations."""

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT

from scgo.exceptions import SCGOValidationError
from scgo.utils.comparators import PureInteratomicDistanceComparator
from scgo.utils.diversity_scorer import DiversityScorer
from scgo.utils.fitness_strategies import (
    FitnessStrategy,
    calculate_fitness,
    get_fitness_from_atoms,
    resolve_fitness_strategy,
    set_fitness_in_atoms,
    validate_fitness_strategy,
)


def test_fitness_strategy_enum():
    """Test FitnessStrategy enum values."""
    assert FitnessStrategy.LOW_ENERGY == "low_energy"
    assert FitnessStrategy.HIGH_ENERGY == "high_energy"
    assert FitnessStrategy.DIVERSITY == "diversity"

    # Test all enum members exist
    assert len(FitnessStrategy) == 3


@pytest.mark.parametrize("strategy", ["low_energy", "high_energy", "diversity"])
def test_validate_fitness_strategy_valid(strategy):
    """Test validation accepts valid strategies."""
    validate_fitness_strategy(strategy)  # Should not raise


def test_validate_fitness_strategy_invalid():
    """Test validation rejects invalid strategies."""
    with pytest.raises(SCGOValidationError, match="must be one of"):
        validate_fitness_strategy("invalid_strategy")

    with pytest.raises(SCGOValidationError, match="must be one of"):
        validate_fitness_strategy("energy")  # Missing prefix

    with pytest.raises(SCGOValidationError, match="must be one of"):
        validate_fitness_strategy("")


def test_resolve_fitness_strategy_inherits_from_top_level():
    """Preset None should inherit from the top-level strategy."""
    assert resolve_fitness_strategy(None, inherit_from="high_energy") == "high_energy"
    assert (
        resolve_fitness_strategy("diversity", inherit_from="low_energy") == "diversity"
    )


def test_resolve_fitness_strategy_rejects_none_when_not_allowed():
    """Unresolved preset sentinels must fail at algorithm boundaries."""
    with pytest.raises(SCGOValidationError, match="cannot be None"):
        resolve_fitness_strategy(None, allow_none=False)


def test_calculate_fitness_low_energy():
    """Test low_energy fitness calculation."""
    atoms = Atoms("Pt3", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    atoms.calc = EMT()
    energy = atoms.get_potential_energy()

    fitness = calculate_fitness(energy, atoms, "low_energy")

    # For low_energy, fitness = -energy
    assert fitness == pytest.approx(-energy)
    assert fitness < 0  # EMT energy is positive, so fitness is negative


def test_calculate_fitness_high_energy():
    """Test high_energy fitness calculation."""
    atoms = Atoms("Pt3", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    atoms.calc = EMT()
    energy = atoms.get_potential_energy()

    fitness = calculate_fitness(energy, atoms, "high_energy")

    # For high_energy, fitness = energy
    assert fitness == pytest.approx(energy)
    assert fitness > 0  # EMT energy is positive


def test_calculate_fitness_diversity_no_references():
    """Test diversity fitness with no references returns 0."""
    atoms = Atoms("Pt3", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    atoms.calc = EMT()
    energy = atoms.get_potential_energy()

    comparator = PureInteratomicDistanceComparator()
    diversity_scorer = DiversityScorer([], comparator)

    fitness = calculate_fitness(
        energy, atoms, "diversity", diversity_scorer=diversity_scorer
    )

    assert fitness == pytest.approx(0.0, abs=1e-8)


def test_calculate_fitness_diversity_missing_scorer():
    """Test diversity fitness raises when scorer is not provided."""
    atoms = Atoms("Pt3", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    atoms.calc = EMT()
    energy = atoms.get_potential_energy()

    with pytest.raises(SCGOValidationError, match="requires a diversity_scorer"):
        calculate_fitness(energy, atoms, "diversity")


def test_calculate_fitness_diversity_with_references():
    """Test diversity fitness calculation with references."""
    # Create test structures
    atoms1 = Atoms("Pt3", positions=[[0, 0, 0], [2, 0, 0], [1, 2, 0]])
    atoms2 = Atoms("Pt3", positions=[[0, 0, 0], [3, 0, 0], [0, 3, 0]])
    atoms_candidate = Atoms("Pt3", positions=[[0, 0, 0], [4, 0, 0], [0, 4, 0]])

    comparator = PureInteratomicDistanceComparator()
    diversity_scorer = DiversityScorer([atoms1, atoms2], comparator)

    atoms_candidate.calc = EMT()
    energy = atoms_candidate.get_potential_energy()

    fitness = calculate_fitness(
        energy, atoms_candidate, "diversity", diversity_scorer=diversity_scorer
    )

    # Fitness should be average dissimilarity (non-negative)
    assert fitness >= 0.0
    assert np.isfinite(fitness)


def test_get_set_fitness_in_atoms():
    """Test storing and retrieving fitness from Atoms object."""
    atoms = Atoms("Pt3", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])

    # Initially no fitness
    assert get_fitness_from_atoms(atoms, default=-999) == -999

    # Set fitness
    set_fitness_in_atoms(atoms, 42.5, "low_energy")

    # Retrieve fitness
    assert get_fitness_from_atoms(atoms) == pytest.approx(42.5)
    assert atoms.info["fitness_strategy"] == "low_energy"

    # Test with different default
    atoms2 = Atoms("Pt2", positions=[[0, 0, 0], [1, 0, 0]])
    assert get_fitness_from_atoms(atoms2) is None
    assert get_fitness_from_atoms(atoms2, default=0.0) == pytest.approx(0.0, abs=1e-8)


def test_fitness_calculation_with_enum():
    """Test fitness calculation accepts FitnessStrategy enum."""
    atoms = Atoms("Pt3", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    atoms.calc = EMT()
    energy = atoms.get_potential_energy()

    # Test with enum
    fitness1 = calculate_fitness(energy, atoms, FitnessStrategy.LOW_ENERGY)

    # Test with string
    fitness2 = calculate_fitness(energy, atoms, "low_energy")

    # Should give same result
    assert fitness1 == pytest.approx(fitness2)


def test_fitness_strategy_unknown():
    """Test that unknown strategy raises error."""
    atoms = Atoms("Pt3", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    atoms.calc = EMT()
    energy = atoms.get_potential_energy()

    # This should not happen in practice due to enum, but test error handling
    with pytest.raises(SCGOValidationError, match="must be one of"):
        # Force invalid enum value (shouldn't happen in practice)
        calculate_fitness(energy, atoms, "invalid")  # Will fail validation first


def test_fitness_storage_persistence():
    """Test that fitness values persist through Atoms operations."""
    atoms = Atoms("Pt3", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    set_fitness_in_atoms(atoms, 10.5, "high_energy")

    # Copy should preserve fitness
    atoms_copy = atoms.copy()
    assert get_fitness_from_atoms(atoms_copy) == pytest.approx(10.5)
    assert atoms_copy.info["fitness_strategy"] == "high_energy"
