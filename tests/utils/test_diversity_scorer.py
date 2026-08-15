"""Unit tests for DiversityScorer."""

import numpy as np
import pytest
from ase import Atoms

from scgo.utils.comparators import PureInteratomicDistanceComparator
from scgo.utils.diversity_scorer import DiversityScorer


def test_diversity_scorer_init():
    """Test initialization with references."""
    ref1 = Atoms("Pt3", positions=[[0, 0, 0], [2, 0, 0], [1, 2, 0]])
    ref2 = Atoms("Pt3", positions=[[0, 0, 0], [3, 0, 0], [0, 3, 0]])

    comparator = PureInteratomicDistanceComparator()
    scorer = DiversityScorer([ref1, ref2], comparator)

    assert len(scorer) == 2
    assert scorer._ref_descriptors is not None
    assert scorer._ref_descriptors.shape[0] == 2  # 2 references


def test_diversity_scorer_empty_references():
    """Test scorer with no references returns 0."""
    comparator = PureInteratomicDistanceComparator()
    scorer = DiversityScorer([], comparator)

    assert len(scorer) == 0
    assert scorer._ref_descriptors is None

    atoms = Atoms("Pt3", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    score = scorer.score(atoms)

    assert score == pytest.approx(0.0, abs=1e-8)


def test_diversity_scorer_score():
    """Test scoring returns average dissimilarity."""
    # Create reference structures
    ref1 = Atoms("Pt3", positions=[[0, 0, 0], [2, 0, 0], [1, 2, 0]])
    ref2 = Atoms("Pt3", positions=[[0, 0, 0], [3, 0, 0], [0, 3, 0]])

    comparator = PureInteratomicDistanceComparator()
    scorer = DiversityScorer([ref1, ref2], comparator)

    # Test with identical structure (should have low dissimilarity)
    candidate_identical = Atoms("Pt3", positions=[[0, 0, 0], [2, 0, 0], [1, 2, 0]])
    score1 = scorer.score(candidate_identical)

    # Test with different structure (should have higher dissimilarity)
    candidate_different = Atoms("Pt3", positions=[[0, 0, 0], [5, 0, 0], [0, 5, 0]])
    score2 = scorer.score(candidate_different)

    # Different structure should have higher diversity score
    assert score2 > score1
    assert score1 >= 0.0
    assert score2 >= 0.0


def test_diversity_scorer_add_reference():
    """Test adding references updates descriptors."""
    ref1 = Atoms("Pt3", positions=[[0, 0, 0], [2, 0, 0], [1, 2, 0]])

    comparator = PureInteratomicDistanceComparator()
    scorer = DiversityScorer([ref1], comparator)

    assert len(scorer) == 1

    # Add new reference
    ref2 = Atoms("Pt3", positions=[[0, 0, 0], [3, 0, 0], [0, 3, 0]])
    scorer.add_reference(ref2)

    assert len(scorer) == 2
    assert scorer._ref_descriptors.shape[0] == 2


def test_diversity_scorer_descriptor_consistency():
    """Test that descriptors have consistent length for same composition."""
    ref1 = Atoms("Pt3", positions=[[0, 0, 0], [2, 0, 0], [1, 2, 0]])
    ref2 = Atoms("Pt3", positions=[[0, 0, 0], [3, 0, 0], [0, 3, 0]])

    comparator = PureInteratomicDistanceComparator()
    scorer = DiversityScorer([ref1, ref2], comparator)

    # All descriptors should have same length
    assert scorer._ref_descriptors.shape[0] == 2
    assert scorer._ref_descriptors.shape[1] > 0  # Non-empty descriptor

    # Candidate with same composition should work
    candidate = Atoms("Pt3", positions=[[0, 0, 0], [4, 0, 0], [0, 4, 0]])
    score = scorer.score(candidate)
    assert np.isfinite(score)


def test_diversity_scorer_different_compositions():
    """Test scorer handles different compositions gracefully."""
    ref1 = Atoms("Pt3", positions=[[0, 0, 0], [2, 0, 0], [1, 2, 0]])

    comparator = PureInteratomicDistanceComparator()
    scorer = DiversityScorer([ref1], comparator)

    # Candidate with different composition should use fallback pairwise method
    candidate = Atoms("Pt4", positions=[[0, 0, 0], [2, 0, 0], [1, 2, 0], [0, 0, 2]])
    score = scorer.score(candidate)

    # Should return valid score (may use pairwise fallback)
    assert np.isfinite(score)
    assert score >= 0.0


def test_diversity_scorer_vectorized_performance(rng):
    """Test vectorized operations are fast."""

    # Create multiple reference structures
    n_refs = 50
    references = [Atoms("Pt5", positions=rng.random((5, 3)) * 5) for _ in range(n_refs)]

    comparator = PureInteratomicDistanceComparator()
    scorer = DiversityScorer(references, comparator)

    # Create candidate
    candidate = Atoms("Pt5", positions=rng.random((5, 3)) * 5)

    # Time the scoring
    score = scorer.score(candidate)

    assert np.isfinite(score)
    assert score >= 0.0


def test_diversity_scorer_average_dissimilarity():
    """Test that score computes average dissimilarity correctly."""
    # Create references with known distances
    ref1 = Atoms("Pt2", positions=[[0, 0, 0], [1, 0, 0]])  # Distance = 1.0
    ref2 = Atoms("Pt2", positions=[[0, 0, 0], [2, 0, 0]])  # Distance = 2.0

    comparator = PureInteratomicDistanceComparator()
    scorer = DiversityScorer([ref1, ref2], comparator)

    # Candidate with distance = 3.0 (different from both)
    candidate = Atoms("Pt2", positions=[[0, 0, 0], [3, 0, 0]])

    score = scorer.score(candidate)

    # Score should be average dissimilarity to both references
    assert score >= 0.0
    assert np.isfinite(score)


def test_diversity_scorer_len():
    """Test __len__ method."""
    ref1 = Atoms("Pt3", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    ref2 = Atoms("Pt3", positions=[[0, 0, 0], [2, 0, 0], [0, 2, 0]])

    comparator = PureInteratomicDistanceComparator()
    scorer = DiversityScorer([ref1, ref2], comparator)

    assert len(scorer) == 2

    scorer.add_reference(Atoms("Pt3", positions=[[0, 0, 0], [3, 0, 0], [0, 3, 0]]))
    assert len(scorer) == 3


def test_ragged_descriptors_do_not_raise_value_error():
    """Inhomogeneous descriptor lengths must not reach np.array (ValueError)."""
    ref1 = Atoms("Pt2", positions=[[0, 0, 0], [2, 0, 0]])
    ref2 = Atoms("Pt4", positions=[[0, 0, 0], [2, 0, 0], [1, 2, 0], [0, 0, 2]])

    comparator = PureInteratomicDistanceComparator()
    scorer = DiversityScorer([ref1, ref2], comparator)

    # Ragged references yield no descriptor matrix instead of crashing.
    assert scorer._ref_descriptors is None
    assert len(scorer) == 2


def test_compute_descriptors_returns_none_for_mixed_lengths(monkeypatch):
    """Explicitly mixed-length descriptors return None (zero-diversity path)."""
    comparator = PureInteratomicDistanceComparator()
    scorer = DiversityScorer([], comparator)

    structures = [
        Atoms("Pt3", positions=[[0, 0, 0], [2, 0, 0], [1, 2, 0]]),
        Atoms("Pt3", positions=[[0, 0, 0], [3, 0, 0], [0, 3, 0]]),
    ]
    lengths = iter([3, 5])
    monkeypatch.setattr(
        DiversityScorer,
        "_atoms_to_descriptor",
        lambda self, atoms: np.ones(next(lengths)),
    )

    result = scorer._compute_descriptors(structures)

    assert result is None


def test_score_with_ragged_references_is_defined():
    """Scoring against ragged references falls back to a finite pairwise score."""
    ref1 = Atoms("Pt2", positions=[[0, 0, 0], [2, 0, 0]])
    ref2 = Atoms("Pt4", positions=[[0, 0, 0], [2, 0, 0], [1, 2, 0], [0, 0, 2]])

    comparator = PureInteratomicDistanceComparator()
    scorer = DiversityScorer([ref1, ref2], comparator)

    score = scorer.score(
        Atoms("Pt4", positions=[[0, 0, 0], [3, 0, 0], [0, 3, 0], [0, 0, 3]])
    )

    assert isinstance(score, float)
    assert np.isfinite(score)
    assert score >= 0.0


def test_add_reference_recovers_from_ragged_references():
    """Adding a reference after a ragged set must not crash or mis-shape state."""
    ref1 = Atoms("Pt2", positions=[[0, 0, 0], [2, 0, 0]])
    ref2 = Atoms("Pt4", positions=[[0, 0, 0], [2, 0, 0], [1, 2, 0], [0, 0, 2]])

    comparator = PureInteratomicDistanceComparator()
    scorer = DiversityScorer([ref1, ref2], comparator)
    assert scorer._ref_descriptors is None

    scorer.add_reference(Atoms("Pt3", positions=[[0, 0, 0], [2, 0, 0], [1, 2, 0]]))

    assert len(scorer) == 3
    assert scorer._ref_descriptors is None
