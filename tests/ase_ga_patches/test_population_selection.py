import numpy as np
import pytest
from ase import Atoms

from scgo.ase_ga_patches.population import Population
from scgo.exceptions import SCGOValidationError
from scgo.metadata.atoms import set_tags
from tests.helpers import create_paired_rngs


class SymbolsComparator:
    """Minimal comparator: candidates with the same formula look alike."""

    def looks_like(self, a, b):
        return a.get_chemical_symbols() == b.get_chemical_symbols()


class FakeDC:
    def __init__(self, candidates):
        self._cands = candidates

    def get_all_relaxed_candidates(self, only_new=False, use_extinct=False):
        return list(self._cands)

    def get_participation_in_pairing(self):
        # No history for this simple fake
        return ({}, set())


def _make_candidate(symbols, raw_score, confid, relax_id):
    a = Atoms(symbols)
    a.set_cell([10.0, 10.0, 10.0])
    a.set_pbc(False)
    # Add metadata used by get_raw_score
    set_tags(a, raw_score=raw_score)
    a.info["confid"] = confid
    a.info["relax_id"] = relax_id
    return a


def test_get_two_candidates_is_deterministic_with_seeded_rng():
    # Create three candidates with different raw scores
    c1 = _make_candidate(["Pt", "Pt"], raw_score=-5.0, confid="c1", relax_id=1)
    c2 = _make_candidate(["Pt", "Pt", "Pt"], raw_score=-10.0, confid="c2", relax_id=2)
    c3 = _make_candidate(["Au", "Pt"], raw_score=-3.0, confid="c3", relax_id=3)

    dc = FakeDC([c1, c2, c3])

    rng1, rng2 = create_paired_rngs(1234)

    pop1 = Population(dc, population_size=3, rng=rng1)
    pop2 = Population(dc, population_size=3, rng=rng2)

    pair1 = pop1.get_two_candidates()
    pair2 = pop2.get_two_candidates()

    assert pair1 is not None and pair2 is not None
    ids1 = tuple(sorted([pair1[0].info["confid"], pair1[1].info["confid"]]))
    ids2 = tuple(sorted([pair2[0].info["confid"], pair2[1].info["confid"]]))

    assert ids1 == ids2


def test_population_constructor_rejects_legacy_randomstate():
    import numpy as _np

    dc = FakeDC([])
    with pytest.raises(SCGOValidationError):
        Population(dc, population_size=2, rng=_np.random.RandomState(1))


def test_population_update_stable_tie_order():
    """Tied raw_score candidates are processed in relax_id/confid order."""
    from scgo.ase_ga_patches.population import _population_candidate_sort_key

    tie_high = _make_candidate(["Pt", "Pt"], raw_score=-5.0, confid=20, relax_id=2)
    tie_low = _make_candidate(["Pt", "Pt"], raw_score=-5.0, confid=10, relax_id=1)

    for input_order in ([tie_high, tie_low], [tie_low, tie_high]):
        ordered = list(input_order)
        ordered.sort(key=_population_candidate_sort_key)
        assert [a.info["relax_id"] for a in ordered] == [1, 2]


def test_add_candidate_on_empty_population_does_not_raise():
    """G7.1: adding to an empty population must not index ``self.pop[-1]``."""
    pop = Population(FakeDC([]), population_size=3, rng=np.random.default_rng(0))
    assert pop.pop == []

    candidate = _make_candidate(["Pt", "Pt"], raw_score=-4.0, confid="c1", relax_id=1)
    pop.__add_candidate__(candidate)

    assert [a.info["confid"] for a in pop.pop] == ["c1"]


def test_strictly_better_duplicate_replaces_top_elite():
    """G7.2: elitism must not block a strictly better copy of the best candidate."""
    c1 = _make_candidate(["Pt", "Pt"], raw_score=-5.0, confid="c1", relax_id=1)
    c2 = _make_candidate(["Pt", "Pt", "Pt"], raw_score=-10.0, confid="c2", relax_id=2)
    c3 = _make_candidate(["Au", "Pt"], raw_score=-3.0, confid="c3", relax_id=3)

    pop = Population(
        FakeDC([c1, c2, c3]),
        population_size=3,
        comparator=SymbolsComparator(),
        rng=np.random.default_rng(0),
    )
    assert pop.pop[0].info["confid"] == "c3"

    better = _make_candidate(["Au", "Pt"], raw_score=-1.0, confid="c4", relax_id=4)
    pop.__add_candidate__(better)

    assert pop.pop[0].info["confid"] == "c4"
    assert "c3" not in [a.info["confid"] for a in pop.pop]
    assert len(pop.pop) == 3


def test_tied_duplicate_keeps_incumbent_elite():
    """G7.2: a tie must not evict the incumbent."""
    c1 = _make_candidate(["Au", "Pt"], raw_score=-3.0, confid="c1", relax_id=1)
    c2 = _make_candidate(["Pt", "Pt"], raw_score=-5.0, confid="c2", relax_id=2)

    pop = Population(
        FakeDC([c1, c2]),
        population_size=2,
        comparator=SymbolsComparator(),
        rng=np.random.default_rng(0),
    )
    tie = _make_candidate(["Au", "Pt"], raw_score=-3.0, confid="c3", relax_id=3)
    pop.__add_candidate__(tie)

    assert [a.info["confid"] for a in pop.pop] == ["c1", "c2"]


def test_get_two_candidates_returns_distinct_confids_for_two_candidate_pool():
    """G7.3: sampling without replacement rules out confid collisions."""
    c1 = _make_candidate(["Pt", "Pt"], raw_score=-5.0, confid="c1", relax_id=1)
    c2 = _make_candidate(["Pt", "Pt", "Pt"], raw_score=-10.0, confid="c2", relax_id=2)

    pop = Population(FakeDC([c1, c2]), population_size=2, rng=np.random.default_rng(5))
    for _ in range(10):
        pair = pop.get_two_candidates()
        assert pair is not None
        assert pair[0].info["confid"] != pair[1].info["confid"]


def test_get_one_candidate_is_deterministic_with_seeded_rng():
    """G7.3: the roulette draw stays reproducible for a fixed seed."""
    c1 = _make_candidate(["Pt", "Pt"], raw_score=-5.0, confid="c1", relax_id=1)
    c2 = _make_candidate(["Pt", "Pt", "Pt"], raw_score=-10.0, confid="c2", relax_id=2)
    c3 = _make_candidate(["Au", "Pt"], raw_score=-3.0, confid="c3", relax_id=3)
    dc = FakeDC([c1, c2, c3])

    rng1, rng2 = create_paired_rngs(2024)
    pop1 = Population(dc, population_size=3, rng=rng1)
    pop2 = Population(dc, population_size=3, rng=rng2)

    picks1 = [pop1.get_one_candidate().info["confid"] for _ in range(8)]
    picks2 = [pop2.get_one_candidate().info["confid"] for _ in range(8)]

    assert picks1 == picks2
    assert set(picks1) <= {"c1", "c2", "c3"}
