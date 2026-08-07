import pytest
from ase import Atoms

from scgo.ase_ga_patches.population import Population
from scgo.database.metadata import add_metadata
from scgo.exceptions import SCGOValidationError
from tests.test_utils import create_paired_rngs


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
    add_metadata(a, raw_score=raw_score)
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


class _CountingComparator:
    """Comparator that records how many pairwise comparisons were requested."""

    def __init__(self):
        self.calls = 0

    def looks_like(self, a1, a2):
        self.calls += 1
        return a1.info["confid"] == a2.info["confid"]


def test_looks_like_counts_scale_with_population_not_history():
    """``looks_like`` bookkeeping must not rescan the full candidate history."""
    population_size = 5
    initial = [
        _make_candidate(["Pt", "Pt"], raw_score=-float(i), confid=i, relax_id=i)
        for i in range(population_size)
    ]
    dc = FakeDC(initial)
    comparator = _CountingComparator()
    pop = Population(dc, population_size=population_size, comparator=comparator)

    baseline = comparator.calls
    for i in range(200):
        confid = 1000 + i
        cand = _make_candidate(
            ["Pt", "Pt"], raw_score=-0.5, confid=confid, relax_id=confid
        )
        pop.__add_candidate__(cand)
        pop.all_cand.append(cand)

    per_insert = (comparator.calls - baseline) / 200.0
    # Comparisons per insert stay bounded by the population size; the previous
    # implementation grew linearly with the number of candidates seen so far.
    assert per_insert <= 2 * population_size


def test_looks_like_counts_population_duplicates():
    """The stored count equals the number of population members that match."""
    from scgo.ase_ga_patches.population import count_looks_like

    class _AlwaysSimilar:
        def looks_like(self, a1, a2):
            return True

    pop_members = [
        _make_candidate(["Pt", "Pt"], raw_score=-float(i), confid=i, relax_id=i)
        for i in range(4)
    ]
    candidate = _make_candidate(["Pt", "Pt"], raw_score=-1.5, confid=99, relax_id=99)

    assert count_looks_like(candidate, pop_members, _AlwaysSimilar()) == 4
    # Same-confid entries (extra relaxation steps) are skipped.
    assert count_looks_like(pop_members[0], pop_members, _AlwaysSimilar()) == 3


def test_write_log_uses_incremental_max_generation(tmp_path):
    """The log records the highest generation without rescanning ``all_cand``."""
    from scgo.database.metadata import add_metadata as _add_metadata

    candidates = []
    for i in range(4):
        cand = _make_candidate(
            ["Pt", "Pt"], raw_score=-float(i), confid=i, relax_id=i + 1
        )
        _add_metadata(cand, generation=i)
        candidates.append(cand)

    logfile = tmp_path / "population.log"
    pop = Population(FakeDC(candidates), population_size=4, logfile=str(logfile))
    assert pop._max_gen == 3

    pop._write_log()
    line = logfile.read_text().strip().splitlines()[-1]
    assert line.split(":")[-2].strip() == "3"

    newer = _make_candidate(["Pt", "Pt"], raw_score=-9.0, confid=99, relax_id=99)
    _add_metadata(newer, generation=7)
    pop.update(new_cand=[newer])
    assert pop._max_gen == 7
    assert logfile.read_text().strip().splitlines()[-1].split(":")[-2].strip() == "7"


def test_write_log_reports_unknown_generation_when_metadata_missing(tmp_path):
    """Missing generation metadata still yields the legacy blank marker."""
    candidates = [
        _make_candidate(["Pt", "Pt"], raw_score=-float(i), confid=i, relax_id=i + 1)
        for i in range(3)
    ]
    logfile = tmp_path / "population.log"
    pop = Population(FakeDC(candidates), population_size=3, logfile=str(logfile))
    pop._write_log()

    line = logfile.read_text().strip().splitlines()[-1]
    assert line.split(":")[-2] == "  "
