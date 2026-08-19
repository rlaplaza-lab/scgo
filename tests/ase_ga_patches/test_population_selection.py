import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase_ga.data import DataConnection

import scgo.algorithms.geneticalgorithm_go_torchsim as ga_mod
from scgo.ase_ga_patches.population import FitnessStrategyPopulation, Population
from scgo.exceptions import SCGOValidationError
from scgo.metadata.atoms import set_tags
from scgo.utils.comparators import PureInteratomicDistanceComparator
from scgo.utils.diversity_scorer import DiversityScorer
from tests.helpers import create_paired_rngs, create_preparedb


class SymbolsComparator:
    """Minimal comparator: candidates with the same formula look alike."""

    def looks_like(self, a, b):
        return a.get_chemical_symbols() == b.get_chemical_symbols()


class FakeDC:
    def __init__(self, candidates):
        self._cands = candidates
        self.already_returned: set = set()

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
    with pytest.raises(
        SCGOValidationError, match="rng must be an instance of numpy.random.Generator"
    ):
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


def test_add_candidate_on_empty_population_does_not_raise(rng):
    """G7.1: adding to an empty population must not index ``self.pop[-1]``."""
    pop = Population(FakeDC([]), population_size=3, rng=rng)
    assert pop.pop == []

    candidate = _make_candidate(["Pt", "Pt"], raw_score=-4.0, confid="c1", relax_id=1)
    pop.__add_candidate__(candidate)

    assert [a.info["confid"] for a in pop.pop] == ["c1"]


def test_strictly_better_duplicate_replaces_top_elite(rng):
    """G7.2: elitism must not block a strictly better copy of the best candidate."""
    c1 = _make_candidate(["Pt", "Pt"], raw_score=-5.0, confid="c1", relax_id=1)
    c2 = _make_candidate(["Pt", "Pt", "Pt"], raw_score=-10.0, confid="c2", relax_id=2)
    c3 = _make_candidate(["Au", "Pt"], raw_score=-3.0, confid="c3", relax_id=3)

    pop = Population(
        FakeDC([c1, c2, c3]),
        population_size=3,
        comparator=SymbolsComparator(),
        rng=rng,
    )
    assert pop.pop[0].info["confid"] == "c3"

    better = _make_candidate(["Au", "Pt"], raw_score=-1.0, confid="c4", relax_id=4)
    pop.__add_candidate__(better)

    assert pop.pop[0].info["confid"] == "c4"
    assert "c3" not in [a.info["confid"] for a in pop.pop]
    assert len(pop.pop) == 3


def test_tied_duplicate_keeps_incumbent_elite(rng):
    """G7.2: a tie must not evict the incumbent."""
    c1 = _make_candidate(["Au", "Pt"], raw_score=-3.0, confid="c1", relax_id=1)
    c2 = _make_candidate(["Pt", "Pt"], raw_score=-5.0, confid="c2", relax_id=2)

    pop = Population(
        FakeDC([c1, c2]),
        population_size=2,
        comparator=SymbolsComparator(),
        rng=rng,
    )
    tie = _make_candidate(["Au", "Pt"], raw_score=-3.0, confid="c3", relax_id=3)
    pop.__add_candidate__(tie)

    assert [a.info["confid"] for a in pop.pop] == ["c1", "c2"]


def test_get_two_candidates_returns_distinct_confids_for_two_candidate_pool(rng):
    """G7.3: sampling without replacement rules out confid collisions."""
    c1 = _make_candidate(["Pt", "Pt"], raw_score=-5.0, confid="c1", relax_id=1)
    c2 = _make_candidate(["Pt", "Pt", "Pt"], raw_score=-10.0, confid="c2", relax_id=2)

    pop = Population(FakeDC([c1, c2]), population_size=2, rng=rng)
    for _ in range(10):
        pair = pop.get_two_candidates()
        assert pair is not None
        assert pair[0].info["confid"] != pair[1].info["confid"]


# ---------------------------------------------------------------------------
# Tests for in-memory update (new_cand=...) path
# ---------------------------------------------------------------------------


def test_update_with_new_cand_does_not_call_get_all_relaxed_candidates(rng):
    """Passing new_cand=... must not trigger a DB read."""
    c1 = _make_candidate(["Pt", "Pt"], raw_score=-5.0, confid=1, relax_id=1)
    dc = FakeDC([c1])
    pop = Population(dc, population_size=3, rng=rng)
    assert len(pop.pop) == 1

    call_log: list[str] = []
    original_fn = dc.get_all_relaxed_candidates

    def _fail(*args, **kwargs):
        call_log.append("called")
        return original_fn(*args, **kwargs)

    dc.get_all_relaxed_candidates = _fail

    c2 = _make_candidate(["Pt", "Pt"], raw_score=-3.0, confid=2, relax_id=2)
    pop.update(new_cand=[c2])

    assert not call_log, (
        "get_all_relaxed_candidates must not be called when new_cand is provided"
    )
    assert any(a.info["confid"] == 2 for a in pop.all_cand)


def test_update_with_empty_new_cand_skips_db_and_adds_nothing(rng):
    """update(new_cand=[]) is valid: no DB call and all_cand unchanged."""
    c1 = _make_candidate(["Pt", "Pt"], raw_score=-5.0, confid=1, relax_id=1)
    dc = FakeDC([c1])
    pop = Population(dc, population_size=3, rng=rng)
    initial_len = len(pop.all_cand)

    call_log: list[str] = []

    def _fail(*args, **kwargs):
        call_log.append("called")
        return list(dc._cands)

    dc.get_all_relaxed_candidates = _fail
    pop.update(new_cand=[])

    assert not call_log
    assert len(pop.all_cand) == initial_len


def test_update_with_new_cand_syncs_already_returned(rng):
    """Gaids from new_cand must be registered so a later update() does not reload them."""
    c1 = _make_candidate(["Pt", "Pt"], raw_score=-5.0, confid=1, relax_id=1)
    c2 = _make_candidate(["Pt", "Pt"], raw_score=-4.0, confid=2, relax_id=2)
    dc = FakeDC([c1])
    pop = Population(dc, population_size=5, rng=rng)

    pop.update(new_cand=[c2])

    # dc.already_returned must contain the gaid of c2 so a subsequent
    # update(new_cand=None) using only_new=True won't re-fetch it.
    assert 2 in dc.already_returned


def test_update_with_new_cand_syncs_already_returned_prevents_db_refetch(tmp_path, rng):
    """After update(new_cand=[...]), a fallback update() must not re-fetch the same gaids.

    Uses a real DataConnection so that ``already_returned`` and ``only_new=True``
    behave as they do in a live GA run.
    """
    pt2 = Atoms("Pt2", positions=[[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
    pt2.center(vacuum=5.0)
    db_path = tmp_path / "nodup.db"
    prep = create_preparedb(pt2, db_path, population_size=5)

    cand = pt2.copy()
    set_tags(cand, raw_score=-1.0)
    prep.add_unrelaxed_candidate(cand, description="cand:0")
    da = DataConnection(str(db_path))

    rel = pt2.copy()
    rel.info["confid"] = 2
    rel.info["key_value_pairs"] = {"raw_score": -1.0}
    rel.info["data"] = {}
    rel.calc = SinglePointCalculator(rel, energy=-1.0)
    da.add_relaxed_step(rel)

    pop = Population(da, population_size=5, rng=rng)
    assert 2 in da.already_returned

    pop.update(new_cand=[rel])

    len_before_fallback = len(pop.all_cand)
    pop.update(new_cand=None)
    assert len(pop.all_cand) == len_before_fallback


def test_update_with_new_cand_ineligible_excluded(rng):
    """Candidates tagged ga_eligible=False must be filtered out by update(new_cand=...)."""
    c_good = _make_candidate(["Pt", "Pt"], raw_score=-5.0, confid=1, relax_id=1)
    c_bad = _make_candidate(["Pt", "Pt"], raw_score=-3.0, confid=2, relax_id=2)
    set_tags(c_bad, ga_eligible=False)
    dc = FakeDC([c_good])
    pop = Population(dc, population_size=5, rng=rng)

    pop.update(new_cand=[c_good, c_bad])

    confids = [a.info["confid"] for a in pop.all_cand]
    assert 2 not in confids, "ga_eligible=False candidate must not enter all_cand"


def test_fitness_strategy_population_update_accepts_new_cand(rng):
    """FitnessStrategyPopulation.update must accept new_cand without TypeError."""
    c1 = _make_candidate(["Pt", "Pt"], raw_score=-5.0, confid=1, relax_id=1)
    dc = FakeDC([c1])
    pop = FitnessStrategyPopulation(
        data_connection=dc,
        population_size=3,
        fitness_strategy="low_energy",
        rng=rng,
    )
    c2 = _make_candidate(["Pt", "Pt"], raw_score=-3.0, confid=2, relax_id=2)
    pop.update(new_cand=[c2])
    assert any(a.info["confid"] == 2 for a in pop.all_cand)


def test_fitness_strategy_population_diversity_update_with_new_cand(rng):
    """FitnessStrategyPopulation with DIVERSITY strategy increments generation count."""
    c1 = _make_candidate(["Pt", "Pt"], raw_score=-5.0, confid=1, relax_id=1)
    c1.set_positions([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
    c1.set_cell([10.0, 10.0, 10.0])
    dc = FakeDC([c1])

    comparator = PureInteratomicDistanceComparator(n_top=2)
    # Seed with one reference so the scorer is truthy (DiversityScorer with
    # empty reference_structures evaluates to False).
    scorer = DiversityScorer(reference_structures=[c1], comparator=comparator)
    pop = FitnessStrategyPopulation(
        data_connection=dc,
        population_size=5,
        fitness_strategy="diversity",
        diversity_scorer=scorer,
        diversity_update_interval=1,
        rng=rng,
    )
    assert pop._generation_count == 0

    c2 = _make_candidate(["Pt", "Pt"], raw_score=-3.0, confid=2, relax_id=2)
    c2.set_positions([[0.1, 0.0, 0.0], [2.6, 0.0, 0.0]])
    c2.set_cell([10.0, 10.0, 10.0])
    pop.update(new_cand=[c2])

    assert pop._generation_count == 1


def test_relax_unrelaxed_candidates_ineligible_not_in_pop(tmp_path, rng):
    """Ineligible structures written to DB must stay out of the population."""
    pt3 = Atoms(
        "Pt3",
        positions=[[0, 0, 0], [2.5, 0, 0], [1.25, 2.165, 0]],
        cell=[10] * 3,
    )
    db_path = tmp_path / "ineligible_pop.db"
    prep = create_preparedb(pt3, db_path, population_size=10)
    cand = pt3.copy()
    set_tags(cand, raw_score=-1.0)
    prep.add_unrelaxed_candidate(cand, description="cand:0")

    da = DataConnection(str(db_path))

    class _DisconnectedRelaxer:
        def relax_batch(self, batch):
            return [
                (
                    -999.0,
                    Atoms(
                        symbols=a.get_chemical_symbols(),
                        positions=[[0, 0, 0], [50, 0, 0], [100, 0, 0]],
                        cell=a.get_cell(),
                        pbc=a.get_pbc(),
                    ),
                )
                for a in batch
            ]

    pop = Population(da, population_size=5, rng=rng)
    assert len(pop.pop) == 0

    called_db_read: list[str] = []
    original_garc = da.get_all_relaxed_candidates

    def _track_garc(*args, **kwargs):
        called_db_read.append("garc")
        return original_garc(*args, **kwargs)

    da.get_all_relaxed_candidates = _track_garc

    ga_mod._relax_unrelaxed_candidates(
        da,
        _DisconnectedRelaxer(),
        population=pop,
        composition=["Pt", "Pt", "Pt"],
        system_type="gas_cluster",
    )

    assert len(pop.pop) == 0
    assert not called_db_read


# ---------------------------------------------------------------------------
# O(pop) uniqueness: looks_like counter tests
# ---------------------------------------------------------------------------


class CountingComparator:
    """Comparator that tracks call count and delegates to SymbolsComparator."""

    def __init__(self):
        self.call_count = 0

    def looks_like(self, a, b):
        self.call_count += 1
        return a.get_chemical_symbols() == b.get_chemical_symbols()


def test_looks_like_counter_rules_for_unique_reject_replace(rng):
    """Unique starts at 0; reject increments incumbent; replacement carries L+1."""
    comp = SymbolsComparator()
    base = _make_candidate(["Au", "Pt"], raw_score=-3.0, confid=1, relax_id=1)
    pop = Population(FakeDC([base]), population_size=2, comparator=comp, rng=rng)
    assert pop.pop[0].info.get("looks_like") == 0

    # Rejected duplicate increments incumbent
    reject = _make_candidate(["Au", "Pt"], raw_score=-4.0, confid=2, relax_id=2)
    pop.__add_candidate__(reject)
    assert pop.pop[0].info["looks_like"] == 1
    assert pop.pop[0].info["confid"] == 1

    # Replacement keeps and increments discovery history (L+1)
    pop.pop[0].info["looks_like"] = 3
    replace = _make_candidate(["Au", "Pt"], raw_score=-1.0, confid=3, relax_id=3)
    pop.__add_candidate__(replace)
    assert pop.pop[0].info["confid"] == 3
    assert pop.pop[0].info["looks_like"] == 4


def test_count_looks_like_not_called_on_all_cand(rng):
    """__add_candidate__ must not compare against all_cand (O(pop) only)."""
    comp = CountingComparator()
    # Build a population of size 2
    c1 = _make_candidate(["Au", "Pt"], raw_score=-3.0, confid=1, relax_id=1)
    c2 = _make_candidate(["Pt", "Pt"], raw_score=-5.0, confid=2, relax_id=2)
    pop = Population(FakeDC([c1, c2]), population_size=2, comparator=comp, rng=rng)
    pop_size = len(pop.pop)

    # Pad all_cand with many extra entries (simulating history)
    for i in range(3, 50):
        extra = _make_candidate(["Au", "Pt"], raw_score=-10.0, confid=i, relax_id=i)
        pop.all_cand.append(extra)

    comp.call_count = 0  # reset after init

    new = _make_candidate(["Au", "Pt"], raw_score=-2.0, confid=99, relax_id=99)
    pop.__add_candidate__(new)

    # At most pop_size comparisons (vs each pop member) for the duplicate check
    assert comp.call_count <= pop_size, (
        f"Expected <= {pop_size} comparisons, got {comp.call_count}. "
        "count_looks_like must not scan all_cand."
    )


def test_init_pop_looks_like_counts_duplicates_in_remaining(rng):
    """__initialize_pop__: duplicates beyond pop_size must still increment incumbent."""
    comp = SymbolsComparator()
    # pop_size=1, two identical candidates beyond that
    c1 = _make_candidate(["Au", "Pt"], raw_score=-1.0, confid=1, relax_id=1)
    c2 = _make_candidate(["Au", "Pt"], raw_score=-2.0, confid=2, relax_id=2)
    c3 = _make_candidate(["Au", "Pt"], raw_score=-3.0, confid=3, relax_id=3)

    pop = Population(
        FakeDC([c1, c2, c3]),
        population_size=1,
        comparator=comp,
        rng=rng,
    )
    assert len(pop.pop) == 1
    # Two duplicates were seen (c2 and c3 look like c1)
    assert pop.pop[0].info["looks_like"] == 2
