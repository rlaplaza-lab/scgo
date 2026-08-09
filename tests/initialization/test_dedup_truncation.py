"""Regression tests for seed-candidate truncation ordering (bug 1.3.5).

``deduplicate_seed_candidates`` documents its result as *unordered* (entries come
out of energy-bin buckets and signature dicts). ``_find_smaller_candidates``
truncated that result to ``_MAX_CANDIDATES_PER_FORMULA`` directly, so an
arbitrary subset of rows survived instead of the lowest-energy ones. The entries
are now re-sorted by energy before truncation.
"""

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.db import connect

from scgo.database import get_global_cache
from scgo.initialization import candidate_discovery
from scgo.initialization.candidate_discovery import (
    _find_smaller_candidates,
    deduplicate_seed_candidates,
)
from scgo.initialization.initialization_config import _COMPOSITION_CACHE_NS
from scgo.metadata.db_stamp import stamp_db

# raw_score values written to the DB; the discovery energy is -raw_score.
PT2_RAW_SCORES = [-1.0, -5.0, -3.0, -2.0, -4.0]


def _pt2_geometry(index: int) -> Atoms:
    """Return a distinct (non-duplicate) Pt2 geometry."""
    return Atoms("Pt2", positions=[[0.0, 0.0, 0.0], [2.2 + 0.3 * index, 0.0, 0.0]])


def _write_discovery_db(tmp_path: Path) -> str:
    """Write a discovery database with several distinct Pt2 minima."""
    db_path = tmp_path / "Pt2_searches" / "run_001" / "cluster.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with connect(str(db_path)) as db:
        for i, raw_score in enumerate(PT2_RAW_SCORES):
            db.write(
                _pt2_geometry(i),
                relaxed=True,
                key_value_pairs={
                    "raw_score": raw_score,
                    "final_unique_minimum": True,
                },
                gaid=i + 1,
            )

    stamp_db(db_path)
    return str(db_path)


@pytest.fixture(autouse=True)
def _clear_composition_cache():
    """Keep the discovery cache from leaking between these tests."""
    get_global_cache().clear_namespace(_COMPOSITION_CACHE_NS)
    yield
    get_global_cache().clear_namespace(_COMPOSITION_CACHE_NS)


class TestSeedCandidateTruncation:
    """Truncation must retain the lowest-energy candidates."""

    def test_truncation_keeps_lowest_energy_rows(self, tmp_path, monkeypatch):
        """Arbitrary dedup ordering does not leak into the truncated result."""
        db_path = _write_discovery_db(tmp_path)

        real_dedup = candidate_discovery.deduplicate_seed_candidates

        def _shuffled_dedup(entries, *args, **kwargs):
            """Emulate the documented (unordered) dedup contract."""
            deduped = real_dedup(entries, *args, **kwargs)
            return list(reversed(deduped))

        monkeypatch.setattr(
            candidate_discovery, "deduplicate_seed_candidates", _shuffled_dedup
        )
        monkeypatch.setattr(candidate_discovery, "_MAX_CANDIDATES_PER_FORMULA", 2)

        candidates = _find_smaller_candidates(["Pt", "Pt", "Pt"], db_path)

        assert "Pt2" in candidates
        energies = [energy for energy, _ in candidates["Pt2"]]
        assert len(energies) == 2

        all_energies = sorted(-score for score in PT2_RAW_SCORES)
        assert energies == pytest.approx(all_energies[:2])

    def test_untruncated_results_are_energy_sorted(self, tmp_path):
        """Discovery results come back in ascending energy order."""
        db_path = _write_discovery_db(tmp_path)

        candidates = _find_smaller_candidates(["Pt", "Pt", "Pt"], db_path)

        energies = [energy for energy, _ in candidates["Pt2"]]
        assert energies == sorted(energies)
        assert energies == pytest.approx(sorted(-score for score in PT2_RAW_SCORES))


class TestDeduplicateSeedCandidates:
    """Sanity checks on the deduplication helper used before truncation."""

    def test_dedup_keeps_all_distinct_geometries(self):
        """Distinct geometries survive deduplication."""
        entries = [
            (float(-score), _pt2_geometry(i)) for i, score in enumerate(PT2_RAW_SCORES)
        ]
        deduped = deduplicate_seed_candidates(entries)
        assert len(deduped) == len(entries)

    def test_dedup_collapses_identical_geometries(self):
        """Identical geometries in the same energy bin collapse to one entry."""
        entries = [(1.0, _pt2_geometry(0)), (1.0, _pt2_geometry(0))]
        deduped = deduplicate_seed_candidates(entries, energy_bin=0.0)
        assert len(deduped) == 1

    def test_lowest_energy_rows_after_sorting(self):
        """Sorting the dedup output by energy keeps the lowest-energy rows."""
        rng = np.random.default_rng(0)
        order = rng.permutation(len(PT2_RAW_SCORES))
        entries = [(float(-PT2_RAW_SCORES[i]), _pt2_geometry(int(i))) for i in order]

        deduped = sorted(deduplicate_seed_candidates(entries), key=lambda e: e[0])
        assert [energy for energy, _ in deduped[:2]] == pytest.approx(
            sorted(-score for score in PT2_RAW_SCORES)[:2]
        )
