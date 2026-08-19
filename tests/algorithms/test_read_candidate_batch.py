"""Tests for TorchSim GA unrelaxed-candidate batch reads."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from ase import Atoms
from ase.calculators.emt import EMT
from ase.calculators.singlepoint import SinglePointCalculator
from ase_ga.data import DataConnection

from scgo.algorithms.geneticalgorithm_go_torchsim import _read_candidate_batch
from scgo.metadata.atoms import set_tags
from tests.helpers import create_preparedb


def test_read_candidate_batch_excludes_relaxed_gaid_and_respects_to_take(
    tmp_path: Path,
) -> None:
    pt2 = Atoms("Pt2", positions=[[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
    pt2.center(vacuum=5.0)
    db_path = tmp_path / "candidates.db"
    prep = create_preparedb(pt2, db_path, population_size=4)
    for i in range(2):
        cand = pt2.copy()
        set_tags(cand, raw_score=-float(i + 1))
        prep.add_unrelaxed_candidate(cand, description=f"cand:{i}")

    da = DataConnection(str(db_path))
    assert len(_read_candidate_batch(da, to_take=1)) == 1
    batch = _read_candidate_batch(da, to_take=10)
    assert len(batch) == 2
    gaid = int(batch[0].info["confid"])

    relaxed = pt2.copy()
    relaxed.info["confid"] = gaid
    relaxed.info.setdefault("key_value_pairs", {})
    relaxed.info.setdefault("data", {})
    relaxed.calc = EMT()
    energy = float(relaxed.get_potential_energy())
    relaxed.calc = SinglePointCalculator(relaxed, energy=energy)
    set_tags(relaxed, raw_score=-energy)
    da.add_relaxed_step(relaxed)

    assert list(da.c.select(relaxed=0, gaid=gaid))
    remaining = _read_candidate_batch(da, to_take=10)
    assert len(remaining) == 1
    assert int(remaining[0].info["confid"]) != gaid


def test_read_candidate_batch_none_returns_all_pending(tmp_path: Path) -> None:
    """to_take=None returns every unrelaxed gaid without capping."""
    pt2 = Atoms("Pt2", positions=[[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
    pt2.center(vacuum=5.0)
    db_path = tmp_path / "all_pending.db"
    prep = create_preparedb(pt2, db_path, population_size=10)
    for i in range(5):
        cand = pt2.copy()
        set_tags(cand, raw_score=-float(i + 1))
        prep.add_unrelaxed_candidate(cand, description=f"cand:{i}")

    da = DataConnection(str(db_path))
    batch = _read_candidate_batch(da, to_take=None)
    assert len(batch) == 5


def test_read_candidate_batch_uses_metadata_only_selects(tmp_path: Path) -> None:
    """The exclusion selects for relaxed/queued/pending must use include_data=False."""
    pt2 = Atoms("Pt2", positions=[[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
    pt2.center(vacuum=5.0)
    db_path = tmp_path / "meta.db"
    prep = create_preparedb(pt2, db_path, population_size=4)
    for i in range(2):
        cand = pt2.copy()
        set_tags(cand, raw_score=-float(i + 1))
        prep.add_unrelaxed_candidate(cand, description=f"cand:{i}")

    da = DataConnection(str(db_path))

    original_select = da.c.select
    # Track only our three bulk-exclusion selects: relaxed=1, queued=1, relaxed=0.
    exclusion_select_kwargs: list[dict] = []

    def _spy_select(*args, **kwargs):
        # The three exclusion calls are identified by relaxed= or queued= kwarg
        # and no gaid filter.  ASE's internal re-select uses limit=2 and no such
        # kwarg so it is correctly excluded from this check.
        if ("relaxed" in kwargs or "queued" in kwargs) and "gaid" not in kwargs:
            exclusion_select_kwargs.append(kwargs)
        return original_select(*args, **kwargs)

    with patch.object(da.c, "select", side_effect=_spy_select):
        _read_candidate_batch(da, to_take=10)

    # Must have captured exactly the three exclusion selects.
    assert len(exclusion_select_kwargs) == 3, (
        f"Expected 3 exclusion selects, got {len(exclusion_select_kwargs)}"
    )
    # All three must be metadata-only.
    for kw in exclusion_select_kwargs:
        assert kw.get("include_data") is False, (
            f"Exclusion select called with include_data={kw.get('include_data')} — "
            "expected False for metadata-only reads"
        )


def test_read_candidate_batch_order_sorted_by_gaid(tmp_path: Path) -> None:
    """Returned candidates must be sorted by gaid regardless of insertion order."""
    pt2 = Atoms("Pt2", positions=[[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
    pt2.center(vacuum=5.0)
    db_path = tmp_path / "order.db"
    prep = create_preparedb(pt2, db_path, population_size=10)
    for i in range(4):
        cand = pt2.copy()
        set_tags(cand, raw_score=-float(4 - i))  # insert in reverse score order
        prep.add_unrelaxed_candidate(cand, description=f"cand:{i}")

    da = DataConnection(str(db_path))
    batch = _read_candidate_batch(da, to_take=None)
    confids = [int(a.info["confid"]) for a in batch]
    assert confids == sorted(confids), "batch must be sorted by gaid"


def test_read_candidate_batch_excludes_many_relaxed_gaids(tmp_path: Path) -> None:
    """With many already-relaxed gaids, only genuinely pending ones are returned."""
    pt2 = Atoms("Pt2", positions=[[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
    pt2.center(vacuum=5.0)
    db_path = tmp_path / "many_relaxed.db"
    prep = create_preparedb(pt2, db_path, population_size=20)
    for i in range(8):
        cand = pt2.copy()
        set_tags(cand, raw_score=-float(i + 1))
        prep.add_unrelaxed_candidate(cand, description=f"cand:{i}")

    da = DataConnection(str(db_path))
    # Relax the first 6 gaids.
    all_unrelaxed = _read_candidate_batch(da, to_take=None)
    assert len(all_unrelaxed) == 8
    for cand in all_unrelaxed[:6]:
        gaid = int(cand.info["confid"])
        rel = pt2.copy()
        rel.info["confid"] = gaid
        rel.info["key_value_pairs"] = {"raw_score": -1.0}
        rel.info["data"] = {}
        rel.calc = SinglePointCalculator(rel, energy=-1.0)
        da.add_relaxed_step(rel)

    remaining = _read_candidate_batch(da, to_take=None)
    assert len(remaining) == 2
    remaining_gaids = {int(a.info["confid"]) for a in remaining}
    relaxed_gaids = {int(c.info["confid"]) for c in all_unrelaxed[:6]}
    assert remaining_gaids.isdisjoint(relaxed_gaids)
