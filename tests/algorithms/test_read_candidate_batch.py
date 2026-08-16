"""Tests for TorchSim GA unrelaxed-candidate batch reads."""

from __future__ import annotations

from pathlib import Path

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
