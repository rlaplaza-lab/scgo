"""Tests for shared end-of-run minima collection."""

from __future__ import annotations

from ase import Atoms

from scgo.algorithms.result_collection import collect_run_minima
from scgo.metadata.atoms import set_tags
from scgo.utils.helpers import extract_energy_from_atoms


class _FakeDA:
    def __init__(self, candidates: list[Atoms]) -> None:
        self._candidates = candidates

    def get_all_relaxed_candidates(self) -> list[Atoms]:
        return list(self._candidates)


def _cand(energy: float, *, run_id: str | None = None, ga_eligible: bool | None = None):
    atoms = Atoms("Pt2", positions=[[0, 0, 0], [1.0, 0, 0]])
    set_tags(atoms, raw_score=-energy, potential_energy=energy)
    if run_id is not None:
        set_tags(atoms, run_id=run_id)
    if ga_eligible is not None:
        set_tags(atoms, ga_eligible=ga_eligible)
    return atoms


def test_collect_run_minima_filters_run_id_and_ga_eligible():
    cands = [
        _cand(-1.0, run_id="run_a", ga_eligible=True),
        _cand(-2.0, run_id="run_b", ga_eligible=True),
        _cand(-0.5, run_id="run_a", ga_eligible=False),
        _cand(-3.0, run_id="run_a"),  # untagged ga_eligible
    ]
    minima = collect_run_minima(
        _FakeDA(cands),
        run_id="run_a",
        ga_eligible_default=True,
    )
    assert len(minima) == 2
    energies = {e for e, _ in minima}
    assert energies == {
        extract_energy_from_atoms(cands[0]),
        extract_energy_from_atoms(cands[3]),
    }


def test_collect_run_minima_ga_default_excludes_untagged():
    cands = [
        _cand(-1.0, run_id="run_a", ga_eligible=True),
        _cand(-2.0, run_id="run_a"),
    ]
    minima = collect_run_minima(
        _FakeDA(cands),
        run_id="run_a",
        ga_eligible_default=False,
    )
    assert len(minima) == 1
    assert minima[0][0] == extract_energy_from_atoms(cands[0])
