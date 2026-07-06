"""Cross-path parity between candidate discovery and canonical DB extract."""

from __future__ import annotations

from ase import Atoms
from ase.db import connect

from scgo.database.helpers import extract_minima_from_database_file
from scgo.database.schema import stamp_scgo_database
from scgo.initialization.candidate_discovery import _load_candidates_from_file


def test_candidate_discovery_matches_extract_minima(tmp_path):
    """Initialization discovery and extract_minima_from_database_file agree."""
    db_path = tmp_path / "test.db"
    with connect(str(db_path)) as db:
        final = Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])
        db.write(
            final,
            relaxed=True,
            key_value_pairs={"raw_score": -10.0, "final_unique_minimum": True},
            gaid=1,
        )
        ts = Atoms("Pt2", positions=[[0, 0, 0], [2.6, 0, 0]])
        db.write(
            ts,
            relaxed=True,
            key_value_pairs={
                "raw_score": -8.0,
                "is_transition_state": True,
            },
            gaid=2,
        )
        non_final = Atoms("Pt2", positions=[[0, 0, 0], [2.7, 0, 0]])
        db.write(non_final, relaxed=True, key_value_pairs={"raw_score": -7.0}, gaid=3)

    stamp_scgo_database(db_path)

    extracted = extract_minima_from_database_file(
        str(db_path), run_id="runx", require_final=False
    )
    discovered = _load_candidates_from_file(str(db_path), db_path.stat().st_mtime)

    assert len(discovered) == len(extracted)
    extracted_sorted = sorted(extracted, key=lambda item: item[0])
    discovered_sorted = sorted(discovered, key=lambda item: item[1])
    for (_, energy, atoms), (exp_energy, exp_atoms) in zip(
        discovered_sorted, extracted_sorted, strict=True
    ):
        assert energy == exp_energy
        assert atoms.get_chemical_symbols() == exp_atoms.get_chemical_symbols()
