"""Tests for TS tagging into the correct per-run database."""

from __future__ import annotations

import os

from ase import Atoms

from scgo.metadata.atoms import set_tags
from scgo.ts_search.ts_network import tag_unique_ts_in_databases


def test_tag_unique_ts_uses_source_db_relpath(tmp_path, monkeypatch):
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_a.mkdir()
    run_b.mkdir()
    db_a = str(run_a / "ga_go.db")
    db_b = str(run_b / "ga_go.db")
    open(db_a, "a").close()
    open(db_b, "a").close()

    monkeypatch.setattr(
        "scgo.ts_search.ts_network.list_discovered_db_paths_with_run",
        lambda *a, **k: [(db_a, "run_a"), (db_b, "run_b")],
    )

    written: list[str] = []

    def fake_add_ts_to_database(**kwargs):
        written.append(kwargs["db_file"])
        return True

    monkeypatch.setattr(
        "scgo.ts_search.ts_network.add_ts_to_database",
        fake_add_ts_to_database,
    )

    min_a = Atoms("Pt2", positions=[[0, 0, 0], [1, 0, 0]])
    min_b = Atoms("Pt2", positions=[[0, 0, 0], [1.1, 0, 0]])
    set_tags(
        min_a,
        source_db="ga_go.db",
        source_db_relpath=os.path.relpath(db_b, tmp_path),
        run_id="run_b",
        confid=1,
    )
    set_tags(
        min_b,
        source_db="ga_go.db",
        source_db_relpath=os.path.relpath(db_b, tmp_path),
        run_id="run_b",
        confid=2,
    )
    minima = [(-1.0, min_a), (-0.9, min_b)]
    ts_atoms = Atoms("Pt2", positions=[[0, 0, 0], [1.05, 0, 0]])
    unique_ts = [
        {
            "ts_energy": -0.5,
            "_atoms_obj": ts_atoms,
            "filename": "ts_0_1.xyz",
            "connected_edges": [
                {
                    "pair_id": "0_1",
                    "minima_indices": [0, 1],
                    "barrier_height": 0.4,
                    "neb_converged": True,
                    "minima_provenance": None,
                }
            ],
        }
    ]

    added = tag_unique_ts_in_databases(unique_ts, minima, str(tmp_path))
    assert added == 1
    assert written == [db_b]
