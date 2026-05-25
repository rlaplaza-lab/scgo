import json
import sqlite3

import numpy as np
from ase import Atoms
from ase.db import connect

from scgo.database.metadata import mark_final_minima_in_db
from scgo.database.schema import stamp_scgo_database
from scgo.utils.helpers import compute_final_id
from tests.test_utils import assert_db_final_row


def _iter_system_kvps(db_path):
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.cursor()
        cur.execute("SELECT key_value_pairs FROM systems")
        for (kvp_json,) in cur.fetchall():
            yield json.loads(kvp_json) if kvp_json else {}


def test_mark_final_minima_prefers_final_id(tmp_path):
    run_id = "run_final_id"
    trial = 1
    dbpath = tmp_path / "fid.db"
    with connect(str(dbpath)) as db:
        db.write(
            Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]]),
            relaxed=True,
            key_value_pairs={"run_id": run_id, "trial_id": trial, "raw_score": -0.5},
        )
        db.write(
            Atoms("Pt2", positions=[[0, 0, 0], [2.6, 0, 0]]),
            relaxed=True,
            key_value_pairs={
                "run_id": run_id,
                "trial_id": trial,
                "raw_score": -0.5,
                "final_id": "persisted-fid",
            },
        )

    stamp_scgo_database(dbpath)

    atoms = Atoms("Pt2", positions=[[0, 0, 0], [2.6, 0, 0]])
    atoms.info["provenance"] = {"run_id": run_id, "trial_id": trial}
    final_info = [
        {
            "atoms": atoms,
            "energy": -0.5,
            "rank": 1,
            "final_written": "foo.xyz",
            "final_id": "persisted-fid",
        }
    ]

    from scgo.database.registry import get_registry

    get_registry(tmp_path).register_database(dbpath, run_id=run_id, trial_id=trial)
    mark_final_minima_in_db(final_info, base_dir=str(tmp_path))

    assert_db_final_row(str(dbpath), run_id, expect_final_id=True)


def test_mark_final_minima_prefers_relaxed_row_when_final_id_duplicated(tmp_path):
    dbpath = tmp_path / "dup.db"
    with connect(str(dbpath)) as db:
        db.write(
            Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]]),
            relaxed=False,
            key_value_pairs={
                "final_id": "dup-fid",
                "relaxed": False,
                "raw_score": -1.0,
            },
        )
        db.write(
            Atoms("Pt2", positions=[[0, 0, 0], [2.6, 0, 0]]),
            relaxed=True,
            key_value_pairs={"final_id": "dup-fid", "relaxed": True, "raw_score": -2.0},
        )
    stamp_scgo_database(dbpath)

    atoms = Atoms("Pt2", positions=[[0, 0, 0], [2.6, 0, 0]])
    final_info = [
        {
            "atoms": atoms,
            "energy": None,
            "rank": 1,
            "final_written": "foo.xyz",
            "final_id": "dup-fid",
        }
    ]
    from scgo.database.registry import get_registry

    get_registry(tmp_path).register_database(dbpath)
    mark_final_minima_in_db(final_info, base_dir=str(tmp_path))

    with sqlite3.connect(str(dbpath)) as conn:
        rows = conn.execute("SELECT key_value_pairs FROM systems").fetchall()
    assert any(
        (json.loads(r[0]) or {}).get("relaxed")
        and (json.loads(r[0]) or {}).get("final_unique_minimum")
        for r in rows
    )


def test_mark_final_minima_skips_entries_without_final_id(tmp_path):
    dbpath = tmp_path / "no-final-id.db"
    with connect(str(dbpath)) as db:
        db.write(
            Atoms("Pt", positions=[[0, 0, 0]]),
            relaxed=True,
            key_value_pairs={"run_id": "r1", "trial_id": 1, "raw_score": -0.1},
        )
    stamp_scgo_database(dbpath)

    atoms = Atoms("Pt", positions=[[0, 0, 0]])
    atoms.info["provenance"] = {"run_id": "r1", "trial_id": 1}
    summary = mark_final_minima_in_db(
        [{"atoms": atoms, "energy": -0.1, "rank": 1, "final_written": "foo.xyz"}],
        base_dir=str(tmp_path),
        db_paths=[str(dbpath)],
    )

    assert summary["rows_updated"] == 0
    assert all(not kv.get("final_unique_minimum") for kv in _iter_system_kvps(dbpath))


def test_mark_final_minima_tags_all_when_final_id_from_db_geometry(tmp_path):
    """final_id must match relaxed DB rows, not write-frame-aligned coordinates."""
    run_id = "run_align"
    trial = 1
    dbpath = tmp_path / "align.db"
    energies = [-1.0, -0.8, -0.6]
    db_positions = [
        [[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [2.7, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [2.9, 0.0, 0.0]],
    ]
    with connect(str(dbpath)) as db:
        for pos, energy in zip(db_positions, energies, strict=True):
            db.write(
                Atoms("Pt2", positions=pos),
                relaxed=True,
                key_value_pairs={
                    "run_id": run_id,
                    "trial_id": trial,
                    "raw_score": -energy,
                },
            )
    stamp_scgo_database(dbpath)

    final_info = []
    for rank, (pos, energy) in enumerate(
        zip(db_positions, energies, strict=True), start=1
    ):
        atoms = Atoms("Pt2", positions=pos)
        atoms.info["provenance"] = {"run_id": run_id, "trial_id": trial}
        # Simulate write-time frame change (e.g. slab remap): geometry differs from DB row.
        write_copy = atoms.copy()
        write_copy.positions[1] += np.array([0.4, 0.0, 0.0])
        wrong_id = compute_final_id(write_copy, energy)
        correct_id = compute_final_id(atoms, energy)
        assert wrong_id != correct_id
        final_info.append(
            {
                "atoms": atoms,
                "energy": energy,
                "rank": rank,
                "final_written": f"minimum_{rank:02d}.xyz",
                "final_id": correct_id,
            }
        )

    from scgo.database.registry import get_registry

    get_registry(tmp_path).register_database(dbpath, run_id=run_id, trial_id=trial)
    summary = mark_final_minima_in_db(final_info, base_dir=str(tmp_path))

    assert summary["rows_updated"] == len(final_info)
    tagged = sum(
        1 for kv in _iter_system_kvps(dbpath) if kv.get("final_unique_minimum")
    )
    assert tagged == len(final_info)
