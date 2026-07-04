import json
import sqlite3

from ase import Atoms
from ase.db import connect

from scgo.database.metadata import mark_final_minima_in_db
from scgo.database.schema import stamp_scgo_database
from tests.test_utils import assert_db_final_row


def _iter_system_kvps(db_path):
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.cursor()
        cur.execute("SELECT key_value_pairs FROM systems")
        for (kvp_json,) in cur.fetchall():
            yield json.loads(kvp_json) if kvp_json else {}


def test_mark_final_minima_prefers_final_id(tmp_path):
    run_id = "run_final_id"
    dbpath = tmp_path / "fid.db"
    with connect(str(dbpath)) as db:
        db.write(
            Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]]),
            relaxed=True,
            key_value_pairs={"run_id": run_id, "raw_score": -0.5},
        )
        db.write(
            Atoms("Pt2", positions=[[0, 0, 0], [2.6, 0, 0]]),
            relaxed=True,
            key_value_pairs={
                "run_id": run_id,
                "raw_score": -0.5,
                "final_id": "persisted-fid",
            },
        )

    stamp_scgo_database(dbpath)

    atoms = Atoms("Pt2", positions=[[0, 0, 0], [2.6, 0, 0]])
    atoms.info["provenance"] = {"run_id": run_id}
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

    get_registry(tmp_path).register_database(dbpath, run_id=run_id)
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
            key_value_pairs={"run_id": "r1", "raw_score": -0.1},
        )
    stamp_scgo_database(dbpath)

    atoms = Atoms("Pt", positions=[[0, 0, 0]])
    atoms.info["provenance"] = {"run_id": "r1"}
    summary = mark_final_minima_in_db(
        [{"atoms": atoms, "energy": -0.1, "rank": 1, "final_written": "foo.xyz"}],
        base_dir=str(tmp_path),
        db_paths=[str(dbpath)],
    )

    assert summary["rows_updated"] == 0
    assert all(not kv.get("final_unique_minimum") for kv in _iter_system_kvps(dbpath))
