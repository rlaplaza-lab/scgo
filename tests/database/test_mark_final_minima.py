import json
import sqlite3

from ase import Atoms
from ase.db import connect

from scgo.metadata.db_stamp import stamp_db
from scgo.metadata.persist import mark_final_minima_in_db
from tests.helpers import assert_db_final_row


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

    stamp_db(dbpath)

    atoms = Atoms("Pt2", positions=[[0, 0, 0], [2.6, 0, 0]])
    atoms.info.setdefault("key_value_pairs", {})["run_id"] = run_id
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
    stamp_db(dbpath)

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
    stamp_db(dbpath)

    atoms = Atoms("Pt", positions=[[0, 0, 0]])
    atoms.info.setdefault("key_value_pairs", {})["run_id"] = "r1"
    summary = mark_final_minima_in_db(
        [{"atoms": atoms, "energy": -0.1, "rank": 1, "final_written": "foo.xyz"}],
        base_dir=str(tmp_path),
        db_paths=[str(dbpath)],
    )

    assert summary["rows_updated"] == 0
    assert all(not kv.get("final_unique_minimum") for kv in _iter_system_kvps(dbpath))


def test_mark_final_minima_fallback_scans_all_db(tmp_path):
    """Find DB rows outside the usual run_*/ layout via registry + base_dir scan."""
    from scgo.database.registry import get_registry

    run_id = "run_test_fallback"
    dbdir = tmp_path / "dbs"
    dbdir.mkdir()
    dbpath = dbdir / "other.db"
    final_id = "fallback-fid"
    pt2 = Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])
    with connect(str(dbpath)) as db:
        db.write(
            pt2,
            relaxed=True,
            key_value_pairs={
                "run_id": run_id,
                "raw_score": -3.4,
                "final_id": final_id,
            },
        )
    stamp_db(dbpath)
    get_registry(tmp_path).register_database(dbpath, run_id=run_id)

    atoms = pt2.copy()
    atoms.info.setdefault("key_value_pairs", {})["run_id"] = run_id
    mark_final_minima_in_db(
        [
            {
                "atoms": atoms,
                "energy": None,
                "rank": 1,
                "final_written": "foo.xyz",
                "final_id": final_id,
            }
        ],
        base_dir=str(tmp_path),
    )
    assert_db_final_row(str(dbpath), run_id, expect_final_id=True)


def test_mark_final_minima_accepts_db_paths_and_returns_summary(tmp_path):
    # Create a DB outside the canonical run_* layout
    dbpath = tmp_path / "external.db"
    db = connect(str(dbpath))

    # Write a candidate row with stable final_id/run_id
    db.write(
        Atoms("Pt", positions=[[0, 0, 0]]),
        relaxed=True,
        key_value_pairs={
            "run_id": "run_ext",
            "raw_score": -0.1,
            "final_id": "fid-ext",
        },
    )

    # Prepare final_minima_info matching the above provenance
    atoms = Atoms("Pt", positions=[[0, 0, 0]])
    atoms.info.setdefault("key_value_pairs", {})["run_id"] = "run_ext"

    final_info = [
        {
            "atoms": atoms,
            "energy": -0.1,
            "rank": 1,
            "final_written": "foo.xyz",
            "final_id": "fid-ext",
        }
    ]

    # Call helper with explicit db_paths list (skip registry discovery)
    summary = mark_final_minima_in_db(
        final_info, base_dir=str(tmp_path), db_paths=[str(dbpath)]
    )

    assert isinstance(summary, dict)
    assert summary.get("rows_updated", 0) >= 1
    assert summary.get("dbs_touched", 0) >= 1
    assert str(dbpath) in summary.get("details", {})

    # Verify DB row was updated with final_unique_minimum and that summary matches actual DB
    assert_db_final_row(str(dbpath), "run_ext", expect_final_id=True)

    kv_list = list(_iter_system_kvps(dbpath))
    assert kv_list
    count_tagged = sum(1 for kv in kv_list if kv.get("final_unique_minimum"))
    assert count_tagged >= 1
    # summary should reflect number of rows actually updated
    assert summary.get("rows_updated", 0) == count_tagged
