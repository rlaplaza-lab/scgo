import sqlite3
from pathlib import Path

import pytest

from scgo.database.discovery import DatabaseDiscovery
from scgo.database.streaming import iter_database_minima
from scgo.metadata.db_stamp import get_db_stamp, is_scgo_db


def _create_dummy_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        # Minimal table to emulate an ASE DB file (no scgo_metadata)
        conn.execute(
            "CREATE TABLE systems (id INTEGER PRIMARY KEY, energy REAL, key_value_pairs TEXT)"
        )
        conn.commit()


def test_get_db_stamp_returns_empty_for_non_scgo_db(tmp_path: Path):
    run_dir = tmp_path / "run_000"
    db_path = run_dir / "ga_go.db"
    _create_dummy_db(db_path)

    assert get_db_stamp(db_path) == {}
    assert not is_scgo_db(db_path)


def test_stamp_db_invalidates_is_scgo_db_cache(tmp_path: Path):
    from scgo.metadata.db_stamp import stamp_db

    run_dir = tmp_path / "run_000"
    db_path = run_dir / "ga_go.db"
    _create_dummy_db(db_path)

    assert not is_scgo_db(db_path)  # caches False
    stamp_db(db_path)
    assert is_scgo_db(db_path)  # must not keep stale False


def test_find_databases_skips_non_scgo_db(tmp_path: Path):
    run_dir = tmp_path / "run_000"
    db_path = run_dir / "ga_go.db"
    _create_dummy_db(db_path)

    discovery = DatabaseDiscovery(tmp_path)
    found = discovery.find_databases(db_filename="*.db")
    # Should skip the non-SCGO DB created above
    assert db_path not in found
    assert found == []


def test_iter_database_minima_skips_non_scgo_db(tmp_path: Path):
    run_dir = tmp_path / "run_000"
    db_path = run_dir / "ga_go.db"
    _create_dummy_db(db_path)

    items = list(iter_database_minima(db_path))
    assert items == []


def test_setup_database_marks_scgo_db(tmp_path: Path):
    from ase import Atoms

    from scgo.database.connection import close_data_connection
    from scgo.database.helpers import setup_database

    run_dir = tmp_path / "run_000"
    template = Atoms(["Pt", "Pt"], positions=[(0, 0, 0), (0, 0, 1)])

    da = setup_database(run_dir, "ga_go.db", template, initial_candidate=template)
    try:
        db_path = run_dir / "ga_go.db"
        meta = get_db_stamp(db_path)
        assert meta.get("created_by") == "scgo"
        assert "schema_version" in meta and int(meta["schema_version"]) >= 1
    finally:
        # Ensure resources cleaned up
        close_data_connection(da)


def test_setup_database_raises_when_stamp_fails(tmp_path: Path, monkeypatch):
    from ase import Atoms

    from scgo.database.exceptions import DatabaseSetupError
    from scgo.database.helpers import setup_database

    run_dir = tmp_path / "run_stamp_fail"
    template = Atoms(["Pt", "Pt"], positions=[(0, 0, 0), (0, 0, 1)])

    def _fail_stamp(_path):
        raise OSError("simulated stamp failure")

    monkeypatch.setattr("scgo.database.helpers.stamp_db", _fail_stamp)
    with pytest.raises(DatabaseSetupError, match="Failed to stamp"):
        setup_database(run_dir, "ga_go.db", template, initial_candidate=template)


def test_load_previous_run_results_skips_corrupt_db(tmp_path: Path, monkeypatch):
    """Multi-file load continues when one extract raises SCGODatabaseError."""
    from ase import Atoms

    from scgo.database.helpers import (
        extract_minima_from_database_file,
        load_previous_run_results,
        setup_database,
    )
    from scgo.exceptions import SCGODatabaseError

    good_run = tmp_path / "run_good"
    bad_run = tmp_path / "run_bad"
    template = Atoms(["Pt", "Pt"], positions=[(0, 0, 0), (0, 0, 2.5)])
    da = setup_database(good_run, "ga_go.db", template, initial_candidate=template)
    try:
        a = template.copy()
        a.info["data"] = {}
        a.info["key_value_pairs"] = {
            "raw_score": -1.0,
            "final_unique_minimum": True,
            "run_id": "run_good",
        }
        da.add_relaxed_step(a)
    finally:
        from scgo.database.connection import close_data_connection

        close_data_connection(da)

    bad_run.mkdir(parents=True)
    (bad_run / "ga_go.db").write_bytes(b"not-a-sqlite-db")

    real_extract = extract_minima_from_database_file

    def _flaky(db_path, run_id, **kwargs):
        if "run_bad" in str(db_path):
            raise SCGODatabaseError("simulated corrupt extract")
        return real_extract(db_path, run_id, **kwargs)

    monkeypatch.setattr(
        "scgo.database.helpers.extract_minima_from_database_file", _flaky
    )
    minima = load_previous_run_results(str(tmp_path), composition=["Pt", "Pt"])
    assert isinstance(minima, list)


def test_find_databases_includes_scgo_db(tmp_path: Path):
    from ase import Atoms

    from scgo.database.helpers import setup_database

    run_dir = tmp_path / "run_000"
    template = Atoms(["Pt", "Pt"], positions=[(0, 0, 0), (0, 0, 1)])

    # Create a proper SCGO DB
    da = setup_database(run_dir, "ga_go.db", template, initial_candidate=template)
    try:
        discovery = DatabaseDiscovery(tmp_path)
        found = discovery.find_databases(db_filename="*.db")
        assert (run_dir / "ga_go.db") in found
    finally:
        from scgo.database.connection import close_data_connection

        close_data_connection(da)
