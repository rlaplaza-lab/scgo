"""Tests for consolidated database discovery."""

from __future__ import annotations

from pathlib import Path

from scgo.database.discovery import DatabaseDiscovery, discover_db_files
from scgo.metadata.db_stamp import stamp_db


def _touch_scgo_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    stamp_db(path)


def test_nested_db_under_run_is_discovered(tmp_path):
    nested = tmp_path / "run_0001" / "subdir" / "ga_go.db"
    _touch_scgo_db(nested)
    via_helper = discover_db_files(
        tmp_path,
        db_filename="*.db",
        merge_registry=False,
        require_scgo=True,
        nested_under_run=True,
    )
    via_discovery = DatabaseDiscovery(tmp_path).find_databases()
    resolved = nested.resolve()
    assert resolved in {p.resolve() for p in via_helper}
    assert resolved in {p.resolve() for p in via_discovery}
