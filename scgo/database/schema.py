"""Schema version helpers for SCGO."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from ase_ga.data import DataConnection

from scgo.database.connection import _run_sqlite
from scgo.database.exceptions import DatabaseMigrationError
from scgo.utils.logging import get_logger

logger = get_logger(__name__)

CURRENT_SCHEMA_VERSION = 2

SCGO_METADATA_DDL = """
CREATE TABLE IF NOT EXISTS scgo_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
)
"""


def _upsert_scgo_metadata_keys(
    conn: sqlite3.Connection, *, schema_version: int
) -> None:
    conn.execute(SCGO_METADATA_DDL)
    conn.execute(
        "INSERT OR REPLACE INTO scgo_metadata (key, value) VALUES ('created_by', 'scgo')"
    )
    conn.execute(
        "INSERT OR REPLACE INTO scgo_metadata (key, value) VALUES ('schema_version', ?)",
        (str(schema_version),),
    )


def get_schema_version(db: DataConnection) -> int:
    """Get current schema version from database.

    Args:
        db: Database connection

    Returns:
        int: Current schema version (0 if not set)
    """
    try:
        with db.c.managed_connection() as conn:
            cursor = conn.execute(
                "SELECT value FROM scgo_metadata WHERE key='schema_version'"
            )
            result = cursor.fetchone()
            return int(result[0]) if result else 0
    except sqlite3.OperationalError:
        # Table doesn't exist - schema version 0
        return 0


def set_schema_version(db: DataConnection, version: int) -> None:
    """Set schema version in database.

    Args:
        db: Database connection
        version: Schema version to set
    """
    with db.c.managed_connection() as conn:
        _upsert_scgo_metadata_keys(conn, schema_version=version)
        conn.commit()
        logger.debug(f"Set schema version to {version}")


def migrate_database(db: DataConnection, target_version: int | None = None) -> bool:
    """Set ``schema_version`` in ``scgo_metadata`` to *target_version* (no data migration).

    Returns True on success; raises DatabaseMigrationError for downgrades or failure.
    """
    if target_version is None:
        target_version = CURRENT_SCHEMA_VERSION

    current_version = get_schema_version(db)

    if current_version == target_version:
        logger.debug(f"Database already at version {target_version}")
        return True

    if current_version > target_version:
        raise DatabaseMigrationError(
            f"Cannot downgrade from version {current_version} to {target_version}"
        )

    try:
        set_schema_version(db, target_version)
        logger.info(
            f"Marked database schema version as {target_version} (no migrations applied)"
        )
        return True
    except (OSError, sqlite3.Error, TypeError, ValueError) as e:
        logger.error(f"Failed to set schema version to {target_version}: {e}")
        raise DatabaseMigrationError(f"Failed to set schema version: {e}") from e


def ensure_schema_version(db: DataConnection) -> None:
    """Bump recorded schema version to :data:`CURRENT_SCHEMA_VERSION` when behind."""
    current_version = get_schema_version(db)

    if current_version < CURRENT_SCHEMA_VERSION:
        logger.info(
            f"Database needs migration from v{current_version} to v{CURRENT_SCHEMA_VERSION}"
        )
        migrate_database(db, CURRENT_SCHEMA_VERSION)
    elif current_version > CURRENT_SCHEMA_VERSION:
        logger.warning(
            f"Database version {current_version} is newer than expected "
            f"{CURRENT_SCHEMA_VERSION}. Update SCGO to latest version."
        )


def get_scgo_metadata(db_path: str | Path) -> dict[str, str]:
    """Return key/value pairs from the `scgo_metadata` table for the DB, or {}.

    This performs a lightweight, read-only check and returns an empty dict for
    non-SCGO databases or on error.
    """
    try:
        db_file = str(db_path)
        # Open read-only to avoid creating files or locking where possible.
        with sqlite3.connect(f"file:{db_file}?mode=ro", uri=True, timeout=0.1) as conn:
            cur = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='scgo_metadata'"
            )
            if cur.fetchone() is None:
                return {}
            rows = conn.execute("SELECT key, value FROM scgo_metadata").fetchall()
            return {r[0]: r[1] for r in rows}
    except (sqlite3.OperationalError, sqlite3.DatabaseError, FileNotFoundError) as exc:
        logger.debug("Could not read scgo_metadata from %s: %s", db_path, exc)
        return {}


_scgo_database_cache: dict[str, bool] = {}


def clear_scgo_database_cache() -> None:
    """Clear the :func:`is_scgo_database` memoization cache."""
    _scgo_database_cache.clear()


def is_scgo_database(db_path: str | Path) -> bool:
    """True if ``scgo_metadata.created_by`` is ``scgo``."""
    key = str(Path(db_path).resolve())
    cached = _scgo_database_cache.get(key)
    if cached is not None:
        return cached
    meta = get_scgo_metadata(db_path)
    result = bool(meta) and meta.get("created_by") == "scgo"
    _scgo_database_cache[key] = result
    return result


def stamp_scgo_database(
    db_path: str | Path, *, schema_version: int | None = None
) -> None:
    """Write ``scgo_metadata`` so :func:`is_scgo_database` accepts this file.

    Used by tests and tools that build SQLite files outside :func:`setup_database`.
    """
    ver = schema_version if schema_version is not None else CURRENT_SCHEMA_VERSION
    path = str(db_path)

    def _stamp(conn: sqlite3.Connection) -> None:
        _upsert_scgo_metadata_keys(conn, schema_version=ver)

    _run_sqlite(path, _stamp)
    clear_scgo_database_cache()
