"""Database health check utilities for SCGO.

Provides tools to diagnose and validate ASE database files.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from scgo.utils.logging import get_logger

logger = get_logger(__name__)


def check_database_health(db_path: str | Path) -> dict:
    """Check database health and return diagnostic information.

    Checks for:
    - File existence and read permission
    - SQLite corruption (``PRAGMA integrity_check``)
    - Presence of the ASE ``systems`` table
    - Journal mode (WAL or rollback journal)
    - Database size and row/table counts

    Args:
        db_path: Path to database file

    Returns:
        dict: Health check results with keys:
            - 'healthy': bool
            - 'errors': list of error messages
            - 'warnings': list of warning messages
            - 'info': dict of database statistics (empty when the file is
              missing or unreadable, since those checks return early)
    """
    db_path = Path(db_path)
    result = {"healthy": True, "errors": [], "warnings": [], "info": {}}

    # Check file existence
    if not db_path.exists():
        result["healthy"] = False
        result["errors"].append(f"Database file does not exist: {db_path}")
        return result

    # Check file permissions
    if not os.access(db_path, os.R_OK):
        result["healthy"] = False
        result["errors"].append(f"Database file is not readable: {db_path}")
        return result

    # Check SQLite integrity
    try:
        conn = sqlite3.connect(str(db_path), timeout=10.0)
        try:
            cur = None
            try:
                # Run integrity check
                cur = conn.execute("PRAGMA integrity_check;")
                integrity_result = cur.fetchone()[0]
                cur.close()
                cur = None

                if integrity_result != "ok":
                    result["healthy"] = False
                    result["errors"].append(
                        f"Integrity check failed: {integrity_result}"
                    )

                # Get database size
                cur = conn.execute("PRAGMA page_count;")
                page_count = cur.fetchone()[0]
                cur.close()
                cur = conn.execute("PRAGMA page_size;")
                page_size = cur.fetchone()[0]
                cur.close()
                cur = None
                db_size_mb = (page_count * page_size) / (1024 * 1024)
                result["info"]["size_mb"] = round(db_size_mb, 2)

                # Check journal mode
                cur = conn.execute("PRAGMA journal_mode;")
                journal_mode = cur.fetchone()[0]
                cur.close()
                cur = None
                result["info"]["journal_mode"] = journal_mode

                # Get table count
                cur = conn.execute(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table';"
                )
                table_count = cur.fetchone()[0]
                cur.close()
                cur = None
                result["info"]["table_count"] = table_count

                # Check for ASE tables
                cur = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='systems';"
                )
                has_systems_table = cur.fetchone() is not None
                cur.close()
                cur = None

                if not has_systems_table:
                    result["warnings"].append(
                        "Missing 'systems' table - may not be an ASE database"
                    )
                else:
                    # Count rows in systems table
                    cur = conn.execute("SELECT COUNT(*) FROM systems;")
                    row_count = cur.fetchone()[0]
                    cur.close()
                    cur = None
                    result["info"]["systems_count"] = row_count
            finally:
                if cur is not None:
                    cur.close()
        finally:
            conn.close()

    except (sqlite3.DatabaseError, sqlite3.OperationalError, OSError) as e:
        result["healthy"] = False
        result["errors"].append(f"Database error: {e}")

    return result


def get_database_statistics(db_path: str | Path) -> dict:
    """Get detailed statistics about a database.

    Args:
        db_path: Path to database file

    Returns:
        dict: Database statistics including:
            - page_count: Number of SQLite pages
            - page_size: SQLite page size
            - size_mb: Database size in megabytes
            - journal_mode: Current journal mode
            - freelist_count: Number of free pages
            - fragmentation_pct: Free pages as a percentage of ``page_count``
            - systems_count: Number of rows in the ``systems`` table, or None
              when that table is missing
            - tables: Sorted list of table names

        The dict is empty when the file does not exist, and may be partial
        when a SQLite or OS error interrupts collection.
    """
    db_path = Path(db_path)
    stats = {}

    if not db_path.exists():
        logger.warning("Database does not exist: %s", db_path)
        return stats

    try:
        conn = sqlite3.connect(str(db_path), timeout=10.0)
        try:
            cur = None
            try:
                # Basic size info
                cur = conn.execute("PRAGMA page_count;")
                page_count = cur.fetchone()[0]
                cur.close()
                cur = conn.execute("PRAGMA page_size;")
                page_size = cur.fetchone()[0]
                cur.close()
                cur = None
                stats["page_count"] = page_count
                stats["page_size"] = page_size
                stats["size_mb"] = round((page_count * page_size) / (1024 * 1024), 2)

                # Journal mode
                cur = conn.execute("PRAGMA journal_mode;")
                stats["journal_mode"] = cur.fetchone()[0]
                cur.close()
                cur = None

                # Freelist (fragmentation indicator)
                cur = conn.execute("PRAGMA freelist_count;")
                freelist = cur.fetchone()[0]
                cur.close()
                cur = None
                stats["freelist_count"] = freelist
                if page_count > 0:
                    stats["fragmentation_pct"] = round((freelist / page_count) * 100, 2)

                # Count systems
                try:
                    cur = conn.execute("SELECT COUNT(*) FROM systems;")
                    stats["systems_count"] = cur.fetchone()[0]
                    cur.close()
                    cur = None
                except sqlite3.OperationalError:
                    stats["systems_count"] = None

                # Get table names
                cur = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
                )
                stats["tables"] = [row[0] for row in cur.fetchall()]
                cur.close()
                cur = None
            finally:
                if cur is not None:
                    cur.close()
        finally:
            conn.close()

    except (sqlite3.DatabaseError, sqlite3.OperationalError, OSError) as e:
        logger.error("Failed to get statistics for %s: %s", db_path, e)

    return stats
