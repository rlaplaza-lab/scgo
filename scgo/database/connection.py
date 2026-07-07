"""Database connection management for SCGO (HPC-oriented)."""

from __future__ import annotations

import contextlib
import os
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from ase_ga.data import DataConnection

from scgo.utils.logging import get_logger

logger = get_logger(__name__)


def _open_ase_db_backend(backend) -> None:
    """Open a persistent ASE DB backend connection for the current scope.

    ASE's ``managed_connection()`` creates ephemeral SQLite handles when
    ``backend.connection`` is ``None``. Entering the backend context keeps a
    single connection for all subsequent operations and allows reliable cleanup
    via :func:`close_data_connection`.
    """
    if backend is None:
        return
    if getattr(backend, "connection", None) is not None:
        return
    if hasattr(backend, "__enter__"):
        backend.__enter__()


def _unwrap_data_connection(da: DataConnection | object) -> DataConnection:
    """Return the underlying :class:`~ase_ga.data.DataConnection` when wrapped."""
    return getattr(da, "_da", da)


def configure_data_connection_settings(
    da: DataConnection,
    *,
    busy_timeout: int = 30000,
    wal_mode: bool = False,
    cache_size_mb: int = 64,
) -> None:
    """Apply SCGO SQLite settings without leaving ASE's backend connection open.

    ASE's ``with backend:`` context manager requires ``backend.connection`` to be
    ``None`` on entry. Use this for long-lived :class:`~ase_ga.data.DataConnection`
    objects (e.g. from :func:`~scgo.database.helpers.setup_database`) that will
    manage connections via ``with da.c:`` or ``managed_connection()``.
    """
    backend = getattr(da, "c", None)
    if backend is None:
        return

    _open_ase_db_backend(backend)
    conn = getattr(backend, "connection", None)
    try:
        if conn is None:
            return
        _ensure_sqlite_json1(conn=conn)
        apply_sqlite_pragmas(
            conn,
            busy_timeout=busy_timeout,
            cache_size_mb=cache_size_mb,
            wal_mode=wal_mode,
        )
    finally:
        if getattr(backend, "connection", None) is not None:
            with contextlib.suppress(
                sqlite3.OperationalError, sqlite3.DatabaseError, AttributeError
            ):
                backend.__exit__(None, None, None)


def activate_data_connection(
    da: DataConnection,
    *,
    busy_timeout: int = 30000,
    wal_mode: bool = False,
    cache_size_mb: int = 64,
) -> None:
    """Open ASE's backend once and apply SCGO SQLite settings for a scoped session.

    Used by :func:`get_connection` where the caller holds one persistent handle for
    the entire context and closes it via :func:`close_data_connection`.
    """
    configure_data_connection_settings(
        da,
        busy_timeout=busy_timeout,
        wal_mode=wal_mode,
        cache_size_mb=cache_size_mb,
    )

    _open_ase_db_backend(getattr(da, "c", None))
    _apply_busy_timeout(da, busy_timeout)

    conn = getattr(getattr(da, "c", None), "connection", None)
    if conn is not None:
        apply_sqlite_pragmas(
            conn,
            busy_timeout=busy_timeout,
            cache_size_mb=cache_size_mb,
            wal_mode=wal_mode,
        )


def apply_sqlite_pragmas(
    conn: sqlite3.Connection,
    *,
    wal_mode: bool = False,
    busy_timeout: int = 30000,
    cache_size_mb: int = 64,
) -> None:
    """Apply PRAGMAs appropriate for SQLite databases in SCGO.

    Modes:
      wal_mode=False (HPC default): rollback journal, memory temp, delete on close.
      wal_mode=True: write-ahead-logging, normal sync, autocheckpoint.
    """
    if wal_mode:
        with contextlib.suppress(sqlite3.OperationalError):
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute(f"PRAGMA busy_timeout={busy_timeout};")
            conn.execute("PRAGMA temp_store=MEMORY;")
            conn.execute(f"PRAGMA cache_size=-{cache_size_mb * 1024};")
            conn.execute("PRAGMA wal_autocheckpoint=1000;")
    else:
        with contextlib.suppress(sqlite3.OperationalError):
            conn.execute(f"PRAGMA busy_timeout={busy_timeout};")
        with contextlib.suppress(sqlite3.OperationalError):
            conn.execute("PRAGMA journal_mode=DELETE;")
        with contextlib.suppress(sqlite3.OperationalError):
            conn.execute("PRAGMA temp_store=MEMORY;")
        with contextlib.suppress(sqlite3.OperationalError):
            conn.execute(f"PRAGMA cache_size=-{cache_size_mb * 1024};")


@contextmanager
def get_connection(
    db_path: str | Path,
    busy_timeout: int = 30000,
    wal_mode: bool = False,
    cache_size_mb: int = 64,
) -> Generator[DataConnection, None, None]:
    """Open and yield an ASE :class:`~ase_ga.data.DataConnection` (with cleanup on exit).

    This is the primary context manager for SCGO database access.

    WAL mode is off by default (``DELETE`` journal) for shared/HPC filesystems;
    pass ``wal_mode=True`` on local disks when you need more write concurrency.

    Args:
        db_path: Path to the ``.db`` file.
        busy_timeout: SQLite busy timeout in milliseconds (default 30s).
        wal_mode: If True, apply WAL-related PRAGMAs.
        cache_size_mb: SQLite page cache size hint in MiB.
    """
    db_path = str(db_path)
    # Configure SQLite before opening DataConnection
    if wal_mode and os.path.exists(db_path):
        try:
            with sqlite3.connect(db_path, timeout=busy_timeout / 1000.0) as conn:
                apply_sqlite_pragmas(
                    conn,
                    wal_mode=True,
                    busy_timeout=busy_timeout,
                    cache_size_mb=cache_size_mb,
                )
                conn.commit()
        except sqlite3.OperationalError as e:
            logger.warning(f"Failed to configure SQLite for {db_path}: {e}")

    da = DataConnection(db_path)

    activate_data_connection(
        da,
        busy_timeout=busy_timeout,
        wal_mode=wal_mode,
        cache_size_mb=cache_size_mb,
    )

    try:
        yield da
    finally:
        close_data_connection(da)


def close_data_connection(da: DataConnection | None, log_errors: bool = True) -> None:
    """Safely close a DataConnection object.

    Handles the fact that ASE's SQLite3Database doesn't have a close()
    method but does support the context manager protocol (__exit__).

    Note:
        ASE database objects may have their internal SQLite connection invalidated
        (set to None) in certain conditions (errors, timeouts, external closes).
        This is a benign state during cleanup and should not produce error messages.

    Args:
        da: DataConnection object to close (can be None)
        log_errors: Whether to log errors at debug level (default True)

    Example:
        >>> da = DataConnection('path/to/db.db')
        >>> try:
        ...     # work with da
        ... finally:
        ...     close_data_connection(da)
    """
    if da is None:
        return

    da = _unwrap_data_connection(da)
    backend = getattr(da, "c", None)
    if backend is None:
        return

    conn = getattr(backend, "connection", None)
    if conn is None:
        return

    try:
        backend.__exit__(None, None, None)
    except (
        sqlite3.OperationalError,
        sqlite3.DatabaseError,
        TypeError,
        AttributeError,
    ) as e:
        if log_errors:
            logger.debug(f"Error closing database connection: {e}")
        with contextlib.suppress(sqlite3.OperationalError, sqlite3.DatabaseError):
            conn.close()
        backend.connection = None


def _run_sqlite(
    db_path: str | Path,
    callback,
    *,
    timeout: float = 30.0,
    commit: bool = True,
) -> None:
    """Run *callback(conn)* on a short-lived SQLite connection with explicit close."""
    conn = sqlite3.connect(str(db_path), timeout=timeout)
    try:
        callback(conn)
        if commit:
            conn.commit()
    finally:
        with contextlib.suppress(sqlite3.OperationalError, sqlite3.DatabaseError):
            conn.close()


def _ensure_sqlite_json1(
    db_path: str | None = None,
    *,
    conn: sqlite3.Connection | None = None,
) -> None:
    """Ensure the SQLite JSON1 extension is available for this database file.

    Raises RuntimeError with a helpful message if JSON functions (e.g. json_extract)
    are not available on the underlying SQLite build.
    """
    try:
        if conn is not None:
            cur = conn.execute("SELECT json_extract('{\"a\": 1}', '$.a')")
            _ = cur.fetchone()
            return
        if db_path is None:
            raise ValueError("db_path is required when conn is not provided")

        def _probe(active_conn: sqlite3.Connection) -> None:
            cur = active_conn.execute("SELECT json_extract('{\"a\": 1}', '$.a')")
            _ = cur.fetchone()

        _run_sqlite(db_path, _probe, timeout=5.0)
    except sqlite3.OperationalError as e:
        raise RuntimeError(
            "SQLite JSON1 extension is required but not available. "
            "Please use a Python build or system SQLite with JSON1 support (e.g., install a sqlite3 package with JSON1 enabled)."
        ) from e


def _apply_busy_timeout(da, busy_timeout: int) -> None:
    """Apply PRAGMA busy_timeout to the connection used by DataConnection.

    Ensures that even when ASE has already created a connection, we configure it
    for concurrent access (retry on lock instead of failing immediately).
    """
    conn = getattr(getattr(da, "c", None), "connection", None)
    if conn is not None:
        with contextlib.suppress(sqlite3.OperationalError):
            conn.execute(f"PRAGMA busy_timeout={busy_timeout};")
