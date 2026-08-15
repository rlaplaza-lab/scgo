"""Simple transaction helpers for SCGO databases."""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager

from ase_ga.data import DataConnection

from scgo.exceptions import (
    SCGOValidationError,
)
from scgo.utils.logging import get_logger

logger = get_logger(__name__)


_VALID_ISOLATION_LEVELS = frozenset({"DEFERRED", "IMMEDIATE", "EXCLUSIVE"})


@contextmanager
def database_transaction(
    db: DataConnection,
    isolation_level: str = "DEFERRED",
) -> Generator[sqlite3.Connection, None, None]:
    """Context manager for a transaction.

    When the underlying connection already has a transaction open (ASE keeps one
    pending after its own writes), the context manager *joins* it: no ``BEGIN``
    is issued and commit/rollback are left to the owner of the outer
    transaction. Only a transaction started here is committed or rolled back
    here.

    Args:
        db: ASE ``DataConnection`` whose backend (``db.c``) exposes
            ``managed_connection()``.
        isolation_level: ``DEFERRED``, ``IMMEDIATE``, or ``EXCLUSIVE``
            (case-insensitive).

    Yields:
        sqlite3.Connection: Raw connection. Commits on success and rolls back on
        error, unless an outer transaction was already open.

    Raises:
        SCGOValidationError: If ``db`` has no usable connection, or if
            ``isolation_level`` is not a valid SQLite isolation level.
    """
    if not hasattr(db, "c") or db.c is None:
        raise SCGOValidationError("Invalid database connection")

    if isolation_level.upper() not in _VALID_ISOLATION_LEVELS:
        raise SCGOValidationError(
            f"Invalid isolation level: {isolation_level!r}. "
            f"Must be one of {sorted(_VALID_ISOLATION_LEVELS)}"
        )

    # Use managed_connection() to get actual SQLite connection
    with db.c.managed_connection() as conn:
        started = not conn.in_transaction
        try:
            if started:
                conn.execute(f"BEGIN {isolation_level.upper()}")
                logger.debug("Started %s transaction", isolation_level)
            else:
                logger.debug(
                    "Joining transaction already open on the connection; "
                    "commit/rollback is left to its owner"
                )

            yield conn  # Yield connection instead of db

            if started:
                conn.commit()
                logger.debug("Transaction committed")
        except Exception:  # broad by design: rollback then re-raise the original error
            if started:
                conn.rollback()
                logger.debug("Transaction rolled back")
            raise
