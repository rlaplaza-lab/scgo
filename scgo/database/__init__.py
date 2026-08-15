"""SCGO Database Module.

Designed for **HPC** use: SQLite on shared filesystems (Lustre, GPFS, NFS-class),
batch jobs, and optional multi-process access. WAL mode is off by default.
Database discovery uses an in-process registry with a filesystem fallback when
the registry has no entries. Prefer job-local scratch for heavy I/O when your
site supports it.
"""

from __future__ import annotations

from scgo.database.cache import get_global_cache
from scgo.database.connection import (
    close_data_connection,
    get_connection,
)
from scgo.database.discovery import list_discovered_db_paths_with_run
from scgo.database.helpers import (
    extract_minima_from_database_file,
    load_previous_run_results,
    load_reference_structures,
    setup_database,
)
from scgo.database.manager import SCGODatabaseManager
from scgo.database.sync import (
    HPC_DATABASE_EXCEPTIONS,
    RetryConfig,
    database_retry,
    retry_transaction,
)
from scgo.database.transactions import database_transaction

__all__ = [
    "get_global_cache",
    "close_data_connection",
    "get_connection",
    "setup_database",
    "extract_minima_from_database_file",
    "load_previous_run_results",
    "load_reference_structures",
    "list_discovered_db_paths_with_run",
    "SCGODatabaseManager",
    "database_transaction",
    "HPC_DATABASE_EXCEPTIONS",
    "RetryConfig",
    "database_retry",
    "retry_transaction",
]
