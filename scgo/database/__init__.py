"""SCGO Database Module

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
from scgo.database.constants import SYSTEMS_JSON_COLUMN
from scgo.database.discovery import list_discovered_db_paths_with_run
from scgo.database.exceptions import (
    DatabaseMigrationError,
    DatabaseSetupError,
)
from scgo.database.helpers import (
    extract_minima_from_database_file,
    extract_transition_states_from_database_file,
    load_previous_run_results,
    load_reference_structures,
    setup_database,
)
from scgo.database.manager import SCGODatabaseManager
from scgo.database.metadata import (
    add_metadata,
    filter_by_metadata,
    get_metadata,
    mark_final_minima_in_db,
    persist_provenance,
    update_metadata,
)
from scgo.database.registry import (
    clear_registry_cache,
    get_registry,
)
from scgo.database.schema import stamp_scgo_database
from scgo.database.sync import (
    HPC_DATABASE_EXCEPTIONS,
    PRESET_CONTENTED,
    RetryConfig,
    database_retry,
    retry_transaction,
    retry_with_backoff,
)
from scgo.database.transactions import database_transaction

__all__ = [
    "get_global_cache",
    "SYSTEMS_JSON_COLUMN",
    "close_data_connection",
    "get_connection",
    "DatabaseSetupError",
    "DatabaseMigrationError",
    "setup_database",
    "extract_minima_from_database_file",
    "extract_transition_states_from_database_file",
    "load_previous_run_results",
    "load_reference_structures",
    "list_discovered_db_paths_with_run",
    "mark_final_minima_in_db",
    "persist_provenance",
    "SCGODatabaseManager",
    "database_transaction",
    "stamp_scgo_database",
    "add_metadata",
    "get_metadata",
    "update_metadata",
    "filter_by_metadata",
    "HPC_DATABASE_EXCEPTIONS",
    "PRESET_CONTENTED",
    "RetryConfig",
    "database_retry",
    "retry_transaction",
    "retry_with_backoff",
    "get_registry",
    "clear_registry_cache",
]
