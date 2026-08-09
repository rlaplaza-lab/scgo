"""Memory-efficient streaming iterators for large databases.

Provides generators for iterating over database contents without loading
everything into memory at once.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from pathlib import Path

from ase import Atoms

from scgo.database.connection import get_connection
from scgo.database.constants import SYSTEMS_JSON_COLUMN
from scgo.database.sync import database_retry
from scgo.exceptions import SCGOValidationError
from scgo.metadata.atoms import set_tags
from scgo.metadata.db_stamp import is_scgo_db
from scgo.utils.helpers import copy_atoms, extract_energy_from_atoms
from scgo.utils.logging import TRACE, get_logger

logger = get_logger(__name__)


def _load_atoms_chunk(row_ids: list[int], da) -> list[tuple[int, Atoms]]:
    """Load atom rows for a chunk of ids through ASE's row decoder.

    ASE stores ``numbers`` / ``positions`` / ``cell`` as blobs, so rows have to
    be decoded by ASE itself (``DataConnection.get_atoms``); a hand-rolled bulk
    ``SELECT *`` cannot turn those raw buffers back into an ``Atoms`` object.

    Args:
        row_ids: ``systems`` row ids to load, in the desired output order
        da: ASE ``DataConnection`` used to decode each row

    Returns:
        ``(row_id, atoms)`` pairs for every row that could be decoded; rows that
        fail to decode are logged and skipped.
    """
    out: list[tuple[int, Atoms]] = []
    for row_id in row_ids:
        try:
            atoms = da.get_atoms(row_id)
        except (
            KeyError,
            IndexError,
            sqlite3.DatabaseError,
            ValueError,
            TypeError,
        ) as exc:
            logger.warning(
                "Failed to fetch atoms id=%s from chunked stream: %s", row_id, exc
            )
            continue
        if atoms is not None:
            out.append((row_id, atoms))
    return out


def relaxed_rows_where_clause(
    *,
    require_final_minimum: bool = False,
    exclude_transition_states: bool = False,
    require_transition_state: bool = False,
    require_final_ts: bool = False,
) -> str:
    """Build SQL WHERE fragment for relaxed-row streaming filters."""
    col = SYSTEMS_JSON_COLUMN
    clauses = [f"json_extract({col}, '$.relaxed') = 1"]
    if require_final_minimum:
        clauses.append(f"json_extract({col}, '$.final_unique_minimum') = 1")
    if exclude_transition_states:
        clauses.append(f"COALESCE(json_extract({col}, '$.is_transition_state'), 0) = 0")
    if require_transition_state:
        clauses.append(f"json_extract({col}, '$.is_transition_state') = 1")
    if require_final_ts:
        clauses.append(f"json_extract({col}, '$.final_unique_ts') = 1")
    return " AND ".join(clauses)


def iter_relaxed_structures(
    da,
    db_path: Path,
    chunk_size: int = 100,
    *,
    require_final_minimum: bool = False,
    exclude_transition_states: bool = False,
    require_transition_state: bool = False,
    require_final_ts: bool = False,
):
    """Yield (energy, atoms_copy) for relaxed rows using chunked id queries."""
    if chunk_size is None or chunk_size <= 0:
        raise SCGOValidationError("chunk_size must be a positive integer")

    where_sql = relaxed_rows_where_clause(
        require_final_minimum=require_final_minimum,
        exclude_transition_states=exclude_transition_states,
        require_transition_state=require_transition_state,
        require_final_ts=require_final_ts,
    )

    with da.c.managed_connection() as conn:
        json_col = SYSTEMS_JSON_COLUMN

        if logger.isEnabledFor(TRACE):
            try:
                cur = conn.execute(f"SELECT COUNT(*) FROM systems WHERE {where_sql}")
                total = int((cur.fetchone() or [0])[0] or 0)
            except (
                sqlite3.DatabaseError,
                sqlite3.OperationalError,
                TypeError,
                ValueError,
            ) as exc:
                logger.debug("COUNT query failed for %s: %s", db_path, exc)
                total = 0
            logger.debug(
                "Streaming %s structures from %s (chunk_size=%s)",
                total,
                db_path,
                chunk_size,
            )

        try:
            cursor = conn.execute(
                f"SELECT id FROM systems WHERE {where_sql} "
                f"ORDER BY CAST(json_extract({json_col}, '$.raw_score') AS REAL) DESC"
            )
        except sqlite3.OperationalError:
            cursor = conn.execute(
                f"SELECT id FROM systems WHERE {where_sql} ORDER BY id"
            )

        while True:
            rows = cursor.fetchmany(chunk_size)
            if not rows:
                break
            row_ids = [int(row_id) for (row_id,) in rows]
            for row_id, candidate in _load_atoms_chunk(row_ids, da):
                energy = extract_energy_from_atoms(candidate)
                if energy is None:
                    logger.log(TRACE, "Skipping candidate id=%s: no energy", row_id)
                    continue

                out = copy_atoms(candidate)
                try:
                    set_tags(out, systems_row_id=int(row_id))
                except (TypeError, ValueError) as e:
                    logger.debug("Failed to attach systems_row_id tag: %s", e)
                yield (energy, out)


def iter_database_minima(
    db_path: str | Path,
    chunk_size: int = 100,
    *,
    require_final_minimum: bool = False,
    exclude_transition_states: bool = False,
    require_transition_state: bool = False,
    require_final_ts: bool = False,
) -> Generator[tuple[float, Atoms], None, None]:
    """Iterate over minima from database in memory-efficient chunks."""
    db_path = Path(db_path)

    if not db_path.exists():
        logger.warning("Database does not exist: %s", db_path)
        return

    if not is_scgo_db(db_path):
        logger.debug("Skipping non-SCGO database: %s", db_path)
        return

    try:
        with get_connection(str(db_path)) as da:
            yield from iter_relaxed_structures(
                da,
                db_path,
                chunk_size,
                require_final_minimum=require_final_minimum,
                exclude_transition_states=exclude_transition_states,
                require_transition_state=require_transition_state,
                require_final_ts=require_final_ts,
            )
    except (sqlite3.DatabaseError, sqlite3.OperationalError, OSError) as e:
        logger.error("Error streaming from %s: %s", db_path, e)
        raise


def count_database_structures(db_path: str | Path) -> int:
    """Count relaxed structures in database without loading them."""
    db_path = Path(db_path)

    if not db_path.exists():
        return 0

    if not is_scgo_db(db_path):
        logger.debug("Skipping count for non-SCGO database: %s", db_path)
        return 0

    where_sql = relaxed_rows_where_clause()

    def _count() -> int:
        with get_connection(str(db_path)) as da, da.c.managed_connection() as conn:
            cur = conn.execute(f"SELECT COUNT(*) FROM systems WHERE {where_sql}")
            res = cur.fetchone()
            return int((res or [0])[0] or 0)

    try:
        return database_retry(
            _count,
            operation_name=f"count structures in {db_path}",
        )
    except (sqlite3.DatabaseError, sqlite3.OperationalError, OSError) as e:
        logger.error("Error counting structures in %s: %s", db_path, e)
        raise
