"""Memory-efficient streaming iterators for large databases.

Provides generators for iterating over database contents without loading
everything into memory at once.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from collections.abc import Generator
from pathlib import Path

from ase import Atoms
from ase.db.row import AtomsRow

from scgo.database.connection import get_connection
from scgo.database.constants import SYSTEMS_JSON_COLUMN
from scgo.database.metadata import add_metadata
from scgo.database.schema import is_scgo_database
from scgo.database.sync import database_retry
from scgo.utils.helpers import extract_energy_from_atoms
from scgo.utils.logging import TRACE, get_logger

logger = get_logger(__name__)


def _load_atoms_chunk(
    conn: sqlite3.Connection, row_ids: list[int], da
) -> list[tuple[int, Atoms]]:
    """Load atom rows for a chunk using one SQL query."""
    if not row_ids:
        return []
    placeholders = ",".join("?" for _ in row_ids)
    by_id: dict[int, Atoms] = {}
    try:
        old_row_factory = conn.row_factory
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute(
                f"SELECT * FROM systems WHERE id IN ({placeholders})",
                tuple(row_ids),
            )
            rows = cur.fetchall()
        finally:
            conn.row_factory = old_row_factory
        for row in rows:
            row_dict = dict(row)
            for key in ("key_value_pairs", "data", "constraints"):
                value = row_dict.get(key)
                if isinstance(value, str):
                    with contextlib.suppress(json.JSONDecodeError):
                        row_dict[key] = json.loads(value)
            with contextlib.suppress(
                TypeError,
                ValueError,
                KeyError,
                json.JSONDecodeError,
                sqlite3.DatabaseError,
            ):
                by_id[int(row["id"])] = AtomsRow(row_dict).toatoms(
                    add_additional_information=True
                )
    except (sqlite3.DatabaseError, sqlite3.OperationalError, TypeError, ValueError):
        by_id = {}

    out: list[tuple[int, Atoms]] = []
    for row_id in row_ids:
        atoms = by_id.get(row_id)
        if atoms is None:
            try:
                atoms = da.get_atoms(row_id)
            except (
                KeyError,
                IndexError,
                sqlite3.DatabaseError,
                ValueError,
                TypeError,
                json.JSONDecodeError,
            ) as exc:
                logger.warning(
                    "Failed to fetch atoms id=%s from chunked stream: %s", row_id, exc
                )
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


# Backward-compatible alias for internal callers during transition.
_relaxed_rows_where_clause = relaxed_rows_where_clause


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
        raise ValueError("chunk_size must be a positive integer")

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
            for row_id, candidate in _load_atoms_chunk(conn, row_ids, da):
                energy = extract_energy_from_atoms(candidate)
                if energy is None:
                    logger.log(TRACE, "Skipping candidate id=%s: no energy", row_id)
                    continue

                out = candidate.copy()
                try:
                    add_metadata(out, systems_row_id=int(row_id))
                except (TypeError, ValueError) as e:
                    logger.debug("Failed to attach systems_row_id metadata: %s", e)
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

    if not is_scgo_database(db_path):
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


def iter_databases_minima(
    db_paths: list[str | Path],
    max_structures: int | None = None,
    **iter_kwargs,
) -> Generator[tuple[float, Atoms], None, None]:
    """Iterate over minima from multiple databases."""
    count = 0

    for db_path in db_paths:
        if max_structures and count >= max_structures:
            logger.debug("Reached max_structures limit (%s)", max_structures)
            break

        for energy, atoms in iter_database_minima(db_path, **iter_kwargs):
            yield (energy, atoms)
            count += 1

            if max_structures and count >= max_structures:
                break

    logger.debug("Streamed %s total structures from %s databases", count, len(db_paths))


def count_database_structures(db_path: str | Path) -> int:
    """Count relaxed structures in database without loading them."""
    db_path = Path(db_path)

    if not db_path.exists():
        return 0

    if not is_scgo_database(db_path):
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
