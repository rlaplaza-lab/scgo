"""Metadata helper functions for SCGO databases."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from ase import Atoms

from scgo.database.constants import SYSTEMS_JSON_COLUMN
from scgo.utils.logging import TRACE, get_logger

logger = get_logger(__name__)

# Cache of generations for which we've already emitted a debug-level
# metadata log. Prevents noisy repeated debug messages when many
# candidates in the same generation call `add_metadata`.
_debug_logged_generations: set[int] = set()


def add_metadata(
    atoms: Atoms,
    run_id: str | None = None,
    generation: int | None = None,
    **extra_metadata: Any,
) -> None:
    """Add metadata to an Atoms object (stored in atoms.info['metadata'])."""
    # Initialize metadata dict if not present
    if "metadata" not in atoms.info:
        atoms.info["metadata"] = {}

    metadata = atoms.info["metadata"]

    # Store standard metadata
    if run_id is not None:
        metadata["run_id"] = run_id
    if generation is not None:
        metadata["generation"] = generation

    # Store extra metadata
    metadata.update(extra_metadata)

    # ASE DB persists key_value_pairs; ensure raw_score and run_id for DB rows
    kv = atoms.info.setdefault("key_value_pairs", {})
    if "raw_score" in extra_metadata:
        kv["raw_score"] = extra_metadata["raw_score"]
    if run_id is not None:
        kv["run_id"] = run_id

    # Provenance for run discovery (used by scgo() results and downstream)
    if run_id is not None:
        prov = atoms.info.setdefault("provenance", {})
        prov["run_id"] = run_id

    # TS search provenance: minima_source_db, minima_confids, minima_unique_ids,
    # ts_endpoint_provenance (per-endpoint dicts linking TS to GO minima rows).
    for key in (
        "minima_source_db",
        "minima_confids",
        "minima_unique_ids",
        "ts_endpoint_provenance",
    ):
        if key in extra_metadata:
            atoms.info.setdefault("provenance", {})[key] = extra_metadata[key]

    # Per-candidate: very verbose trace-level record so callers that want
    # full candidate-level detail can enable trace logging.
    keys = list(metadata.keys())
    # Use numeric TRACE level via logger.log(...) because the `.trace`
    # convenience method may not be installed in all runtime setups.
    logger.log(TRACE, "Added metadata to atoms: %s", keys)

    # Per-generation: emit a single debug-level message the first time
    # `add_metadata` is called for a particular generation to reduce
    # repeated debug noise during GA population processing.
    if generation is not None and generation not in _debug_logged_generations:
        logger.debug("Added metadata to atoms: %s", keys)
        _debug_logged_generations.add(generation)


def get_metadata(atoms: Atoms, key: str, default: Any = None) -> Any:
    """Retrieve a metadata value from an Atoms object, or return default.

    Order: ``metadata`` (canonical), ``provenance`` (TS / discovery), then
    ``key_value_pairs`` (ASE DB persisted fields, e.g. raw_score from GA).
    """
    for src in (
        atoms.info.get("metadata", {}),
        atoms.info.get("provenance", {}),
        atoms.info.get("key_value_pairs", {}),
    ):
        if key in src:
            return src[key]
    return default


def get_all_metadata(atoms: Atoms) -> dict[str, Any]:
    """Return metadata from atoms.info['metadata'] (canonical source)."""
    return dict(atoms.info.get("metadata", {}))


def _parse_key_value_pairs_row(row: tuple) -> dict[str, Any]:
    """Parse ``key_value_pairs`` JSON from ``SELECT id, energy, key_value_pairs`` rows."""
    try:
        kv_json = row[2]
        if not kv_json:
            return {}
        return json.loads(kv_json)
    except (json.JSONDecodeError, TypeError, ValueError, IndexError) as exc:
        logger.debug("Failed to parse key_value_pairs row: %s", exc)
        return {}


def _is_row_relaxed_row(row: tuple) -> bool:
    return bool(_parse_key_value_pairs_row(row).get("relaxed"))


def _find_first_relaxed_row(rows: list) -> tuple | None:
    for r in rows:
        if _is_row_relaxed_row(r):
            return r
    return None


def _match_row_by_stored_final_id(
    conn,
    *,
    kvp: str,
    select_cols: str,
    final_id: str,
) -> tuple | None:
    fid_conditions = [
        f"CAST(json_extract({kvp}, '$.final_id') AS TEXT) = ?",
        f"CAST(json_extract({kvp}, '$.unique_id') AS TEXT) = ?",
        "CAST(unique_id AS TEXT) = ?",
    ]
    fid_params = [final_id, final_id, final_id]
    query = (
        f"SELECT {select_cols} FROM systems WHERE "
        + " OR ".join(fid_conditions)
        + " ORDER BY rowid ASC"
    )
    rows = conn.execute(query, tuple(fid_params)).fetchall()
    if not rows:
        return None
    return _find_first_relaxed_row(rows) or rows[0]


def update_metadata(atoms: Atoms, **updates: Any) -> None:
    """Update ``atoms.info['metadata']``; mirror ``raw_score`` into key_value_pairs (ASE)."""
    if "metadata" not in atoms.info:
        atoms.info["metadata"] = {}

    atoms.info["metadata"].update(updates)

    if "raw_score" in updates:
        if "key_value_pairs" not in atoms.info:
            atoms.info["key_value_pairs"] = {}
        atoms.info["key_value_pairs"]["raw_score"] = updates["raw_score"]


def persist_provenance(
    atoms: Atoms,
    run_id: str | None = None,
) -> None:
    """Persist run provenance to ``atoms.info`` for discovery."""
    add_metadata(atoms, run_id=run_id)


def filter_by_metadata(
    structures: list[Atoms],
    **filters: Any,
) -> list[Atoms]:
    """Return structures whose metadata match all provided filters."""
    return [
        atoms
        for atoms in structures
        if all(get_metadata(atoms, key) == value for key, value in filters.items())
    ]


def mark_final_minima_in_db(
    final_minima_info: list[dict],
    base_dir: str | Path,
    db_paths: list[str | Path] | None = None,
) -> dict:
    """Mark final unique minima in database ``systems.key_value_pairs`` JSON rows.

    Rows are matched by ``final_id`` stored in ``key_value_pairs`` at relaxed
    persist time (:func:`scgo.utils.helpers.ensure_final_id` via the database
    adapter's ``add_relaxed_step``).

    Args:
        final_minima_info: List of dicts with keys: 'energy' (float), 'atoms' (Atoms),
            'rank' (1-based int), 'final_written' (str filepath or filename),
            'final_id' (str, required)
        base_dir: Base output directory to search for database files (used by discovery)
        db_paths: Optional explicit list of database files to search/update

    Returns:
        dict: summary containing counts, e.g. {"dbs_touched": int, "rows_updated": int, "details": {db_path: rows}}
    """
    # Inline to avoid circular import: connection → sync → utils → helpers → metadata
    from scgo.database.connection import get_connection
    from scgo.database.discovery import DatabaseDiscovery
    from scgo.database.sync import retry_transaction

    discovery = DatabaseDiscovery(base_dir)

    total_rows_updated = 0
    dbs_touched: set[str] = set()
    details: dict[str, int] = {}

    updates_by_db: dict[str, list[dict[str, Any]]] = {}
    for info in final_minima_info:
        atoms = info.get("atoms")
        rank = info.get("rank")
        final_written = info.get("final_written")
        final_id = info.get("final_id")

        if atoms is None:
            logger.warning("mark_final_minima_in_db: missing atoms entry, skipping")
            continue

        if final_id is None:
            logger.warning("mark_final_minima_in_db: missing final_id, skipping")
            continue

        run_id = get_metadata(atoms, "run_id")

        if db_paths:
            db_files = [Path(p) for p in db_paths]
        else:
            db_files = discovery.find_databases(run_id=run_id)

        if not db_files:
            logger.warning(
                "mark_final_minima_in_db: no databases found for "
                f"run={run_id} — check output layout, registry, or pass db_paths"
            )
            continue

        for db_path in db_files:
            db_key = str(db_path)
            updates_by_db.setdefault(db_key, []).append(
                {
                    "run_id": run_id,
                    "rank": rank,
                    "final_written": final_written,
                    "final_id": str(final_id),
                }
            )

    for db_key, db_updates in updates_by_db.items():
        db_path = Path(db_key)
        try:
            with get_connection(db_path) as db:

                def _mark_rows(
                    conn: sqlite3.Connection,
                    updates: list[dict[str, Any]] = db_updates,
                ) -> int:
                    kvp = SYSTEMS_JSON_COLUMN
                    select_cols = f"id, energy, {kvp}"
                    rows_updated_this_db = 0
                    for update in updates:
                        row = _match_row_by_stored_final_id(
                            conn,
                            kvp=kvp,
                            select_cols=select_cols,
                            final_id=update["final_id"],
                        )
                        if row is None:
                            continue

                        row_id, _, kv_col = row

                        try:
                            existing = json.loads(kv_col) if kv_col else {}
                        except (json.JSONDecodeError, TypeError, ValueError):
                            existing = {}

                        run_id = update["run_id"]
                        rank = update["rank"]
                        final_written = update["final_written"]
                        fid = update["final_id"]

                        if run_id is not None:
                            existing["run_id"] = run_id

                        fw_val = (
                            os.path.basename(str(final_written))
                            if final_written is not None
                            else None
                        )
                        final_keys = {
                            "final_unique_minimum": True,
                            "final_rank": int(rank) if rank is not None else None,
                            "final_written": fw_val,
                            "final_id": fid,
                        }
                        existing.update(
                            {k: v for k, v in final_keys.items() if v is not None}
                        )

                        conn.execute(
                            f"UPDATE systems SET {kvp} = ? WHERE id = ?",
                            (json.dumps(existing), row_id),
                        )
                        rows_updated_this_db += 1
                    return rows_updated_this_db

                rows_updated_this_db = retry_transaction(
                    db,
                    _mark_rows,
                    operation_name="mark_final_minima",
                    isolation_level="IMMEDIATE",
                )
                if rows_updated_this_db > 0:
                    total_rows_updated += rows_updated_this_db
                    dbs_touched.add(db_key)
                    details[db_key] = details.get(db_key, 0) + rows_updated_this_db
        except (
            sqlite3.DatabaseError,
            sqlite3.OperationalError,
            OSError,
            json.JSONDecodeError,
            ValueError,
        ) as e:
            logger.warning(f"mark_final_minima_in_db: failed for {db_path}: {e}")
            continue

    return {
        "dbs_touched": len(dbs_touched),
        "rows_updated": total_rows_updated,
        "details": details,
    }
