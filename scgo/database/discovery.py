"""Database discovery service for SCGO.

Centralizes logic for finding and indexing database files across
run directories with caching for performance.
"""

from __future__ import annotations

import glob
import os
import sqlite3
from pathlib import Path

from scgo.database.connection import get_connection
from scgo.database.registry import get_registry
from scgo.database.schema import clear_scgo_database_cache, is_scgo_database
from scgo.database.streaming import relaxed_rows_where_clause
from scgo.utils.helpers import get_cluster_formula, get_composition_counts
from scgo.utils.logging import get_logger
from scgo.utils.run_tracking import load_run_metadata, resolve_run_id_from_db_path

logger = get_logger(__name__)

_discovery_by_base: dict[str, DatabaseDiscovery] = {}


def _filter_scgo_databases(db_files: list[Path]) -> list[Path]:
    """Keep only databases marked as SCGO."""
    return [p for p in db_files if is_scgo_database(p)]


class DatabaseDiscovery:
    """Service for discovering and indexing database files."""

    def __init__(self, base_dir: str | Path):
        """Initialize database discovery.

        Args:
            base_dir: Base directory to search (usually output directory)
        """
        self.base_dir = Path(base_dir)
        self._cache: dict[str, list[Path]] = {}

        self._registry = get_registry(self.base_dir)
        logger.debug("Initialized DatabaseDiscovery for %s", self.base_dir)

    def find_databases(
        self,
        composition: list[str] | None = None,
        run_id: str | None = None,
        db_filename: str = "*.db",
        use_cache: bool = True,
    ) -> list[Path]:
        """Find databases matching criteria."""
        cache_key = self._build_cache_key(composition, run_id, db_filename)

        if use_cache and cache_key in self._cache:
            logger.trace("Using cached results for: %s", cache_key)
            return self._cache[cache_key]

        if db_filename == "*.db":
            db_files = self._registry.find_databases(
                composition=composition,
                run_id=run_id,
            )
            logger.debug("Registry found %d databases", len(db_files))

            if db_files:
                filtered = _filter_scgo_databases(db_files)
                if len(filtered) != len(db_files):
                    logger.debug(
                        "Dropped %d non-SCGO paths from registry results",
                        len(db_files) - len(filtered),
                    )
                db_files = filtered
                if db_files:
                    if use_cache:
                        self._cache[cache_key] = db_files
                    return db_files

            logger.debug("Registry returned no databases; running filesystem scan")

        pattern = self._build_glob_pattern(run_id, db_filename)
        full_pattern = str(self.base_dir / pattern)
        db_files = [Path(p) for p in glob.glob(full_pattern, recursive=True)]

        logger.debug("Found %d databases matching pattern: %s", len(db_files), pattern)

        if composition:
            db_files = self._filter_by_composition(db_files, composition)
            logger.debug("After composition filter: %d databases remain", len(db_files))

        orig_count = len(db_files)
        db_files = _filter_scgo_databases(db_files)
        if len(db_files) != orig_count:
            logger.debug(
                "Filtered non-SCGO DBs: %d -> %d databases", orig_count, len(db_files)
            )

        if use_cache:
            self._cache[cache_key] = db_files

        return db_files

    def clear_cache(self) -> None:
        """Clear discovery caches after filesystem changes."""
        self._cache.clear()
        clear_scgo_database_cache()
        logger.debug("Cleared database discovery caches")

    def _build_cache_key(
        self,
        composition: list[str] | None,
        run_id: str | None,
        db_filename: str,
    ) -> str:
        """Build unique cache key from parameters."""
        comp_str = "-".join(sorted(composition)) if composition else "any"
        run_str = run_id or "any"
        return f"{comp_str}:{run_str}:{db_filename}"

    def _build_glob_pattern(
        self,
        run_id: str | None,
        db_filename: str,
    ) -> str:
        """Build glob pattern for database search."""
        if run_id:
            return f"{run_id}/{db_filename}"
        return f"run_*/{db_filename}"

    def _get_first_relaxed_candidate(self, db) -> object | None:
        """Get one relaxed candidate via SQL."""
        where_sql = relaxed_rows_where_clause()
        try:
            with db.c.managed_connection() as conn:
                cur = conn.execute(
                    f"SELECT id FROM systems WHERE {where_sql} ORDER BY id ASC LIMIT 1"
                )
                row = cur.fetchone()
            rowid = row[0] if row else None
            if rowid is None:
                return None
            return db.get_atoms(rowid)
        except (
            AttributeError,
            sqlite3.DatabaseError,
            sqlite3.OperationalError,
            TypeError,
            ValueError,
        ) as e:
            logger.debug("Failed relaxed-candidate probe: %s", e)
            return None

    def _filter_by_composition(
        self,
        db_files: list[Path],
        composition: list[str],
    ) -> list[Path]:
        """Filter database files by composition."""
        target_counts = get_composition_counts(composition)
        target_formula = get_cluster_formula(composition)
        filtered = []
        run_formula_cache: dict[str, str | None] = {}

        for db_path in db_files:
            try:
                run_id = resolve_run_id_from_db_path(str(db_path), base_dir=str(self.base_dir))
                if run_id:
                    if run_id not in run_formula_cache:
                        metadata = load_run_metadata(str(self.base_dir / run_id))
                        run_formula_cache[run_id] = metadata.formula if metadata else None
                    known_formula = run_formula_cache[run_id]
                    if known_formula is not None:
                        if known_formula == target_formula:
                            filtered.append(db_path)
                        continue
                with get_connection(db_path) as db:
                    first_candidate = self._get_first_relaxed_candidate(db)

                    if not first_candidate:
                        continue

                    symbols = first_candidate.get_chemical_symbols()
                    cand_counts = get_composition_counts(symbols)

                    if cand_counts == target_counts:
                        filtered.append(db_path)

            except (
                sqlite3.DatabaseError,
                sqlite3.OperationalError,
                OSError,
                ValueError,
                KeyError,
                AttributeError,
            ) as e:
                logger.debug("Error checking composition for %s: %s", db_path, e)
                continue

        return filtered


def _get_discovery(base_dir: str | Path) -> DatabaseDiscovery:
    """Return a cached :class:`DatabaseDiscovery` for *base_dir*."""
    key = os.path.abspath(str(base_dir))
    if key not in _discovery_by_base:
        _discovery_by_base[key] = DatabaseDiscovery(key)
    return _discovery_by_base[key]


def _glob_run_database_paths(
    base_dir: str,
    db_filename: str | None = None,
) -> list[Path]:
    """Filesystem fallback when registry-backed discovery returns nothing."""
    pattern_name = db_filename if db_filename else "*.db"
    return sorted(
        Path(p)
        for p in glob.glob(os.path.join(base_dir, "run_*", pattern_name), recursive=False)
    )


def list_discovered_db_paths_with_run(
    base_dir: str | Path,
    *,
    composition: list[str] | None = None,
    use_cache: bool = True,
    db_filename: str | None = None,
) -> list[tuple[str, str]]:
    """List DB paths via :class:`DatabaseDiscovery` with run parsed from layout.

    Returns tuples ``(absolute_path, run_id)``. ``run_id`` is empty if the path
    is not under a recognizable ``run_*`` directory.
    """
    base_s = os.path.abspath(str(base_dir))
    discovery = _get_discovery(base_s)
    filename_pattern = db_filename if db_filename else "*.db"
    db_paths = discovery.find_databases(
        composition=composition,
        use_cache=use_cache,
        db_filename=filename_pattern,
    )
    if not db_paths:
        db_paths = _filter_scgo_databases(_glob_run_database_paths(base_s, db_filename))

    out: list[tuple[str, str]] = []
    for db_path in db_paths:
        db_path_str = os.path.abspath(str(db_path))
        run_id = resolve_run_id_from_db_path(db_path_str, base_dir=base_s)
        if not run_id:
            logger.warning(
                "Could not resolve run_id for database %s under %s",
                db_path_str,
                base_s,
            )
        out.append((db_path_str, run_id))
    return out
