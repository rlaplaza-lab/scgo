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
from scgo.database.constants import SYSTEMS_JSON_COLUMN
from scgo.database.registry import get_registry
from scgo.database.schema import is_scgo_database
from scgo.database.streaming import count_database_structures
from scgo.utils.helpers import get_composition_counts
from scgo.utils.logging import get_logger
from scgo.utils.run_tracking import get_run_id_from_dir

logger = get_logger(__name__)


def _filter_scgo_databases(db_files: list[Path]) -> list[Path]:
    """Keep only databases marked as SCGO."""
    return [p for p in db_files if is_scgo_database(p)]


class DatabaseDiscovery:
    """Service for discovering and indexing database files.

    Provides centralized database finding with caching to avoid
    repeated filesystem scans.

    Example:
        >>> discovery = DatabaseDiscovery("output")
        >>>
        >>> # Find all databases for Pt3
        >>> db_files = discovery.find_databases(composition=["Pt", "Pt", "Pt"])
        >>>
        >>> # Find databases for specific run
        >>> db_files = discovery.find_databases(run_id="run_20260204_120000")
        >>>
        >>> # Clear cache when filesystem changes
        >>> discovery.clear_cache()
    """

    def __init__(self, base_dir: str | Path):
        """Initialize database discovery.

        Args:
            base_dir: Base directory to search (usually output directory)
        """
        self.base_dir = Path(base_dir)
        self._cache: dict[str, list[Path]] = {}
        self._metadata_cache: dict[Path, dict] = {}
        self._cache_hit_count = 0
        self._cache_miss_count = 0

        # Registry for fast lookups - use global cache
        self._registry = get_registry(self.base_dir)
        logger.debug("Using registry for fast database discovery")
        logger.debug(f"Initialized DatabaseDiscovery for {self.base_dir}")

    def _maybe_log_cache_summary(self) -> None:
        """Emit compact INFO cache stats periodically without per-key noise."""
        total = self._cache_hit_count + self._cache_miss_count
        if total == 0 or total % 25 != 0:
            return
        hit_rate = int(100 * self._cache_hit_count / total)
        logger.info(
            "Database discovery cache summary: %d hits, %d misses (%d%% hit rate)",
            self._cache_hit_count,
            self._cache_miss_count,
            hit_rate,
        )

    def find_databases(
        self,
        composition: list[str] | None = None,
        run_id: str | None = None,
        db_filename: str = "*.db",
        use_cache: bool = True,
    ) -> list[Path]:
        """Find databases matching criteria.

        Args:
            composition: Filter by composition (e.g., ["Pt", "Pt", "Pt"])
            run_id: Filter by specific run (e.g., "run_20260204_120000")
            db_filename: Database filename pattern (default "*.db")
            use_cache: Whether to use cached results (default True)

        Returns:
            List of Path objects for matching databases

        Example:
            >>> # Find all Pt3 databases
            >>> db_files = discovery.find_databases(composition=["Pt"]*3)
            >>>
            >>> # Find specific run
            >>> db_files = discovery.find_databases(
            ...     run_id="run_20260204_120000",
            ...     db_filename="ga_go.db"
            ... )
        """
        # Build cache key
        cache_key = self._build_cache_key(composition, run_id, db_filename)

        # Check cache
        if use_cache and cache_key in self._cache:
            self._cache_hit_count += 1
            logger.trace("Using cached results for: %s", cache_key)
            self._maybe_log_cache_summary()
            return self._cache[cache_key]
        if use_cache:
            self._cache_miss_count += 1
            self._maybe_log_cache_summary()

        # Try registry first for fast lookup
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

        # Build glob pattern
        pattern = self._build_glob_pattern(run_id, db_filename)

        # Find matching files
        full_pattern = str(self.base_dir / pattern)
        db_files = [Path(p) for p in glob.glob(full_pattern, recursive=True)]

        logger.debug("Found %d databases matching pattern: %s", len(db_files), pattern)

        # Filter by composition if specified
        if composition:
            db_files = self._filter_by_composition(db_files, composition)
            logger.debug("After composition filter: %d databases remain", len(db_files))

        orig_count = len(db_files)
        db_files = _filter_scgo_databases(db_files)
        if len(db_files) != orig_count:
            logger.debug(
                "Filtered non-SCGO DBs: %d -> %d databases", orig_count, len(db_files)
            )

        # Cache results
        if use_cache:
            self._cache[cache_key] = db_files

        return db_files

    def get_database_info(self, db_path: Path) -> dict:
        """Get metadata about a database.

        Args:
            db_path: Path to database file

        Returns:
            dict: Database metadata (run_id, count, etc.)
        """
        # Check cache
        if db_path in self._metadata_cache:
            return self._metadata_cache[db_path]

        info = {
            "path": str(db_path),
            "exists": db_path.exists(),
            "size_mb": 0,
            "run_id": None,
            "structure_count": 0,
        }

        if not db_path.exists():
            self._metadata_cache[db_path] = info
            return info

        # Get file size
        info["size_mb"] = db_path.stat().st_size / (1024 * 1024)

        # Parse run_id from path
        parts = db_path.parts
        for _part in parts:
            if _part.startswith("run_"):
                info["run_id"] = _part

        # Count structures
        try:
            info["structure_count"] = count_database_structures(db_path)
        except (
            sqlite3.DatabaseError,
            sqlite3.OperationalError,
            OSError,
            ValueError,
        ) as e:
            logger.debug("Failed to count structures in %s: %s", db_path, e)

        # Cache result
        self._metadata_cache[db_path] = info
        return info

    def get_statistics(self) -> dict:
        """Get statistics about discovered databases.

        Returns:
            dict: Statistics (total_databases, total_structures, etc.)
        """
        all_dbs = self.find_databases(use_cache=False)

        stats = {
            "total_databases": len(all_dbs),
            "total_size_mb": 0,
            "total_structures": 0,
            "by_run": {},
            "by_composition": {},
        }

        for db_path in all_dbs:
            info = self.get_database_info(db_path)
            stats["total_size_mb"] += info["size_mb"]
            stats["total_structures"] += info["structure_count"]

            # Count by run
            run_id = info.get("run_id")
            if run_id:
                stats["by_run"][run_id] = stats["by_run"].get(run_id, 0) + 1

        return stats

    def clear_cache(self) -> None:
        """Clear all caches.

        Call this if filesystem has changed (new runs, databases added/removed).
        """
        from scgo.database.schema import clear_scgo_database_cache

        self._cache.clear()
        self._metadata_cache.clear()
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
        """Get one relaxed candidate via SQL (``json_extract`` on the systems JSON column)."""
        try:
            with db.c.managed_connection() as conn:
                cur = conn.execute(
                    f"SELECT id FROM systems WHERE json_extract({SYSTEMS_JSON_COLUMN}, '$.relaxed') = 1 "
                    "ORDER BY id ASC LIMIT 1"
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
        """Filter database files by composition.

        Checks if database contains structures with matching composition.
        """
        target_counts = get_composition_counts(composition)
        filtered = []

        for db_path in db_files:
            try:
                with get_connection(db_path) as db:
                    first_candidate = self._get_first_relaxed_candidate(db)

                    if not first_candidate:
                        continue

                    # Check first candidate's composition
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


def list_discovered_db_paths_with_run(
    base_dir: str | Path,
    *,
    composition: list[str] | None = None,
    use_cache: bool = True,
) -> list[tuple[str, str]]:
    """List DB paths via :class:`DatabaseDiscovery` with run parsed from layout.

    Returns tuples ``(absolute_path, run_id)``. ``run_id`` is empty if the path
    is not under a recognizable ``run_*`` directory.
    """
    base_s = os.path.abspath(str(base_dir))
    discovery = DatabaseDiscovery(base_s)
    out: list[tuple[str, str]] = []
    for db_path in discovery.find_databases(
        composition=composition, use_cache=use_cache
    ):
        db_path_str = os.path.abspath(str(db_path))
        rel = os.path.relpath(db_path_str, base_s)
        parts = rel.split(os.sep)
        run_id = ""
        for part in parts:
            resolved = get_run_id_from_dir(part)
            if resolved is not None:
                run_id = resolved
                break
            if part.startswith("run_"):
                run_id = part
                break
        if not run_id:
            logger.warning(
                "Could not resolve run_id for database %s under %s",
                db_path_str,
                base_s,
            )
        out.append((db_path_str, run_id))
    return out
