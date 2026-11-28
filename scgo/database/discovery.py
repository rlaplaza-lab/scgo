"""Database discovery service for SCGO.

Centralizes finding and indexing database files across run directories.
Path lists are not memoized: GO may write a new ``run_*/ga_go.db`` after an
earlier same-process discovery (e.g. loading previous results), and TS must
see that file on reload.
"""

from __future__ import annotations

import glob
import os
import sqlite3
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from ase.data import chemical_symbols

from scgo.database.connection import get_connection
from scgo.database.registry import get_registry
from scgo.database.streaming import relaxed_rows_where_clause
from scgo.metadata.db_stamp import is_scgo_db
from scgo.metadata.run_dir import load_run_dir_record, resolve_run_id_from_db_path
from scgo.utils.helpers import get_cluster_formula, get_composition_counts
from scgo.utils.logging import get_logger
from scgo.utils.messages import COULD_NOT_RESOLVE_RUN_ID_FOR_DATABASE

logger = get_logger(__name__)

_discovery_by_base: dict[str, DatabaseDiscovery] = {}


def _filter_scgo_databases(db_files: list[Path]) -> list[Path]:
    """Keep only databases marked as SCGO."""
    return [p for p in db_files if is_scgo_db(p)]


def _build_run_glob_pattern(
    run_id: str | None,
    db_filename: str,
    *,
    nested_under_run: bool,
) -> str:
    """Build a glob under ``run_*`` (or a specific run) for *db_filename*."""
    nest = "**/" if nested_under_run else ""
    if run_id:
        return f"{run_id}/{nest}{db_filename}"
    return f"run_*/{nest}{db_filename}"


def discover_db_files(
    base_dir: str | Path,
    *,
    pattern: str | None = None,
    db_filename: str | None = None,
    run_id: str | None = None,
    composition: list[str] | None = None,
    merge_registry: bool = True,
    require_scgo: bool = True,
    nested_under_run: bool = True,
    path_filter: Callable[[Path], bool] | None = None,
    composition_filter: Callable[[list[Path], list[str]], list[Path]] | None = None,
) -> list[Path]:
    """Discover database files under *base_dir*.

    Prefer *pattern* when set; otherwise search ``run_*/`` (optionally nested)
    for *db_filename*. Callers select filters via the keyword flags.
    """
    root = Path(base_dir)
    by_resolved: dict[str, Path] = {}
    filename = db_filename if db_filename is not None else "*.db"

    if merge_registry and pattern is None and filename == "*.db":
        registry = get_registry(root)
        registry_files = registry.find_databases(
            composition=composition,
            run_id=run_id,
        )
        logger.debug("Registry found %d databases", len(registry_files))
        filtered = (
            _filter_scgo_databases(registry_files) if require_scgo else registry_files
        )
        if require_scgo and len(filtered) != len(registry_files):
            logger.debug(
                "Dropped %d non-SCGO paths from registry results",
                len(registry_files) - len(filtered),
            )
        for path in filtered:
            by_resolved[str(path.resolve())] = Path(path)

    if pattern is not None:
        glob_pattern = pattern
    else:
        glob_pattern = _build_run_glob_pattern(
            run_id, filename, nested_under_run=nested_under_run
        )

    pattern_path = Path(glob_pattern)
    if pattern_path.is_absolute():
        full_pattern = str(pattern_path)
    else:
        full_pattern = str(root / glob_pattern)

    glob_files = [Path(p) for p in glob.glob(full_pattern, recursive=True)]
    logger.debug(
        "Found %d databases matching pattern: %s", len(glob_files), glob_pattern
    )

    if path_filter is not None:
        glob_files = [p for p in glob_files if path_filter(p)]

    if composition is not None and composition_filter is not None:
        glob_files = composition_filter(glob_files, composition)
        logger.debug("After composition filter: %d databases remain", len(glob_files))

    if require_scgo:
        orig_count = len(glob_files)
        glob_files = _filter_scgo_databases(glob_files)
        if len(glob_files) != orig_count:
            logger.debug(
                "Filtered non-SCGO DBs: %d -> %d databases",
                orig_count,
                len(glob_files),
            )

    for path in glob_files:
        by_resolved[str(path.resolve())] = path

    return list(by_resolved.values())


class DatabaseDiscovery:
    """Service for discovering and indexing database files."""

    def __init__(self, base_dir: str | Path):
        """Initialize database discovery.

        Args:
            base_dir: Base directory to search (usually output directory)
        """
        self.base_dir = Path(base_dir)
        self._registry = get_registry(self.base_dir)
        logger.debug("Initialized DatabaseDiscovery for %s", self.base_dir)

    def find_databases(
        self,
        composition: list[str] | None = None,
        run_id: str | None = None,
        db_filename: str = "*.db",
    ) -> list[Path]:
        """Find databases matching criteria.

        Registry hits are always merged with a ``run_*/**/`` filesystem scan so a
        DB on disk that is not yet registered is still found.
        """
        return discover_db_files(
            self.base_dir,
            db_filename=db_filename,
            run_id=run_id,
            composition=composition,
            merge_registry=True,
            require_scgo=True,
            nested_under_run=True,
            composition_filter=self._filter_by_composition,
        )

    def _first_relaxed_probe(self, db) -> tuple[Counter[str] | None, object | None]:
        """``(counts, atoms)`` for the first relaxed row: prefer SQL ``species``, else ``get_atoms``.

        Both are ``None`` when no relaxed row exists or the probe fails.
        """
        where_sql = relaxed_rows_where_clause()
        try:
            with db.c.managed_connection() as conn:
                cur = conn.execute(
                    f"SELECT id FROM systems WHERE {where_sql} ORDER BY id ASC LIMIT 1"
                )
                row = cur.fetchone()
                rowid = row[0] if row else None
                if rowid is None:
                    return None, None
                try:
                    species_rows = conn.execute(
                        "SELECT Z, n FROM species WHERE id = ?", (rowid,)
                    ).fetchall()
                except sqlite3.DatabaseError:
                    species_rows = []
            if species_rows:
                counts: Counter[str] = Counter()
                for z, n in species_rows:
                    counts[chemical_symbols[int(z)]] += int(n)
                return counts, None
            return None, db.get_atoms(rowid)
        except (
            AttributeError,
            IndexError,
            sqlite3.DatabaseError,
            TypeError,
            ValueError,
        ) as e:
            logger.warning("Failed relaxed-candidate probe: %s", e)
            return None, None

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
                run_id = resolve_run_id_from_db_path(
                    str(db_path), base_dir=str(self.base_dir)
                )
                if run_id:
                    if run_id not in run_formula_cache:
                        record = load_run_dir_record(str(self.base_dir / run_id))
                        run_formula_cache[run_id] = record.formula if record else None
                    known_formula = run_formula_cache[run_id]
                    if known_formula is not None and known_formula == target_formula:
                        filtered.append(db_path)
                        continue
                    # Metadata formula is often mobile-only (Pt5 vs C150Pt5); probe.
                with get_connection(db_path) as db:
                    cand_counts, first_candidate = self._first_relaxed_probe(db)

                    if cand_counts is None:
                        if not first_candidate:
                            continue
                        cand_counts = get_composition_counts(
                            first_candidate.get_chemical_symbols()
                        )

                    if cand_counts == target_counts:
                        filtered.append(db_path)

            except (
                sqlite3.DatabaseError,
                OSError,
                ValueError,
                KeyError,
                AttributeError,
            ) as e:
                logger.warning("Error checking composition for %s: %s", db_path, e)
                continue

        return filtered


def _get_discovery(base_dir: str | Path) -> DatabaseDiscovery:
    """Return a cached :class:`~scgo.database.discovery.DatabaseDiscovery` for *base_dir*."""
    key = os.path.abspath(str(base_dir))
    if key not in _discovery_by_base:
        _discovery_by_base[key] = DatabaseDiscovery(key)
    return _discovery_by_base[key]


def clear_discovery_cache(base_dir: str | Path | None = None) -> None:
    """Drop process-wide database discovery instances for *base_dir* (or all).

    Args:
        base_dir: When set, clear only that resolved path; otherwise clear all.
    """
    if base_dir is None:
        _discovery_by_base.clear()
        return
    _discovery_by_base.pop(os.path.abspath(str(base_dir)), None)


def list_discovered_db_paths_with_run(
    base_dir: str | Path,
    *,
    composition: list[str] | None = None,
    db_filename: str | None = None,
) -> list[tuple[str, str | None]]:
    """List DB paths via :class:`~scgo.database.discovery.DatabaseDiscovery` with run parsed from layout.

    Returns tuples ``(absolute_path, run_id)``. ``run_id`` is ``None`` if the path
    is not under a recognizable ``run_*`` directory.
    """
    base_s = os.path.abspath(str(base_dir))
    discovery = _get_discovery(base_s)
    filename_pattern = db_filename if db_filename else "*.db"
    db_paths = discovery.find_databases(
        composition=composition,
        db_filename=filename_pattern,
    )

    out: list[tuple[str, str | None]] = []
    for db_path in db_paths:
        db_path_str = os.path.abspath(str(db_path))
        run_id = resolve_run_id_from_db_path(db_path_str, base_dir=base_s)
        if not run_id:
            logger.warning(
                COULD_NOT_RESOLVE_RUN_ID_FOR_DATABASE,
                db_path_str,
                base_s,
            )
        out.append((db_path_str, run_id))
    return out
