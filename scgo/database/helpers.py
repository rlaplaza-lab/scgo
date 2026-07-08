"""SQLite database setup and helpers for SCGO (ASE ``DataConnection``)."""

from __future__ import annotations

import contextlib
import glob
import heapq
import multiprocessing
import os
import sqlite3
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect as ase_db_connect
from ase_ga.data import DataConnection

from scgo.constants import PENALTY_ENERGY
from scgo.database.connection import (
    _run_sqlite,
    apply_sqlite_pragmas,
    close_data_connection,
    get_connection,
    open_data_connection_for_setup,
)
from scgo.database.constants import SYSTEMS_JSON_COLUMN
from scgo.database.discovery import list_discovered_db_paths_with_run
from scgo.database.exceptions import DatabaseSetupError
from scgo.database.metadata import add_metadata, get_metadata
from scgo.database.registry import get_registry
from scgo.database.schema import (
    is_scgo_database,
    stamp_scgo_database,
)
from scgo.database.streaming import iter_database_minima, iter_relaxed_structures
from scgo.database.sync import PRESET_AGGRESSIVE, database_retry, retry_with_backoff
from scgo.utils.helpers import (
    ensure_directory_exists,
    ensure_final_id,
    get_cluster_formula,
    get_composition_counts,
)
from scgo.utils.logging import get_logger
from scgo.utils.run_tracking import load_run_metadata, resolve_run_id_from_db_path

logger = get_logger(__name__)
_MIN_DB_PARALLEL_LOAD_TASKS = 4


def _ensure_database_indices(
    db_path: str,
    *,
    enable_expression_indexes: bool = True,
    enable_wal_mode: bool = False,
) -> None:
    """Create SQLite indices for performance."""
    try:

        def _create_indices(conn: sqlite3.Connection) -> None:
            apply_sqlite_pragmas(
                conn,
                wal_mode=enable_wal_mode,
                busy_timeout=30000,
                cache_size_mb=64,
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_energy ON systems(energy)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_id ON systems(id)")
            with contextlib.suppress(sqlite3.OperationalError):
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_unique_id ON systems(unique_id)"
                )

            if enable_expression_indexes:
                json_col = SYSTEMS_JSON_COLUMN
                with contextlib.suppress(sqlite3.OperationalError):
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_systems_relaxed_json "
                        f"ON systems(json_extract({json_col}, '$.relaxed'))"
                    )
                with contextlib.suppress(sqlite3.OperationalError):
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_systems_raw_score_json "
                        f"ON systems(CAST(json_extract({json_col}, '$.raw_score') AS REAL))"
                    )
                with contextlib.suppress(sqlite3.OperationalError):
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_systems_final_unique_json "
                        f"ON systems(json_extract({json_col}, '$.final_unique_minimum'))"
                    )

        database_retry(
            lambda: _run_sqlite(db_path, _create_indices),
            config=PRESET_AGGRESSIVE,
            operation_name=f"create indices on {db_path}",
        )
        logger.debug(f"Database indices created for {db_path}")
    except sqlite3.OperationalError as e:
        if enable_wal_mode:
            logger.warning(
                "Failed to enable WAL mode for %s: %s. Continuing with default mode.",
                db_path,
                e,
            )
        else:
            logger.debug(f"Could not create all indices on {db_path}: {e}")
    except OSError as e:
        logger.warning(f"Unexpected error creating indices on {db_path}: {e}")


def _register_database_best_effort(
    base_dir: str | Path, db_file: str, atoms_template: Atoms | None, run_id: str | None
) -> None:
    """Best-effort register DB in registry (no exceptions)."""
    comp_list = None
    if atoms_template is not None:
        try:
            comp_list = atoms_template.get_chemical_symbols()
        except (AttributeError, TypeError) as e:
            logger.debug(
                "Could not extract composition from atoms_template for %s: %s",
                db_file,
                e,
            )
            comp_list = None

    base_path = Path(base_dir)

    search_root = next(
        (p for p in base_path.parents if p.name.endswith("_searches")), None
    )
    if search_root is not None:
        registry_roots: list[Path] = [search_root]
    else:
        registry_roots = [base_path]

    for root in registry_roots:
        try:
            get_registry(root).register_database(
                Path(db_file),
                composition=comp_list,
                run_id=run_id,
            )
            logger.debug("Registered database in registry root %s: %s", root, db_file)
        except (ValueError, OSError) as _e:
            logger.debug(
                "Registry registration failed for %s in %s: %s", db_file, root, _e
            )


def setup_database(
    output_dir: str | Path,
    db_filename: str,
    atoms_template: Atoms,
    initial_candidate: Atoms | None = None,
    initial_population: list[Atoms] | None = None,
    remove_existing: bool = True,
    remove_aux_files: bool = False,
    enable_wal_mode: bool = False,
    enable_expression_indexes: bool = True,
    run_id: str | None = None,
) -> DataConnection:
    """Create/open an ASE `DataConnection` for `db_filename` in `output_dir`."""
    output_dir_str = str(output_dir)
    ensure_directory_exists(output_dir_str)
    db_file = os.path.join(output_dir_str, db_filename)

    if remove_aux_files:
        for suffix in ["-shm", "-wal", "-journal"]:
            aux_file = db_file + suffix
            if os.path.exists(aux_file):
                with contextlib.suppress(OSError):
                    os.remove(aux_file)

    if remove_existing and os.path.exists(db_file):

        def _remove_db():
            os.remove(db_file)

        try:
            retry_with_backoff(
                _remove_db,
                max_retries=5,
                initial_delay=0.1,
                backoff_factor=2.0,
                exception_types=(OSError,),
            )
        except OSError as e:
            logger.warning(f"Failed to remove database {db_file}: {e}")

    all_atom_numbers = [int(num) for num in atoms_template.get_atomic_numbers()]

    with ase_db_connect(db_file) as prep_db:
        prep_db.write(
            atoms_template,
            data={"stoichiometry": all_atom_numbers},
            simulation_cell=True,
        )

        if initial_population is not None:
            for candidate in initial_population:
                gaid = prep_db.write(
                    candidate,
                    origin="StartingCandidateUnrelaxed",
                    relaxed=0,
                    generation=0,
                    extinct=0,
                )
                prep_db.update(gaid, gaid=gaid)
                candidate.info["confid"] = gaid
        elif initial_candidate is not None:
            gaid = prep_db.write(
                initial_candidate,
                origin="StartingCandidateUnrelaxed",
                relaxed=0,
                generation=0,
                extinct=0,
            )
            prep_db.update(gaid, gaid=gaid)
            initial_candidate.info["confid"] = gaid

        with contextlib.suppress(AttributeError, sqlite3.OperationalError):
            prep_db.vacuum()

    try:
        da = database_retry(
            lambda: open_data_connection_for_setup(
                db_file,
                wal_mode=enable_wal_mode,
            ),
            config=PRESET_AGGRESSIVE,
            operation_name=f"setup database connection for {db_file}",
        )

        _ensure_database_indices(
            db_file,
            enable_expression_indexes=enable_expression_indexes,
            enable_wal_mode=enable_wal_mode,
        )

        db_path_obj = Path(db_file)
        try:
            sz = db_path_obj.stat().st_size if db_path_obj.exists() else 0
        except OSError:
            sz = -1
        logger.debug(
            "Database setup: path=%s size=%s wal=%s",
            db_file,
            sz,
            enable_wal_mode,
        )

        try:
            stamp_scgo_database(db_file)
        except (
            sqlite3.DatabaseError,
            sqlite3.OperationalError,
            OSError,
            ValueError,
        ) as e:
            logger.warning("Failed to stamp SCGO database %s: %s", db_file, e)

        _register_database_best_effort(output_dir_str, db_file, atoms_template, run_id)

        class _DBAdapter:
            def __init__(self, da_obj, expected_atomic_numbers):
                self._da = da_obj
                self._expected_atomic_numbers = expected_atomic_numbers
                self._last_unrelaxed_metadata = None

            def __getattr__(self, name):
                return getattr(self._da, name)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                with contextlib.suppress(OSError, RuntimeError, AttributeError):
                    close_data_connection(self._da)

            def add_relaxed_step(self, a, *args, **kwargs):
                if Counter(int(x) for x in a.get_atomic_numbers()) != Counter(
                    self._expected_atomic_numbers
                ):
                    raise AssertionError(
                        "Candidate composition does not match database stoichiometry"
                    )

                if "key_value_pairs" not in a.info:
                    a.info["key_value_pairs"] = {}

                if "raw_score" not in a.info.get("key_value_pairs", {}):
                    try:
                        energy = a.get_potential_energy()
                        a.info.setdefault("key_value_pairs", {})["raw_score"] = -float(
                            energy
                        )
                    except (AttributeError, RuntimeError, ValueError):
                        logger.warning(
                            "raw_score missing and energy could not be computed for candidate; "
                            "assigning PENALTY_ENERGY and continuing."
                        )
                        a.info.setdefault("metadata", {})["potential_energy"] = (
                            PENALTY_ENERGY
                        )
                        a.info.setdefault("key_value_pairs", {})[
                            "raw_score"
                        ] = -PENALTY_ENERGY
                        zero_forces = np.zeros((len(a), 3), dtype=np.float64)
                        a.calc = SinglePointCalculator(
                            a, energy=PENALTY_ENERGY, forces=zero_forces
                        )

                ensure_final_id(a)

                return self._da.add_relaxed_step(a, *args, **kwargs)

            def add_unrelaxed_candidate(self, a, *args, **kwargs):
                self._last_unrelaxed_metadata = (
                    a.info.get("metadata", {}).copy() if a.info.get("metadata") else {}
                )

                prov_src = a.info.get("metadata") or {}
                kv = a.info.setdefault("key_value_pairs", {})
                for _k in ("run_id", "trial_id", "confid", "gaid", "id"):
                    if _k in prov_src and _k not in kv:
                        kv[_k] = prov_src[_k]

                a.info.setdefault("data", {})
                return self._da.add_unrelaxed_candidate(a, *args, **kwargs)

            def get_an_unrelaxed_candidate(self, *args, **kwargs):
                u = self._da.get_an_unrelaxed_candidate(*args, **kwargs)
                if (
                    u is not None
                    and "metadata" not in u.info
                    and self._last_unrelaxed_metadata
                ):
                    u.info["metadata"] = self._last_unrelaxed_metadata.copy()
                return u

        return _DBAdapter(da, all_atom_numbers)
    except (sqlite3.DatabaseError, sqlite3.OperationalError, OSError) as e:
        logger.error("Failed to open database after all retries: %s", e)
        raise DatabaseSetupError(f"Failed to setup database {db_file}: {e}") from e


def _extract_structures_from_db(
    db_path: str | Path,
    run_id: str,
    *,
    iter_relaxed_kwargs: dict,
    sort: bool = False,
    persist: bool = False,
    source_db_relpath: str | None = None,
    empty_log: Callable[[], None] | None = None,
) -> list[tuple[float, Atoms]]:
    """Load relaxed structures from a stamped SCGO database file."""
    db_path = str(db_path)

    if not os.path.exists(db_path):
        return []

    if not is_scgo_database(db_path):
        logger.debug("Skipping extract: not an SCGO database %s", db_path)
        return []

    def _extract() -> list[tuple[float, Atoms]]:
        with get_connection(db_path) as da:
            out: list[tuple[float, Atoms]] = []
            for energy, atoms in iter_relaxed_structures(
                da,
                Path(db_path),
                chunk_size=100,
                **iter_relaxed_kwargs,
            ):
                out.append((float(energy), atoms) if sort else (energy, atoms))

            if sort:
                out.sort(key=lambda x: x[0])

            if empty_log is not None and not out:
                empty_log()

            metadata_kwargs: dict[str, str] = {
                "run_id": run_id,
                "source_db": os.path.basename(db_path),
            }
            if source_db_relpath is not None:
                metadata_kwargs["source_db_relpath"] = source_db_relpath
            for _, atoms in out:
                add_metadata(atoms, **metadata_kwargs)

            if persist:
                try:
                    with da.c.managed_connection() as conn:
                        for _, atoms in out:
                            row_id = get_metadata(atoms, "systems_row_id", None)
                            if row_id is None:
                                continue
                            row_id = int(row_id)

                            k = SYSTEMS_JSON_COLUMN
                            conn.execute(
                                f"UPDATE systems SET {k} = json_set(COALESCE({k}, '{{}}'), '$.run_id', ?) WHERE id = ?",
                                (run_id, row_id),
                            )
                        conn.commit()
                except (
                    sqlite3.DatabaseError,
                    sqlite3.OperationalError,
                    OSError,
                    ValueError,
                    TypeError,
                ) as e:
                    logger.debug(
                        "Failed to persist provenance to DB %s: %s", db_path, e
                    )

            return out

    try:
        return database_retry(
            _extract,
            operation_name=f"extract structures from {db_path}",
        )
    except (sqlite3.DatabaseError, OSError, ValueError, AttributeError) as e:
        logger.warning("Failed to extract structures from %s: %s", db_path, e)
        return []


def extract_minima_from_database_file(
    db_path: str | Path,
    run_id: str,
    *,
    require_final: bool = True,
    persist: bool = False,
    source_db_relpath: str | None = None,
) -> list[tuple[float, Atoms]]:
    """Return minima from ``db_path`` annotated with ``run_id``."""
    return _extract_structures_from_db(
        db_path,
        run_id,
        iter_relaxed_kwargs={
            "require_final_minimum": require_final,
            "exclude_transition_states": True,
        },
        persist=persist,
        source_db_relpath=source_db_relpath,
        empty_log=(
            lambda: logger.debug(
                "No final_unique_minimum-tagged rows in %s (require_final=True)",
                db_path,
            )
        )
        if require_final
        else None,
    )


def extract_transition_states_from_database_file(
    db_path: str | Path,
    run_id: str,
    *,
    require_final_unique_ts: bool = True,
) -> list[tuple[float, Atoms]]:
    """Return transition-state rows from ``db_path`` with provenance."""
    return _extract_structures_from_db(
        db_path,
        run_id,
        iter_relaxed_kwargs={
            "require_transition_state": True,
            "require_final_ts": require_final_unique_ts,
        },
        sort=True,
    )


def load_previous_run_results(
    base_output_dir: str,
    db_filename: str | None = None,
    composition: list[str] | None = None,
    current_run_id: str | None = None,
    parallel: bool = True,
    max_workers: int | None = None,
    prefer_final_unique: bool = True,
) -> list[tuple[float, Atoms]]:
    """Load minima from previous runs for a composition."""
    all_db_files: list[tuple[str, str | None]] = []

    if not os.path.exists(base_output_dir):
        return []

    discovered_entries = list_discovered_db_paths_with_run(
        base_output_dir,
        composition=composition,
        use_cache=True,
        db_filename=db_filename,
    )

    if discovered_entries:
        by_run: dict[str, list[str]] = {}
        for db_path_str, run_id in discovered_entries:
            if not run_id:
                logger.warning(
                    "Skipping database %s: could not resolve run_id from path layout",
                    db_path_str,
                )
                continue
            if run_id == current_run_id:
                continue
            by_run.setdefault(run_id, []).append(db_path_str)
        for run_id, db_list in by_run.items():
            run_dir = os.path.join(base_output_dir, run_id)
            metadata = load_run_metadata(run_dir)
            if composition is not None and metadata and metadata.formula:
                expected_formula = get_cluster_formula(composition)
                if metadata.formula != expected_formula:
                    continue
            all_db_files.extend((p, run_id) for p in db_list)

    if not all_db_files:
        logger.info(f"No databases found in {base_output_dir}")
        return []

    if max_workers is None:
        resolved_max_workers = max(1, multiprocessing.cpu_count() // 2)
    else:
        resolved_max_workers = max_workers

    use_parallel = (
        parallel
        and len(all_db_files) >= _MIN_DB_PARALLEL_LOAD_TASKS
        and multiprocessing.current_process().name == "MainProcess"
        and resolved_max_workers > 1
    )

    all_minima: list[tuple[float, Atoms]] = []

    if use_parallel:
        logger.info(
            f"Loading {len(all_db_files)} databases in parallel "
            f"with {resolved_max_workers} workers"
        )

        try:
            with ProcessPoolExecutor(max_workers=resolved_max_workers) as executor:
                futures = {
                    executor.submit(
                        _load_single_database_worker,
                        db_path,
                        composition,
                        run_id,
                        prefer_final_unique,
                    ): (db_path, run_id)
                    for db_path, run_id in all_db_files
                }

                for future in as_completed(futures):
                    db_path, run_id = futures[future]
                    try:
                        minima = future.result(timeout=30)
                        all_minima.extend(minima)
                        if minima:
                            logger.debug(
                                f"Loaded {len(minima)} minima from {os.path.basename(db_path)}"
                            )
                    except (
                        OSError,
                        sqlite3.DatabaseError,
                        RuntimeError,
                        TimeoutError,
                        ValueError,
                    ) as e:
                        logger.error(
                            f"Failed to load {db_path} in parallel worker: {e}"
                        )
        except (
            OSError,
            sqlite3.DatabaseError,
            RuntimeError,
            ValueError,
        ) as e:
            raise RuntimeError(
                f"Parallel minima loading failed for {base_output_dir}: {type(e).__name__}: {e}"
            ) from e

    else:
        logger.info(f"Loading {len(all_db_files)} databases sequentially")

        for db_path, run_id in all_db_files:
            minima = extract_minima_from_database_file(
                db_path, run_id or "", require_final=prefer_final_unique
            )
            filtered_minima = _filter_minima_by_composition(minima, composition)
            all_minima.extend(filtered_minima)
            if filtered_minima:
                logger.debug(
                    f"Loaded {len(filtered_minima)} minima from {os.path.basename(db_path)}"
                )

    logger.info(
        f"Loaded {len(all_minima)} total minima from previous runs "
        f"(excluding {current_run_id})"
    )
    return all_minima


def load_reference_structures(
    db_glob_pattern: str,
    composition: list[str] | None = None,
    max_structures: int = 100,
    base_dir: str | Path | None = None,
) -> list[Atoms]:
    """Load up to `max_structures` final minima from databases matching pattern."""
    pattern_path = Path(db_glob_pattern)
    if pattern_path.is_absolute():
        search_glob = str(pattern_path)
    else:
        root = Path(base_dir) if base_dir is not None else Path.cwd()
        search_glob = str(root / db_glob_pattern)
    db_files = [
        p for p in glob.glob(search_glob, recursive=True) if is_scgo_database(p)
    ]

    if not db_files:
        logger.warning(f"No database files found matching pattern: {db_glob_pattern}")
        return []

    target_counts = None
    if composition is not None:
        target_counts = get_composition_counts(composition)

    heap: list[tuple[float, int, Atoms]] = []
    counter = 0

    for db_file in db_files:
        try:
            resolved_run_id = resolve_run_id_from_db_path(db_file, base_dir=base_dir)
            for energy, atoms in iter_database_minima(
                db_file,
                chunk_size=200,
                require_final_minimum=True,
                exclude_transition_states=True,
            ):
                if target_counts is not None:
                    atoms_symbols = atoms.get_chemical_symbols()
                    atoms_counts = get_composition_counts(atoms_symbols)
                    if atoms_counts != target_counts:
                        continue

                if len(heap) < max_structures:
                    add_metadata(
                        atoms,
                        run_id=resolved_run_id,
                        source_db_relpath=os.path.relpath(
                            db_file,
                            str(base_dir) if base_dir is not None else os.getcwd(),
                        ),
                    )
                    heapq.heappush(heap, (-energy, counter, atoms))
                    counter += 1
                elif energy < -heap[0][0]:
                    counter += 1
                    add_metadata(
                        atoms,
                        run_id=resolved_run_id,
                        source_db_relpath=os.path.relpath(
                            db_file,
                            str(base_dir) if base_dir is not None else os.getcwd(),
                        ),
                    )
                    heapq.heapreplace(heap, (-energy, counter, atoms))
        except (sqlite3.DatabaseError, OSError, ValueError) as e:
            logger.debug(f"Failed to extract minima from {db_file}: {e}")
            continue

    if not heap:
        logger.warning("No final unique minima found in databases matching the pattern")
        return []

    sorted_structures = sorted(heap, key=lambda x: -x[0])
    reference_atoms = [atoms for _, _, atoms in sorted_structures]

    logger.info(
        f"Loaded {len(reference_atoms)} final reference structures for diversity calculation "
        f"from {len(db_files)} databases"
    )

    return reference_atoms


def _filter_minima_by_composition(
    minima: list[tuple[float, Atoms]],
    composition: list[str] | None = None,
) -> list[tuple[float, Atoms]]:
    """Filter minima by stoichiometric composition."""
    if composition is None:
        return minima

    target_counts = get_composition_counts(composition)
    filtered = []
    for energy, atoms in minima:
        atoms_counts = get_composition_counts(atoms.get_chemical_symbols())
        if atoms_counts == target_counts:
            filtered.append((energy, atoms))

    return filtered


def _load_single_database_worker(
    db_path: str,
    composition: list[str] | None = None,
    run_id: str | None = None,
    require_final: bool = False,
) -> list[tuple[float, Atoms]]:
    """Load minima from a single database in subprocess."""
    db_path = str(db_path)

    if not os.path.exists(db_path):
        return []

    try:
        minima = extract_minima_from_database_file(
            db_path, run_id or "", require_final=require_final
        )
    except (sqlite3.DatabaseError, OSError, ValueError) as e:
        logger.error(f"Failed to extract minima from {db_path} in worker: {e}")
        return []

    return _filter_minima_by_composition(minima, composition)
