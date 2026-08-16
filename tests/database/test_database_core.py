"""Unified database tests for SCGO.

Consolidates database setup, connection management, transactions, metadata,
pooling, robustness, and discovery tests into a single module aligned with
the current SCGO database APIs.
"""

from __future__ import annotations

import gc
import os
import sqlite3
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

import pytest
from ase import Atoms
from ase.calculators.emt import EMT

from scgo.database import (
    RetryConfig,
    close_data_connection,
    database_retry,
    database_transaction,
    get_connection,
    setup_database,
)
from scgo.database.discovery import (
    DatabaseDiscovery,
    clear_discovery_cache,
    list_discovered_db_paths_with_run,
)
from scgo.database.streaming import (
    iter_database_minima,
    iter_relaxed_structures,
)
from scgo.exceptions import SCGOValidationError
from scgo.metadata.atoms import (
    get_tag,
    set_tags,
)
from tests.helpers import create_test_atoms


def _final_kvp(raw_score: float) -> dict[str, float | bool]:
    """``key_value_pairs`` for relaxed rows that are canonical final minima."""
    return {"raw_score": raw_score, "final_unique_minimum": True}


@contextmanager
def _setup_test_db(
    tmp_path: Path,
    filename: str,
    template: Atoms,
    *,
    initial_candidate: Atoms | None,
    **setup_kwargs,
):
    da = setup_database(
        tmp_path,
        filename,
        template,
        initial_candidate=initial_candidate,
        **setup_kwargs,
    )
    db_file = Path(tmp_path) / filename
    try:
        yield da, db_file
    finally:
        close_data_connection(da)
        gc.collect()


def _register_unrelaxed(da, atoms: Atoms, *, description: str = "test:insert") -> None:
    atoms.info.setdefault("key_value_pairs", {})
    atoms.info.setdefault("data", {})
    da.add_unrelaxed_candidate(atoms, description=description)


def _build_relaxed_db(tmp_path: Path, filename: str, n_rows: int) -> Path:
    """Create an SCGO db with ``n_rows`` relaxed final minima (raw_score -10-i)."""
    template = Atoms("Pt2", positions=[[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
    da = setup_database(tmp_path, filename, template, initial_candidate=template)
    try:
        for i in range(n_rows):
            a = template.copy()
            a.positions[1][0] += 0.05 * i
            a.info["key_value_pairs"] = _final_kvp(-10.0 - i)
            a.info["data"] = {"tag": f"row_{i}"}
            da.add_relaxed_step(a)
    finally:
        close_data_connection(da)
        gc.collect()
    return Path(tmp_path) / filename


class _RecordingConnection:
    """Proxy around a live sqlite3 connection that records ``execute`` calls.

    Statements containing ``fail_on`` raise ``sqlite3.OperationalError`` so the
    caller's error handling can be exercised.
    """

    def __init__(self, real, fail_on: str | None = None):
        self._real = real
        self._fail_on = fail_on
        self.statements: list[str] = []

    def __getattr__(self, name):
        return getattr(self._real, name)

    def execute(self, sql, *args, **kwargs):
        self.statements.append(sql)
        if self._fail_on is not None and self._fail_on in sql:
            raise sqlite3.OperationalError(f"simulated failure for {self._fail_on!r}")
        return self._real.execute(sql, *args, **kwargs)


def _stream_with_recording_connection(
    db_file: Path, *, fail_on: str | None = None, chunk_size: int = 2
) -> tuple[list[tuple[float, Atoms]], list[str]]:
    """Stream ``db_file`` while recording SQL issued on the streaming connection."""
    proxies: list[_RecordingConnection] = []

    with get_connection(db_file) as da:
        real_managed = da.c.managed_connection

        @contextmanager
        def _managed(commit_frequency=5000):
            with real_managed(commit_frequency) as real:
                proxy = _RecordingConnection(real, fail_on=fail_on)
                proxies.append(proxy)
                yield proxy

        da.c.managed_connection = _managed
        yielded = list(
            iter_relaxed_structures(da, Path(db_file), chunk_size=chunk_size)
        )

    statements = [sql for proxy in proxies for sql in proxy.statements]
    return yielded, statements


def _count_open_files() -> int:
    """Count number of open file descriptors for current process."""
    try:
        pid = os.getpid()
        fd_dir = f"/proc/{pid}/fd"
        if os.path.exists(fd_dir):
            return len(os.listdir(fd_dir))
    except (OSError, PermissionError):
        pass
    return -1


# Mirrors TorchSim GA batch writes under CI load: production busy_timeout with
# a slightly more patient retry budget than PRESET_AGGRESSIVE (still transient-
# lock only). Two parallel jobs × one GA-sized batch each.
_STRESS_WRITE_RETRY = RetryConfig(
    max_retries=12,
    initial_delay=0.25,
    max_delay=8.0,
    backoff_factor=1.8,
)
_CONCURRENT_STRESS_WORKERS = 2
_CONCURRENT_STRESS_BATCH_SIZE = 5


def _write_to_database(args):
    """Helper function for multiprocess database writing."""
    db_path, batch_size, worker_id = args

    # Stagger workers so both processes do not open and write on the same tick.
    time.sleep(0.05 * worker_id)

    atoms_list = []
    for i in range(batch_size):
        atoms = create_test_atoms(
            ["Pt", "Pt"],
            positions=[[0, 0, 0], [2.5 + i * 0.1, 0, 0]],
            raw_score=-10.0 - i * 0.1,
        )
        atoms.info["data"] = {"worker_tag": f"w{worker_id}"}
        atoms_list.append(atoms)

    with get_connection(db_path) as da:
        for atoms in atoms_list:
            database_retry(
                lambda _a=atoms: da.add_unrelaxed_candidate(
                    _a, description=f"concurrent_stress:w{worker_id}"
                ),
                config=_STRESS_WRITE_RETRY,
                operation_name=f"concurrent_stress_write:w{worker_id}:unrelaxed",
            )
            database_retry(
                lambda _a=atoms: da.add_relaxed_step(_a),
                config=_STRESS_WRITE_RETRY,
                operation_name=f"concurrent_stress_write:w{worker_id}:relaxed",
            )
    return True, worker_id


class TestDatabaseSetupAndFlow:
    """Core database setup and integration workflows."""

    def test_setup_database_schema(self, tmp_path, pt3_atoms):
        """setup_database creates a valid SQLite database schema."""
        with _setup_test_db(
            tmp_path, "test.db", pt3_atoms, initial_candidate=pt3_atoms
        ) as (_, db_file):
            pass

        assert db_file.exists()

        conn = sqlite3.connect(db_file)
        try:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = {row[0] for row in cursor.fetchall()}
            cursor.close()
        finally:
            conn.close()

        assert "systems" in tables

        conn = sqlite3.connect(db_file)
        try:
            cursor = conn.execute("PRAGMA table_info(systems);")
            systems_columns = {row[1] for row in cursor.fetchall()}
            cursor.close()
        finally:
            conn.close()

        assert {"id", "energy", "fmax"}.issubset(systems_columns)

    def test_database_error_handling(self, tmp_path, pt3_atoms):
        """Invalid candidates are handled gracefully."""
        with _setup_test_db(tmp_path, "test.db", pt3_atoms, initial_candidate=None) as (
            da,
            _,
        ):
            invalid_atoms = Atoms(
                "Au3", positions=[[0, 0, 0], [2.5, 0, 0], [1.25, 2.165, 0]]
            )

            # Expect the database to raise a validation error when adding an invalid atom set
            with pytest.raises(SCGOValidationError, match="Candidate composition"):
                da.add_relaxed_step(invalid_atoms)

            # Ensure retrieving candidates still returns a valid list
            assert isinstance(da.get_all_relaxed_candidates(), list)

    @pytest.mark.parametrize(
        "template,candidate,expected_symbols,should_raise",
        [
            pytest.param(
                Atoms(["Pt", "Au"], positions=[[0, 0, 0], [2.5, 0, 0]]),
                Atoms(["Au", "Pt"], positions=[[0, 0, 0], [2.5, 0, 0]]),
                ["Pt", "Au"],
                False,
                id="permuted_atomic_order",
            ),
            pytest.param(
                Atoms(
                    ["Pt", "Au", "Au"],
                    positions=[[0, 0, 0], [2.5, 0, 0], [0, 2.5, 0]],
                ),
                Atoms(
                    ["Au", "Pt", "Au"],
                    positions=[[0, 0, 0], [2.5, 0, 0], [0, 2.5, 0]],
                ),
                ["Pt", "Au", "Au"],
                False,
                id="permuted_with_duplicates",
            ),
            pytest.param(
                Atoms("Pt2"),
                Atoms("Pt", positions=[[0, 0, 0]]),
                None,
                True,
                id="rejects_different_counts",
            ),
        ],
    )
    def test_add_relaxed_step_atomic_order_and_stoichiometry(
        self, tmp_path, template, candidate, expected_symbols, should_raise
    ):
        """Atomic order can be permuted (including duplicates) but stoichiometry cannot."""
        with _setup_test_db(tmp_path, "test.db", template, initial_candidate=None) as (
            da,
            _,
        ):
            if should_raise:
                with pytest.raises(SCGOValidationError, match="Candidate composition"):
                    da.add_relaxed_step(candidate)
                return

            _register_unrelaxed(da, candidate)
            da.add_relaxed_step(candidate)
            rows = da.get_all_relaxed_candidates()
            assert len(rows) == 1
            inserted = rows[0]
            assert Counter(inserted.get_chemical_symbols()) == Counter(expected_symbols)

    def test_add_relaxed_step_missing_raw_score_assigns_penalty(self, tmp_path):
        """If raw_score is missing and energy can't be computed, add_relaxed_step should
        assign PENALTY_ENERGY and set raw_score so GA runs continue instead of failing."""
        with _setup_test_db(
            tmp_path, "test.db", Atoms("Pt2"), initial_candidate=None
        ) as (da, _):
            a = Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])
            # Ensure no raw_score present
            a.info.pop("key_value_pairs", None)

            # Force get_potential_energy to raise to mimic a calculator failure
            def _bad_energy():
                raise RuntimeError("no energy")

            a.get_potential_energy = _bad_energy

            # Insert as unrelaxed candidate to ensure ASE-assigned identifiers exist
            _register_unrelaxed(da, a)

            # This should NOT raise; adapter will assign a penalty and insert
            da.add_relaxed_step(a)

            candidates = da.get_all_relaxed_candidates()
            assert len(candidates) == 1
            inserted = candidates[0]

            from scgo.constants import PENALTY_ENERGY
            from scgo.utils.helpers import extract_energy_from_atoms

            # Ensure energy extraction sees the penalty energy and raw_score exists
            energy = extract_energy_from_atoms(inserted)
            assert energy == PENALTY_ENERGY
            assert (
                inserted.info.get("key_value_pairs", {}).get("raw_score")
                == -PENALTY_ENERGY
            )

    def test_add_relaxed_step_persists_final_id(self, tmp_path):
        """Relaxed rows must carry final_id for mark_final_minima_in_db matching."""
        from scgo.metadata.atoms import ensure_final_id
        from scgo.utils.helpers import extract_energy_from_atoms

        with _setup_test_db(
            tmp_path, "test.db", Atoms("Pt2"), initial_candidate=None
        ) as (da, _):
            a = Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])
            a.calc = EMT()
            _register_unrelaxed(da, a)
            da.add_relaxed_step(a)

            inserted = da.get_all_relaxed_candidates()[0]
            kv = inserted.info.get("key_value_pairs", {})
            assert kv.get("final_id")
            energy = extract_energy_from_atoms(inserted)
            assert kv["final_id"] == ensure_final_id(inserted, energy)


class TestDatabaseConnections:
    """Connection interfaces."""

    def test_get_database_basic(self, tmp_path, pt2_atoms):
        with _setup_test_db(
            tmp_path, "test.db", pt2_atoms, initial_candidate=pt2_atoms
        ) as (_da, _db_path):
            pass

        with get_connection(tmp_path / "test.db") as db:
            assert isinstance(db.get_all_relaxed_candidates(), list)


class TestTransactions:
    """Transaction management utilities."""

    @pytest.mark.parametrize(
        "raise_inside,expected_delta",
        [
            pytest.param(False, 1, id="commit"),
            pytest.param(True, 0, id="rollback"),
        ],
    )
    def test_transaction_commit_or_rollback(
        self, tmp_path, pt2_atoms, raise_inside, expected_delta
    ):
        with _setup_test_db(tmp_path, "test.db", pt2_atoms, initial_candidate=None) as (
            _da,
            _db_path,
        ):
            pass

        with (
            get_connection(tmp_path / "test.db") as db,
            db.c.managed_connection() as conn,
        ):
            initial = conn.execute("SELECT COUNT(*) FROM systems").fetchone()[0]

        if raise_inside:
            with (
                pytest.raises(SCGOValidationError, match="Test error"),
                get_connection(tmp_path / "test.db") as db,
                database_transaction(db) as conn,
            ):
                conn.execute(
                    "INSERT INTO systems (username, numbers, positions, cell) "
                    "VALUES ('test', '[78,78]', '[[0,0,0],[2.5,0,0]]', "
                    "'[[10,0,0],[0,10,0],[0,0,10]]')"
                )
                raise SCGOValidationError("Test error")
        else:
            with (
                get_connection(tmp_path / "test.db") as db,
                database_transaction(db) as conn,
            ):
                conn.execute(
                    "INSERT INTO systems (username, numbers, positions, cell) "
                    "VALUES ('test', '[78,78]', '[[0,0,0],[2.5,0,0]]', "
                    "'[[10,0,0],[0,10,0],[0,0,10]]')"
                )

        with (
            get_connection(tmp_path / "test.db") as db,
            db.c.managed_connection() as conn,
        ):
            count = conn.execute("SELECT COUNT(*) FROM systems").fetchone()[0]

        assert count == initial + expected_delta

    def test_transaction_joins_already_open_transaction(self, tmp_path, pt2_atoms):
        """An outer (ASE-owned) transaction must be joined, not re-``BEGIN``-ed."""
        with _setup_test_db(tmp_path, "test.db", pt2_atoms, initial_candidate=None) as (
            _da,
            _db_path,
        ):
            pass

        insert_sql = (
            "INSERT INTO systems (username, numbers, positions, cell) "
            "VALUES (?, '[78,78]', '[[0,0,0],[2.5,0,0]]', "
            "'[[10,0,0],[0,10,0],[0,0,10]]')"
        )

        with (
            get_connection(tmp_path / "test.db") as db,
            db.c.managed_connection() as conn,
        ):
            initial = conn.execute("SELECT COUNT(*) FROM systems").fetchone()[0]

            # Mimic a pending ASE write: the connection is already in a transaction.
            conn.execute(insert_sql, ("outer",))
            if not conn.in_transaction:
                conn.execute("BEGIN")
            assert conn.in_transaction

            with database_transaction(db) as inner_conn:
                assert inner_conn is conn
                inner_conn.execute(insert_sql, ("inner",))

            # The joined transaction still belongs to the outer writer.
            assert conn.in_transaction
            conn.commit()

        with (
            get_connection(tmp_path / "test.db") as db,
            db.c.managed_connection() as conn,
        ):
            count = conn.execute("SELECT COUNT(*) FROM systems").fetchone()[0]
            usernames = {row[0] for row in conn.execute("SELECT username FROM systems")}

        assert count == initial + 2
        assert {"outer", "inner"}.issubset(usernames)

    def test_retry_transaction_context(self, tmp_path, pt2_atoms):
        with _setup_test_db(tmp_path, "test.db", pt2_atoms, initial_candidate=None) as (
            _da,
            _db_path,
        ):
            pass

        # Use retry_transaction for database transactions with lock retry
        from scgo.database.sync import RetryConfig, retry_transaction

        config = RetryConfig(max_retries=3, initial_delay=0.1)

        def _insert(conn):
            conn.execute(
                "INSERT INTO systems (username, numbers, positions, cell) "
                "VALUES ('test', '[78,78]', '[[0,0,0],[2.5,0,0]]', "
                "'[[10,0,0],[0,10,0],[0,0,10]]')"
            )

        with get_connection(tmp_path / "test.db") as db:
            retry_transaction(
                db,
                _insert,
                config=config,
                operation_name="transaction (test)",
            )

    def test_retry_transaction_retries_body_operational_error(
        self, tmp_path, pt2_atoms
    ):
        """Body-raised retryable errors must retry (callable API, not yield CM)."""
        with _setup_test_db(tmp_path, "test.db", pt2_atoms, initial_candidate=None) as (
            _da,
            _db_path,
        ):
            pass

        from scgo.database.sync import RetryConfig, retry_transaction

        attempts = {"n": 0}

        def _flaky(conn):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise sqlite3.OperationalError("database is locked")
            conn.execute(
                "INSERT INTO systems (username, numbers, positions, cell) "
                "VALUES ('retry', '[78,78]', '[[0,0,0],[2.5,0,0]]', "
                "'[[10,0,0],[0,10,0],[0,0,10]]')"
            )
            return "ok"

        with get_connection(tmp_path / "test.db") as db:
            result = retry_transaction(
                db,
                _flaky,
                config=RetryConfig(max_retries=3, initial_delay=0.01),
                operation_name="flaky body",
            )
        assert result == "ok"
        assert attempts["n"] == 2


class TestFilesystemSync:
    """Filesystem synchronization utilities."""

    def test_get_connection_retries_on_transient_lock(self, monkeypatch, tmp_path):
        """get_connection retries sqlite lock errors when opening."""
        import scgo.database.connection as conn_mod

        db_path = tmp_path / "test.db"
        db_path.touch()

        attempts = {"n": 0}
        original_dc = conn_mod.DataConnection

        def flaky_data_connection(path):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise sqlite3.OperationalError("database is locked")
            return original_dc(path)

        monkeypatch.setattr(conn_mod, "DataConnection", flaky_data_connection)
        monkeypatch.setattr("scgo.database.sync.time.sleep", lambda _: None)

        with get_connection(db_path) as db:
            assert db is not None

        assert attempts["n"] == 3

    def test_database_retry_skips_non_retryable_operational_error(self):
        """database_retry must not mask schema or SQL logic failures."""
        attempts = {"n": 0}

        def bad_query():
            attempts["n"] += 1
            raise sqlite3.OperationalError("no such table: systems")

        with pytest.raises(sqlite3.OperationalError, match="no such table"):
            database_retry(
                bad_query,
                config=RetryConfig(max_retries=5, initial_delay=0.01),
            )

        assert attempts["n"] == 1

    def test_database_retry_retries_transient_lock(self, monkeypatch):
        """database_retry retries lock errors that clear on a later attempt."""
        attempts = {"n": 0}

        def flaky_read():
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise sqlite3.OperationalError("database is locked")
            return "ok"

        monkeypatch.setattr("scgo.database.sync.time.sleep", lambda _: None)

        result = database_retry(
            flaky_read,
            config=RetryConfig(max_retries=5, initial_delay=0.01),
        )
        assert result == "ok"
        assert attempts["n"] == 3

    def test_database_retry_oserror_exception_types(self):
        attempts = []

        def flaky_operation():
            attempts.append(1)
            if len(attempts) < 3:
                raise OSError("Transient error")
            return "success"

        result = database_retry(
            flaky_operation,
            config=RetryConfig(max_retries=5, initial_delay=0.01),
            exception_types=(OSError,),
        )
        assert result == "success"
        assert len(attempts) == 3


class TestDiscovery:
    """Database discovery."""

    def test_discovery_find_databases(self, tmp_path):
        run_dir = tmp_path / "run_20260204_120000"
        run_dir.mkdir(parents=True)

        atoms = Atoms("Pt3")
        da = setup_database(run_dir, "ga_go.db", atoms, initial_candidate=atoms)
        close_data_connection(da)
        del da

        discovery = DatabaseDiscovery(tmp_path)
        db_files = discovery.find_databases()

        assert any(Path(str(f)).name == "ga_go.db" for f in db_files)

    def test_discovery_filter_by_run(self, tmp_path):
        for run_num in range(2):
            run_dir = tmp_path / f"run_2026020{run_num}_120000"
            run_dir.mkdir(parents=True)

            atoms = Atoms("Pt3")
            da = setup_database(run_dir, "ga_go.db", atoms, initial_candidate=atoms)
            close_data_connection(da)
        del da

        discovery = DatabaseDiscovery(tmp_path)
        db_files = discovery.find_databases(run_id="run_20260200_120000")

        assert len(db_files) == 1

    def test_find_databases_uncached(self, tmp_path):
        run_dir = tmp_path / "run_20260204_120000"
        run_dir.mkdir(parents=True)

        atoms = Atoms("Pt3")
        with _setup_test_db(run_dir, "ga_go.db", atoms, initial_candidate=atoms) as (
            _da,
            _db_path,
        ):
            pass

        discovery = DatabaseDiscovery(tmp_path)
        db_files = discovery.find_databases(db_filename="*.db", use_cache=False)
        assert db_files

    def test_empty_result_is_not_cached(self, tmp_path):
        """Discovery must see DBs written after an earlier miss or hit.

        GO may query before any DB exists, then write one; or load prior DBs
        then write the current run. Same-process TS reload must see all of them.
        """
        clear_discovery_cache()
        discovery = DatabaseDiscovery(tmp_path)
        atoms = Atoms("Pt3")

        assert discovery.find_databases() == []

        run1 = tmp_path / "run_20260204_120000"
        run1.mkdir(parents=True)
        da = setup_database(run1, "ga_go.db", atoms, initial_candidate=atoms)
        close_data_connection(da)
        del da

        assert len(discovery.find_databases()) == 1
        assert len(list_discovered_db_paths_with_run(tmp_path)) == 1

        run2 = tmp_path / "run_20260204_130000"
        run2.mkdir(parents=True)
        da2 = setup_database(run2, "ga_go.db", atoms, initial_candidate=atoms)
        close_data_connection(da2)
        del da2

        found = discovery.find_databases()
        assert len(found) == 2
        assert {Path(str(p)).parent.name for p in found} == {
            "run_20260204_120000",
            "run_20260204_130000",
        }
        assert len(list_discovered_db_paths_with_run(tmp_path, use_cache=True)) == 2


class TestRobustness:
    """Robustness, concurrency, and retry behavior."""

    def test_context_manager_exception_cleanup(self, tmp_path, pt2_atoms):
        with _setup_test_db(
            tmp_path, "test.db", pt2_atoms, initial_candidate=pt2_atoms
        ) as (_da, _db_path):
            pass

        with (
            pytest.raises(KeyError, match="Test exception"),
            get_connection(tmp_path / "test.db") as db,
        ):
            _ = db.get_all_relaxed_candidates()
            raise KeyError("Test exception")

        with get_connection(tmp_path / "test.db") as db:
            assert isinstance(db.get_all_relaxed_candidates(), list)

    @pytest.mark.slow
    def test_no_file_handle_leak_many_connections(self, tmp_path, pt2_atoms):
        initial_fd_count = _count_open_files()
        if initial_fd_count < 0:
            pytest.skip("Cannot count file descriptors on this system")

        for i in range(25):
            with _setup_test_db(
                tmp_path,
                f"test_{i}.db",
                pt2_atoms,
                initial_candidate=pt2_atoms,
            ) as (_da, _db_path):
                pass
            with get_connection(tmp_path / f"test_{i}.db") as db:
                _ = db.get_all_relaxed_candidates()

        gc.collect()

        final_fd_count = _count_open_files()
        fd_increase = final_fd_count - initial_fd_count
        assert fd_increase < 5

    @pytest.mark.slow
    def test_add_ts_to_database_no_file_handle_leak(self, tmp_path, pt2_atoms):
        from scgo.ts_search.ts_network import add_ts_to_database
        from tests.helpers import mark_test_minima_as_final

        with _setup_test_db(
            tmp_path,
            "ts_leak.db",
            pt2_atoms,
            initial_candidate=pt2_atoms,
        ) as (_da, db_path):
            pass

        mark_test_minima_as_final(db_path)

        initial_fd_count = _count_open_files()
        if initial_fd_count < 0:
            pytest.skip("Cannot count file descriptors on this system")

        ts = pt2_atoms.copy()
        ts.info.setdefault("key_value_pairs", {})["raw_score"] = -0.5

        for i in range(25):
            assert add_ts_to_database(
                ts_structure=ts,
                ts_energy=0.5,
                minima_idx_1=0,
                minima_idx_2=1,
                db_file=str(db_path),
                pair_id=f"{i}_1",
                barrier_height=0.1,
            )

        gc.collect()

        fd_increase = _count_open_files() - initial_fd_count
        assert fd_increase < 5

    @pytest.mark.slow
    @pytest.mark.xdist_group(name="no_nested_pool")
    def test_concurrent_write_stress(self, tmp_path, pt2_atoms):
        """Two processes append GA-sized batches using production retry settings."""
        with _setup_test_db(
            tmp_path,
            "concurrent.db",
            pt2_atoms,
            initial_candidate=None,
            enable_wal_mode=True,
        ) as (_da, db_file):
            pass

        n_workers = _CONCURRENT_STRESS_WORKERS
        batch_size = _CONCURRENT_STRESS_BATCH_SIZE

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(_write_to_database, (str(db_file), batch_size, wid))
                for wid in range(n_workers)
            ]

            results = [future.result() for future in as_completed(futures)]

        assert all(r[0] for r in results), f"Worker failures: {results}"

        n_relaxed = len(list(iter_database_minima(db_file)))
        assert n_relaxed == n_workers * batch_size

    def test_setup_database_wal_mode(self, tmp_path, pt2_atoms):
        with _setup_test_db(
            tmp_path,
            "test.db",
            pt2_atoms,
            initial_candidate=pt2_atoms,
            enable_wal_mode=True,
        ) as (_da, _db_path):
            pass

        conn = sqlite3.connect(str(tmp_path / "test.db"))
        try:
            cur = conn.execute("PRAGMA journal_mode;")
            mode = cur.fetchone()[0]
            cur.close()
        finally:
            conn.close()

        assert mode.lower() == "wal"

    def test_open_data_connection_for_setup_applies_pragmas_then_closes(
        self, tmp_path, pt2_atoms
    ):
        from ase.db import connect as ase_db_connect
        from ase_ga.data import DataConnection

        from scgo.database.connection import (
            _apply_scgo_sqlite_settings,
            open_data_connection_for_setup,
        )

        db_path = tmp_path / "setup_pragmas.db"
        with ase_db_connect(str(db_path)) as prep_db:
            prep_db.write(pt2_atoms, simulation_cell=True)

        # Direct helper path: PRAGMAs are visible on a live connection.
        da_live = DataConnection(str(db_path))
        _apply_scgo_sqlite_settings(
            da_live,
            busy_timeout=99999,
            cache_size_mb=128,
            wal_mode=False,
            close_after=False,
        )
        try:
            conn = da_live.c.connection
            assert conn is not None
            assert conn.execute("PRAGMA busy_timeout;").fetchone()[0] == 99999
            assert conn.execute("PRAGMA cache_size;").fetchone()[0] == -(128 * 1024)
        finally:
            close_data_connection(da_live)

        # Setup opener must leave the ASE backend closed for later ``with da.c:``.
        da = open_data_connection_for_setup(
            db_path, busy_timeout=99999, cache_size_mb=128, wal_mode=False
        )
        try:
            assert da.c.connection is None
        finally:
            close_data_connection(da)

    def test_setup_database_reuse_keeps_single_simulation_cell(
        self, tmp_path, pt2_atoms
    ):
        """Resume (remove_existing=False) must not write a second template row."""
        from ase.db import connect as ase_db_connect

        da = setup_database(
            tmp_path,
            "reuse.db",
            pt2_atoms,
            initial_candidate=pt2_atoms,
            remove_existing=True,
        )
        close_data_connection(da)

        da2 = setup_database(
            tmp_path,
            "reuse.db",
            pt2_atoms,
            remove_existing=False,
        )
        close_data_connection(da2)

        db_file = tmp_path / "reuse.db"
        with ase_db_connect(str(db_file)) as db:
            assert len(list(db.select(simulation_cell=True))) == 1

        other = Atoms("Cu2", positions=[[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
        with pytest.raises(SCGOValidationError, match="stoichiometry"):
            setup_database(
                tmp_path,
                "reuse.db",
                other,
                remove_existing=False,
            )


class TestDatabaseStreaming:
    """Test streaming iterators."""

    def test_iter_database_minima(self, tmp_path, rng):
        """Test iterating over database minima."""
        db_file = tmp_path / "test.db"
        atoms = Atoms("Pt3", positions=[[0, 0, 0], [2.5, 0, 0], [1.25, 2.0, 0]])
        da = setup_database(tmp_path, "test.db", atoms, initial_candidate=atoms)

        for i in range(10):
            a = atoms.copy()
            a.positions += rng.random((3, 3)) * 0.1

            set_tags(a, raw_score=-30.0 - i)
            a.info["data"] = {"tag": f"test_{i}"}
            da.add_relaxed_step(a)

        close_data_connection(da)
        del da

        count = 0
        for energy, atoms_obj in iter_database_minima(db_file):
            assert isinstance(energy, float)
            assert isinstance(atoms_obj, Atoms)
            count += 1

        assert count > 0

    def test_iter_database_minima_chunked(self, tmp_path, rng, monkeypatch):
        """Ensure streaming honors chunk_size and does not call get_all_relaxed_candidates."""
        db_file = tmp_path / "test.db"
        atoms = Atoms("Pt3", positions=[[0, 0, 0], [2.5, 0, 0], [1.25, 2.0, 0]])
        da = setup_database(tmp_path, "test.db", atoms, initial_candidate=atoms)

        for i in range(10):
            a = atoms.copy()
            a.positions += rng.random((3, 3)) * 0.1
            a.info["key_value_pairs"] = {"raw_score": -30.0 - i}
            a.info["data"] = {"tag": f"test_{i}"}
            da.add_relaxed_step(a)

        close_data_connection(da)
        del da

        # Prevent accidental use of the in-memory loader
        from ase_ga.data import DataConnection

        def _fail(*args, **kwargs):
            raise AssertionError(
                "get_all_relaxed_candidates must not be called during streaming"
            )

        monkeypatch.setattr(DataConnection, "get_all_relaxed_candidates", _fail)

        yielded = list(iter_database_minima(db_file, chunk_size=3))
        assert len(yielded) == 10
        assert all(isinstance(e, float) for e, _ in yielded)

    def test_iter_database_minima_logs_row_failure_and_continues(
        self, tmp_path, rng, monkeypatch, caplog
    ):
        """Row-level failures should be logged and streaming must continue."""
        db_file = tmp_path / "test.db"
        atoms = Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])
        da = setup_database(tmp_path, "test.db", atoms, initial_candidate=atoms)

        # Add a few relaxed rows
        for i in range(3):
            a = atoms.copy()
            a.positions += rng.random((2, 3)) * 0.1

            set_tags(a, raw_score=-10.0 - i)
            a.info["data"] = {"tag": f"test_{i}"}
            da.add_relaxed_step(a)

        close_data_connection(da)
        del da

        # Make row decoding fail once to simulate a malformed blob row.
        from ase.db.row import AtomsRow

        orig_toatoms = AtomsRow.toatoms
        call = {"n": 0}

        def _maybe_fail(self, *args, **kwargs):
            call["n"] += 1
            if call["n"] == 1:
                raise ValueError("simulated malformed row")
            return orig_toatoms(self, *args, **kwargs)

        monkeypatch.setattr(AtomsRow, "toatoms", _maybe_fail)

        caplog.clear()
        items = list(iter_database_minima(db_file))

        # Ensure streaming returned remaining rows and skipped the failing one
        assert len(items) >= 2
        assert any(
            "Failed to decode atoms id=" in rec.message
            for rec in caplog.records
            if rec.levelname == "WARNING"
        )

    def test_streaming_returns_every_relaxed_row(self, tmp_path):
        """Every relaxed row is streamed with its energy and systems_row_id tag."""
        n_rows = 5
        db_file = _build_relaxed_db(tmp_path, "stream.db", n_rows)

        yielded, _statements = _stream_with_recording_connection(db_file)

        assert len(yielded) == n_rows
        assert sorted(e for e, _ in yielded) == [10.0 + i for i in range(n_rows)]

        row_ids = []
        for energy, atoms_obj in yielded:
            row_id = get_tag(atoms_obj, "systems_row_id")
            assert isinstance(row_id, int)
            row_ids.append(row_id)
            assert len(atoms_obj) == 2
            assert atoms_obj.get_chemical_symbols() == ["Pt", "Pt"]
            assert get_tag(atoms_obj, "raw_score") == pytest.approx(-energy)
        assert len(set(row_ids)) == n_rows

        with sqlite3.connect(db_file) as raw:
            db_ids = {
                row[0]
                for row in raw.execute(
                    "SELECT id FROM systems "
                    "WHERE json_extract(key_value_pairs, '$.relaxed') = 1"
                )
            }
        assert set(row_ids) == db_ids

    def test_streaming_does_not_issue_bulk_select_star(self, tmp_path, monkeypatch):
        """Chunk load uses ``WHERE id IN``; no unbounded ``SELECT *`` or N get_atoms."""
        from ase_ga.data import DataConnection

        db_file = _build_relaxed_db(tmp_path, "stream_no_bulk.db", 4)
        get_atoms_calls: list[int] = []
        real_get_atoms = DataConnection.get_atoms

        def _counting_get_atoms(self, row_id, *a, **k):
            get_atoms_calls.append(int(row_id))
            return real_get_atoms(self, row_id, *a, **k)

        monkeypatch.setattr(DataConnection, "get_atoms", _counting_get_atoms)

        yielded, statements = _stream_with_recording_connection(db_file)

        assert len(yielded) == 4
        unbounded = [
            s
            for s in statements
            if "SELECT * FROM systems" in s and "WHERE ID IN" not in s.upper()
        ]
        assert unbounded == [], f"unexpected unbounded bulk query: {unbounded}"
        assert any("WHERE ID IN" in s.upper() for s in statements)
        assert get_atoms_calls == []

    def test_streaming_survives_failing_bulk_query(self, tmp_path):
        """A failing bulk id-IN query must fall back without ``UnboundLocalError``."""
        n_rows = 4
        db_file = _build_relaxed_db(tmp_path, "stream_failure.db", n_rows)

        try:
            yielded, _statements = _stream_with_recording_connection(
                db_file, fail_on="WHERE id IN"
            )
        except UnboundLocalError as exc:  # pragma: no cover - regression guard
            pytest.fail(f"chunk loader leaked UnboundLocalError: {exc}")

        assert len(yielded) == n_rows
        assert sorted(e for e, _ in yielded) == [10.0 + i for i in range(n_rows)]


class TestRunIdPersistence:
    """Persisting ``run_id`` back into database rows."""

    def test_persisted_run_id_is_queryable_via_ase_select(self, tmp_path):
        """``persist=True`` must update ASE's key-index tables, not just JSON."""
        from ase.db import connect as ase_db_connect

        from scgo.database.helpers import extract_minima_from_database_file

        run_dir = tmp_path / "run_20250101_000000_000000"
        run_dir.mkdir(parents=True)
        n_rows = 3
        db_file = _build_relaxed_db(run_dir, "ga_go.db", n_rows)

        run_id = "run_persist_index"
        minima = extract_minima_from_database_file(db_file, run_id, persist=True)
        assert len(minima) == n_rows

        with ase_db_connect(str(db_file)) as ase_db:
            selected = list(ase_db.select(run_id=run_id))

        assert len(selected) == n_rows
        assert all(row.key_value_pairs.get("run_id") == run_id for row in selected)
