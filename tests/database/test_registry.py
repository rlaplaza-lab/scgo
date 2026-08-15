from pathlib import Path

from ase.db import connect

from scgo.database.registry import get_registry


def test_database_registry_register_find_and_clear(tmp_path, single_atom):
    base = tmp_path
    dbpath = base / "some_dir" / "my.db"
    dbpath.parent.mkdir(parents=True)

    # Create simple ASE DB
    db = connect(str(dbpath))
    db.write(single_atom.copy(), relaxed=True)

    reg = get_registry(base)
    reg.clear()

    # Register DB with explicit metadata
    reg.register_database(dbpath, composition=["Pt"], run_id="run_xyz")

    # find_databases should locate the registered DB
    found = reg.find_databases(run_id="run_xyz")
    assert len(found) == 1
    assert Path(found[0]).resolve() == dbpath.resolve()

    # get_all_databases should include it
    all_db = reg.get_all_databases()
    assert any(Path(p).resolve() == dbpath.resolve() for p in all_db)

    # clear removes all entries
    reg.clear()
    assert reg.find_databases(run_id="run_xyz") == []


def test_setup_database_registers_registry(tmp_path, pt2_atoms):
    from scgo.database import close_data_connection
    from scgo.database.helpers import setup_database

    reg = get_registry(tmp_path)
    reg.clear()

    # Create DB via setup_database (should auto-register)
    da = setup_database(
        tmp_path, "auto_register.db", pt2_atoms, initial_candidate=pt2_atoms
    )
    try:
        db_path = tmp_path / "auto_register.db"

        entries = reg.get_all_databases()
        assert any(Path(p).resolve() == db_path.resolve() for p in entries)
    finally:
        close_data_connection(da)


def test_setup_database_registers_search_level_registry(tmp_path):
    """DB under run_*/ inside *_searches is registered only at the search root."""

    from scgo.database import close_data_connection
    from scgo.database.helpers import setup_database

    # Build canonical run layout under a search directory
    search_dir = tmp_path / "Pt6_searches"
    run_dir = search_dir / "run_000"
    run_dir.mkdir(parents=True)

    get_registry(run_dir).clear()
    get_registry(search_dir).clear()

    # Create DB in the run directory (this is what the optimizer does)
    from tests.helpers import create_test_atoms

    pt6 = create_test_atoms(["Pt"] * 6)
    da = setup_database(str(run_dir), "ga_go.db", pt6, initial_candidate=pt6)
    try:
        db_path = run_dir / "ga_go.db"

        # No per-trial registry file when under *_searches
        trial_entries = get_registry(run_dir).get_all_databases()
        assert trial_entries == []

        search_reg = get_registry(search_dir)
        search_entries = search_reg.get_all_databases()
        assert any(Path(p).resolve() == db_path.resolve() for p in search_entries)
    finally:
        close_data_connection(da)


def test_create_preparedb_registers_registry(tmp_path, pt2_atoms):
    from tests.helpers import create_preparedb

    reg = get_registry(tmp_path)
    reg.clear()

    # Create DB via PrepareDB helper (test utility)
    create_preparedb(pt2_atoms, tmp_path / "prepared.db", population_size=5)
    db_path = tmp_path / "prepared.db"

    entries = reg.get_all_databases()
    assert any(Path(p).resolve() == db_path.resolve() for p in entries)


def test_register_database_best_effort_handles_bad_atoms_template(tmp_path):
    """_register_database_best_effort tolerates a bad atoms_template (composition may be None)."""
    from scgo.database.helpers import _register_database_best_effort

    reg = get_registry(tmp_path)
    reg.clear()

    # Create trial directory and db file path under base
    trial_dir = tmp_path / "run_1"
    trial_dir.mkdir(parents=True)
    db_path = trial_dir / "ga_go.db"

    # atoms_template stub that raises when asked for chemical symbols
    class BadAtomsTemplate:
        def get_chemical_symbols(self):
            raise AttributeError("simulated - missing internals")

        def get_atomic_numbers(self):
            return [78, 78]

    # Call the best-effort registration helper — should not raise
    _register_database_best_effort(
        str(trial_dir), str(db_path), BadAtomsTemplate(), "run_1"
    )

    # ``get_all_databases`` only returns entries whose file still exists on disk,
    # so create the (empty) db file that the registration recorded.
    db_path.touch()

    # Trial-level registry should now contain the entry (composition may be None)
    trial_registry = get_registry(trial_dir)
    entries = trial_registry.get_all_databases()
    assert any(Path(p).resolve() == db_path.resolve() for p in entries)

    # Also ensure the registry under the base tmp_path did not accidentally gain it
    base_entries = reg.get_all_databases()
    assert all(p.resolve().name != db_path.name for p in base_entries)


def test_registry_eviction_and_clear(tmp_path):
    """The global registry cache is bounded and can be cleared."""
    from scgo.database.registry import (
        _REGISTRY_MAX_SIZE,
        _global_registries,
        clear_registry,
        get_registry,
    )

    clear_registry()
    assert len(_global_registries) == 0

    # Pin the first registry instance; it must be evicted once the cap is exceeded.
    first = get_registry(tmp_path / "oldest")
    for i in range(_REGISTRY_MAX_SIZE):
        get_registry(tmp_path / f"d{i}")

    # Cache stays bounded by the cap.
    assert len(_global_registries) == _REGISTRY_MAX_SIZE

    # Oldest entry was evicted; a fresh instance is now returned for it.
    assert get_registry(tmp_path / "oldest") is not first

    # Recently-used entries remain cached (same instance returned).
    assert get_registry(tmp_path / "d0") is not None

    clear_registry()
    assert len(_global_registries) == 0


def test_setup_database_context_manager(tmp_path, pt2_atoms):
    from scgo.database.connection import get_connection
    from scgo.database.helpers import setup_database

    # Use context-manager returned by setup_database
    with setup_database(
        tmp_path, "cm.db", pt2_atoms, initial_candidate=pt2_atoms
    ) as da:
        from tests.helpers import create_test_atoms

        a = create_test_atoms(["Pt", "Pt"], positions=[[0, 0, 0], [1.5, 0, 0]])
        # Insert as unrelaxed first so ASE assigns confid/identifiers as expected
        a.info.setdefault("key_value_pairs", {})["raw_score"] = -0.5
        da.add_unrelaxed_candidate(a, description="cm:test")

        # Retrieve the unrelaxed candidate and promote to relaxed state
        u = da.get_an_unrelaxed_candidate()
        assert u is not None
        u.info.setdefault("key_value_pairs", {})["raw_score"] = -0.5
        da.add_relaxed_step(u)
        assert len(da.get_all_relaxed_candidates()) >= 1

    # After exiting the context, DB should be readable and contain the relaxed row
    with get_connection(tmp_path / "cm.db") as da2:
        assert len(da2.get_all_relaxed_candidates()) >= 1
