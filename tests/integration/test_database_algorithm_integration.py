"""BH/GA algorithm-to-database integration tests.

These are full-workflow tests (real BH and GA runs that write databases), so they
are marked ``integration`` and ``slow``. Fast CI excludes ``integration``; the
CPU slow job selects ``slow and not benchmark`` (and therefore these tests).
"""

from __future__ import annotations

import pytest
from ase.calculators.emt import EMT

from scgo.algorithms import ga_go
from scgo.algorithms.basinhopping_go import bh_go
from scgo.database import get_connection
from tests.helpers import assert_run_id_persisted


@pytest.mark.slow
@pytest.mark.integration
def test_algorithm_database_integration(tmp_path, pt3_atoms, rng):
    """BH and GA integration creates database entries."""
    # Test BH
    atoms_bh = pt3_atoms.copy()
    atoms_bh.calc = EMT()
    _ = bh_go(
        atoms=atoms_bh,
        output_dir=str(tmp_path / "bh_test"),
        niter=1,
        dr=0.2,
        niter_local_relaxation=2,
        rng=rng,
    )
    db_bh = tmp_path / "bh_test" / "bh_go.db"
    assert db_bh.exists()

    with get_connection(db_bh) as db:
        assert len(db.get_all_relaxed_candidates()) > 0

    # Test GA
    calc_ga = EMT()
    _ = ga_go(
        composition=["Pt", "Pt", "Pt"],
        output_dir=str(tmp_path / "ga_test"),
        calculator=calc_ga,
        niter=1,
        population_size=2,
        niter_local_relaxation=2,
        rng=rng,
    )
    db_ga = tmp_path / "ga_test" / "ga_go.db"
    assert db_ga.exists()

    with get_connection(db_ga) as db:
        assert len(db.get_all_relaxed_candidates()) > 0


@pytest.mark.slow
@pytest.mark.integration
def test_ga_runs_store_run_id_in_key_value_pairs(tmp_path, rng):
    """Running `ga_go` with a `run_id` should persist it in key_value_pairs for
    relaxed candidates (so discovery/filtering by run_id works)."""
    run_id = "run_test_write"
    outdir = tmp_path / "ga_run"
    _ = ga_go(
        composition=["Pt"] * 5,
        output_dir=str(outdir),
        calculator=EMT(),
        niter=1,
        population_size=2,
        niter_local_relaxation=1,
        rng=rng,
        run_id="run_test_write",
        clean=True,
    )

    db_file = outdir / "ga_go.db"
    assert db_file.exists()

    with get_connection(db_file) as da:
        rows = da.get_all_relaxed_candidates()

    matched = []
    for r in rows:
        try:
            assert_run_id_persisted(r, run_id)
            matched.append(r)
        except AssertionError:
            continue

    assert matched, "No relaxed candidates had run_id stored in key_value_pairs"
