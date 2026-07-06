"""End-to-end SCGO workflow tests."""

import os
import sqlite3
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT
from ase.io import read

from scgo.minima_search import run_trials
from scgo.param_presets import get_testing_params
from scgo.runner_api import (
    _run_go_campaign_compositions,
    _run_go_trials,
    build_one_element_compositions,
    build_two_element_compositions,
)
from scgo.utils.run_helpers import initialize_params
from tests.constants import REPRODUCIBILITY_ATOL, REPRODUCIBILITY_RTOL
from tests.test_utils import (
    assert_exported_minima_xyz_equal,
    assert_minima_lists_equal,
    isolated_workflow_cwd,
)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize(
    "optimizer,opt_kwargs",
    [
        (
            "bh",
            {
                "niter": 2,
                "dr": 0.2,
                "niter_local_relaxation": 2,
                "temperature": 0.01,
                "system_type": "gas_cluster",
            },
        ),
        (
            "ga",
            {
                "niter": 1,
                "population_size": 2,
                "niter_local_relaxation": 2,
                "mutation_probability": 0.3,
                "vacuum": 8.0,
                "n_jobs_population_init": 1,
                "system_type": "gas_cluster",
            },
        ),
    ],
)
def test_full_optimizer_workflow(tmp_path, rng, optimizer, opt_kwargs):
    """Test complete optimization workflow from initialization to output files for any optimizer."""
    comp = ["Pt", "Pt", "Pt"]
    output_dir = str(tmp_path / f"{optimizer}_campaign")

    # Run a minimal campaign
    results = run_trials(
        composition=comp,
        global_optimizer=optimizer,
        global_optimizer_kwargs=opt_kwargs,
        output_dir=output_dir,
        calculator_for_global_optimization=EMT(),
        validate_with_hessian=False,
        rng=rng,
    )

    # Verify results structure
    assert isinstance(results, list)
    if results:
        for energy, atoms in results:
            assert np.isfinite(energy)
            assert isinstance(atoms, Atoms)
            assert len(atoms) == 3
            assert atoms.get_chemical_symbols() == comp

    assert os.path.exists(output_dir)

    from scgo.utils.run_tracking import get_run_directories

    run_dirs = get_run_directories(output_dir)
    assert len(run_dirs) > 0
    run_dir = run_dirs[0]

    run_dir = run_dirs[0]
    assert os.path.exists(run_dir)
    assert os.path.exists(os.path.join(output_dir, "final_unique_minima"))

    db_name = f"{optimizer}_go.db"
    db_path = os.path.join(run_dir, db_name)
    assert os.path.exists(db_path)

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        assert "systems" in tables
        cols = [r[1] for r in cursor.execute("PRAGMA table_info(systems)").fetchall()]

        # Check for run_id persistence
        cursor.execute(
            f"SELECT {'metadata, ' if 'metadata' in cols else ''}key_value_pairs FROM systems LIMIT 5"
        )
        rows = cursor.fetchall()
        found_runid = False
        for row in rows:
            import json

            meta = json.loads(row[0]) if len(row) > 1 and row[0] else {}
            kv = json.loads(row[-1]) if row[-1] else {}
            if meta.get("run_id") or kv.get("run_id"):
                found_runid = True
                break
        assert found_runid is True, f"No run_id persisted in {optimizer} DB rows"

    # Verify GA-specific logs
    if optimizer == "ga":
        assert os.path.exists(os.path.join(run_dir, "population.log"))

    # Verify XYZ files
    if results:
        xyz_dir = os.path.join(output_dir, "final_unique_minima")
        xyz_files = list(Path(xyz_dir).glob("*.xyz"))
        if xyz_files:
            atoms_from_file = read(str(xyz_files[0]))
            assert len(atoms_from_file) == 3
            assert "provenance" in atoms_from_file.info
            assert atoms_from_file.info["provenance"].get("run_id")


@pytest.mark.integration
@pytest.mark.slow
def test_full_workflow_reproducible_with_fixed_seed(tmp_path):
    """End-to-end GA workflow is repeatable when re-run with the same seed.

    Exercises the public runner path (:func:`scgo.runner_api._run_go_trials` →
    algorithm selection → :func:`scgo.minima_search.run_trials` →
    :func:`scgo.minima_search.scgo` → DB persistence → deduplication →
    ``final_unique_minima`` export). Uses fixed mutation weights (adaptive disabled)
    and relies on explicit NumPy RNG plumbing only (no ``random.seed``).
    """
    import random

    from scgo.utils.run_tracking import get_run_directories

    composition = ["Pt", "Pt", "Pt", "Pt"]
    seed = 271828
    params = get_testing_params()
    params["validate_with_hessian"] = False
    params["optimizer_params"]["ga"].update(
        {
            "niter": 1,
            "population_size": 3,
            "niter_local_relaxation": 2,
            "mutation_probability": 0.3,
            "n_jobs_population_init": 1,
            "n_jobs_offspring": 1,
            "use_adaptive_mutations": False,
            "previous_search_glob": ".__scgo_no_prior_runs__/**/*.db",
        }
    )
    merged_params = initialize_params(deepcopy(params))
    assert merged_params["optimizer_params"]["ga"]["use_adaptive_mutations"] is False

    def _run_once(output_dir: Path) -> list[tuple[float, Atoms]]:
        return _run_go_trials(
            composition,
            system_type="gas_cluster",
            params=deepcopy(params),
            seed=seed,
            verbosity=0,
            clean=True,
            output_dir=output_dir,
            calculator_for_global_optimization=EMT(),
        )

    out_a = tmp_path / "workflow_a"
    out_b = tmp_path / "workflow_b"

    # Pollute Python's global RNG; reproducibility must not depend on re-seeding it.
    random.seed(0xDEADBEEF)
    random.random()

    with isolated_workflow_cwd(out_a):
        results1 = _run_once(out_a.resolve())

    random.random()

    with isolated_workflow_cwd(out_b):
        results2 = _run_once(out_b.resolve())

    assert_minima_lists_equal(
        results1,
        results2,
        rtol=REPRODUCIBILITY_RTOL,
        atol=REPRODUCIBILITY_ATOL,
    )

    for output_dir in (out_a, out_b):
        assert output_dir.is_dir()
        run_dirs = get_run_directories(str(output_dir))
        assert len(run_dirs) == 1
        run_dir = Path(run_dirs[0])

        final_xyz_dir = output_dir / "final_unique_minima"
        assert final_xyz_dir.is_dir()
        assert list(final_xyz_dir.glob("*.xyz")), "expected exported minima XYZ files"

        run_dir = Path(run_dirs[0])
        assert (run_dir / "ga_go.db").exists()

    assert_exported_minima_xyz_equal(
        out_a / "final_unique_minima",
        out_b / "final_unique_minima",
        rtol=REPRODUCIBILITY_RTOL,
        atol=REPRODUCIBILITY_ATOL,
    )


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_multicore
def test_full_workflow_parallel_offspring_reproducible(tmp_path):
    """Serial vs parallel offspring produce identical workflow results for a fixed seed."""
    composition = ["Pt", "Pt", "Pt", "Pt"]
    seed = 161803
    base_params = get_testing_params()
    base_params["validate_with_hessian"] = False

    def _run_once(output_dir: Path, n_jobs_offspring: int) -> list[tuple[float, Atoms]]:
        params = deepcopy(base_params)
        params["optimizer_params"]["ga"].update(
            {
                "niter": 1,
                "population_size": 3,
                "niter_local_relaxation": 2,
                "mutation_probability": 0.3,
                "n_jobs_population_init": 1,
                "n_jobs_offspring": n_jobs_offspring,
            }
        )
        return _run_go_trials(
            composition,
            system_type="gas_cluster",
            params=params,
            seed=seed,
            verbosity=0,
            clean=True,
            output_dir=output_dir,
            calculator_for_global_optimization=EMT(),
        )

    out_serial = tmp_path / "workflow_serial_offspring"
    out_parallel = tmp_path / "workflow_parallel_offspring"

    with isolated_workflow_cwd(out_serial):
        results_serial = _run_once(out_serial.resolve(), n_jobs_offspring=1)

    with isolated_workflow_cwd(out_parallel):
        results_parallel = _run_once(out_parallel.resolve(), n_jobs_offspring=2)

    assert_minima_lists_equal(
        results_serial,
        results_parallel,
        rtol=REPRODUCIBILITY_RTOL,
        atol=REPRODUCIBILITY_ATOL,
    )


def test_single_run_campaign(tmp_path, rng):
    """Test single run creates DB at run root."""
    comp = ["Pt", "Pt"]
    output_dir = str(tmp_path / "single_run")

    results = run_trials(
        composition=comp,
        global_optimizer="bh",
        global_optimizer_kwargs={
            "niter": 1,
            "dr": 0.3,
            "niter_local_relaxation": 2,
            "temperature": 0.01,
            "system_type": "gas_cluster",
        },
        output_dir=output_dir,
        calculator_for_global_optimization=EMT(),
        validate_with_hessian=False,
        rng=rng,
    )

    if len(results) > 1:
        energies = [energy for energy, _ in results]
        assert energies == sorted(energies)

    from scgo.utils.run_tracking import get_run_directories

    run_dirs = get_run_directories(output_dir)
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert os.path.exists(os.path.join(run_dir, "bh_go.db"))


@pytest.mark.slow
@pytest.mark.integration
def test_campaign_one_element(tmp_path):
    """Test single-element campaign workflow."""
    params = get_testing_params()

    # Run campaign for Pt2 and Pt3
    results = _run_go_campaign_compositions(
        build_one_element_compositions("Pt", 2, 3),
        system_type="gas_cluster",
        params=params,
        seed=456,
        output_dir=str(tmp_path / "campaign"),
    )

    # Verify results structure
    assert isinstance(results, dict)
    assert "Pt2" in results
    assert "Pt3" in results

    # Verify each composition has results
    for formula, minima_list in results.items():
        assert isinstance(minima_list, list)
        if minima_list:
            for energy, atoms in minima_list:
                assert np.isfinite(energy)
                assert isinstance(atoms, Atoms)
                assert atoms.get_chemical_formula() == formula


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize(
    "min_atoms,max_atoms,niter,population_size",
    [
        (2, 4, 1, 5),  # Small clusters: Pt2-4 (replaces test_runner_pt2_pt4.py)
        (5, 6, 3, 10),  # Medium clusters: Pt5-6 (replaces test_runner_pt5_pt6.py)
    ],
)
def test_campaign_one_element_varying_cluster_sizes(
    tmp_path, min_atoms, max_atoms, niter, population_size
):
    """Test single-element campaigns with varying cluster sizes and parameters.

    This test replaces the standalone test_runner_pt2_pt4.py and test_runner_pt5_pt6.py
    runners, providing coverage for:
    - Small clusters (Pt2-4) with minimal iterations (quick validation)
    - Medium clusters (Pt5-6) with moderate iterations (broader validation)
    """
    params = get_testing_params()

    # Adjust GA parameters based on cluster size
    params["optimizer_params"]["ga"]["niter"] = niter
    params["optimizer_params"]["ga"]["population_size"] = population_size
    params["optimizer_params"]["ga"]["n_jobs_population_init"] = 1  # Sequential

    # Run campaign for the specified cluster size range
    results = _run_go_campaign_compositions(
        build_one_element_compositions("Pt", min_atoms, max_atoms),
        system_type="gas_cluster",
        params=params,
        seed=456,
        output_dir=str(tmp_path / f"campaign_pt{min_atoms}_{max_atoms}"),
    )

    # Verify results structure
    assert isinstance(results, dict)

    # Check expected compositions are present
    expected_formulas = {f"Pt{i}" for i in range(min_atoms, max_atoms + 1)}
    actual_formulas = set(results.keys())
    assert expected_formulas == actual_formulas, (
        f"Expected formulas {expected_formulas}, got {actual_formulas}"
    )

    # Verify each composition has valid results
    total_structures = 0
    for formula, minima_list in results.items():
        assert isinstance(minima_list, list)
        if minima_list:
            total_structures += len(minima_list)
            for energy, atoms in minima_list:
                assert np.isfinite(energy)
                assert isinstance(atoms, Atoms)
                assert atoms.get_chemical_formula() == formula
                assert len(atoms) >= min_atoms
                assert len(atoms) <= max_atoms

    # At least one formula should have found some minima
    formulas_with_minima = sum(1 for m in results.values() if m)
    assert formulas_with_minima > 0, "No minima found across all compositions"


@pytest.mark.slow
@pytest.mark.integration
def test_campaign_two_elements(tmp_path):
    """Test bimetallic campaign workflow."""
    params = get_testing_params()

    # Run campaign for Au-Pt bimetallic clusters
    results = _run_go_campaign_compositions(
        build_two_element_compositions("Au", "Pt", 2, 2),
        system_type="gas_cluster",
        params=params,
        seed=789,
        output_dir=str(tmp_path / "campaign"),
    )

    # Verify all expected compositions are present
    expected_formulas = ["Au2", "AuPt", "Pt2"]
    for formula in expected_formulas:
        assert formula in results
        assert isinstance(results[formula], list)

        if results[formula]:
            for energy, atoms in results[formula]:
                assert np.isfinite(energy)
                assert isinstance(atoms, Atoms)
                assert atoms.get_chemical_formula() == formula


@pytest.mark.integration
@pytest.mark.slow
def test__run_go_trials_integration(tmp_path):
    """Test the high-level _run_go_trials function."""
    comp = ["Pt", "Pt"]
    params = get_testing_params()

    # Use clean=True and output_dir for isolation
    results = _run_go_trials(
        comp,
        "gas_cluster",
        params=params,
        seed=999,
        clean=True,
        output_dir=str(tmp_path / "pt2_searches"),
    )

    # Verify results structure
    assert isinstance(results, list)
    if results:
        for energy, atoms in results:
            assert np.isfinite(energy)
            assert isinstance(atoms, Atoms)
            assert len(atoms) == 2
            assert atoms.get_chemical_symbols() == ["Pt", "Pt"]


@pytest.mark.slow
def test__run_go_trials_deterministic_with_same_seed(tmp_path):
    """_run_go_trials should be deterministic for a fixed seed."""
    comp = ["Pt", "Pt"]
    params = get_testing_params()
    out = str(tmp_path / "pt2_det")

    # Use clean=True and output_dir for isolation; same dir for both runs
    results1 = _run_go_trials(
        comp,
        "gas_cluster",
        params=deepcopy(params),
        seed=1234,
        clean=True,
        output_dir=out,
    )
    results2 = _run_go_trials(
        comp,
        "gas_cluster",
        params=deepcopy(params),
        seed=1234,
        clean=True,
        output_dir=out,
    )

    assert len(results1) == len(results2)
    for (e1, a1), (e2, a2) in zip(results1, results2, strict=False):
        assert np.isclose(e1, e2)
        assert np.allclose(a1.get_positions(), a2.get_positions())


@pytest.mark.slow
@pytest.mark.integration
def test_output_directory_creation(tmp_path, rng):
    """Test that output directories are created correctly."""
    comp = ["Pt"]
    output_dir = str(tmp_path / "test_output")

    # Run minimal trial
    results = run_trials(
        composition=comp,
        global_optimizer="bh",
        global_optimizer_kwargs={
            "niter": 1,
            "niter_local_relaxation": 1,
            "system_type": "gas_cluster",
        },
        output_dir=output_dir,
        calculator_for_global_optimization=EMT(),
        validate_with_hessian=False,
        rng=rng,
    )

    # Verify directory structure (new structure: run_*/)
    assert os.path.exists(output_dir)
    from scgo.utils.run_tracking import get_run_directories

    run_dirs = get_run_directories(output_dir)
    assert len(run_dirs) > 0
    run_dir = run_dirs[0]
    assert os.path.exists(run_dir)
    assert os.path.exists(os.path.join(output_dir, "final_unique_minima"))

    # Verify file naming convention
    if results:
        xyz_dir = os.path.join(output_dir, "final_unique_minima")
        xyz_files = list(Path(xyz_dir).glob("*.xyz"))
        assert len(xyz_files) > 0

        # Check naming convention: Pt1_minimum_01_run_{run_id}.xyz
        xyz_file = xyz_files[0]
        assert "minimum_" in xyz_file.name
        assert "run_" in xyz_file.name
        assert xyz_file.name.endswith(".xyz")


@pytest.mark.slow
@pytest.mark.integration
def test_write_timing_json_at_run_level(tmp_path, rng):
    """Timing JSON is written alongside metadata.json at run root."""
    output_dir = str(tmp_path / "timing_output")
    run_trials(
        composition=["Pt", "Pt"],
        global_optimizer="bh",
        global_optimizer_kwargs={
            "niter": 1,
            "niter_local_relaxation": 1,
            "write_timing_json": True,
            "system_type": "gas_cluster",
        },
        output_dir=output_dir,
        calculator_for_global_optimization=EMT(),
        validate_with_hessian=False,
        rng=rng,
    )
    from scgo.utils.run_tracking import get_run_directories

    run_dir = get_run_directories(output_dir)[0]
    run_timing = os.path.join(run_dir, "timing.json")
    assert os.path.isfile(run_timing)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.requires_mace
def test_bh_high_energy_strategy(tmp_path, rng):
    """Test Basin Hopping with high_energy fitness strategy."""
    from scgo.param_presets import get_high_energy_params

    composition = ["Pt", "Pt", "Pt"]
    params = get_high_energy_params()
    params["optimizer_params"]["bh"]["niter"] = 5
    params["optimizer_params"]["bh"]["niter_local_relaxation"] = 2

    results = _run_go_trials(
        composition,
        "gas_cluster",
        params=params,
        seed=42,
        verbosity=1,
        output_dir=str(tmp_path / "bh_high_energy"),
    )

    # Should find some structures
    assert isinstance(results, list)

    # Check fitness values are stored (only in newly computed structures)
    # Note: Fitness is only stored during the current run, not loaded from old databases
    if results:
        # At least some results should have fitness information
        has_fitness = sum(1 for _, atoms in results if "fitness" in atoms.info)
        # We expect at least one structure to have fitness info (from current run)
        # But old structures from database won't have it
        if has_fitness > 0:
            for energy, atoms in results:
                if "fitness" in atoms.info:
                    assert "fitness_strategy" in atoms.info
                    assert atoms.info["fitness_strategy"] == "high_energy"
                    # For high_energy, fitness should equal energy
                    assert atoms.info["fitness"] == pytest.approx(energy)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.requires_mace
def test_ga_diversity_strategy(tmp_path, rng):
    """Test Genetic Algorithm with diversity fitness strategy."""
    from scgo.param_presets import get_diversity_params

    ref_dir = tmp_path / "ref"
    div_dir = tmp_path / "div"

    # First, create some reference structures under tmp_path
    comp_ref = ["Pt", "Pt", "Pt"]
    params_ref = get_testing_params()
    params_ref["optimizer_params"]["ga"]["niter"] = 2
    params_ref["optimizer_params"]["ga"]["population_size"] = 5
    params_ref["optimizer_params"]["ga"][
        "n_jobs_population_init"
    ] = -2  # Parallel for tests

    ref_results = _run_go_trials(
        comp_ref,
        "gas_cluster",
        params=params_ref,
        seed=42,
        clean=True,
        output_dir=str(ref_dir),
    )
    assert len(ref_results) > 0

    # Run with diversity strategy using reference DBs from ref_dir
    composition = ["Pt", "Pt", "Pt"]
    ref_glob = str(ref_dir / "**" / "*.db")
    params_div = get_diversity_params(reference_db_glob=ref_glob)
    params_div["optimizer_params"]["ga"]["niter"] = 2
    params_div["optimizer_params"]["ga"]["population_size"] = 5
    params_div["optimizer_params"]["ga"][
        "n_jobs_population_init"
    ] = -2  # Parallel for tests

    results = _run_go_trials(
        composition,
        "gas_cluster",
        params=params_div,
        seed=43,
        verbosity=1,
        clean=False,
        output_dir=str(div_dir),
    )

    # Should find diverse structures
    assert isinstance(results, list)

    # Check fitness values (only in newly computed structures)
    # Note: Fitness is only stored during the current run, not loaded from old databases
    if results:
        # At least some results should have fitness information
        has_fitness = sum(1 for _, atoms in results if "fitness" in atoms.info)
        # We expect at least one structure to have fitness info (from current run)
        if has_fitness > 0:
            for _energy, atoms in results:
                if "fitness" in atoms.info:
                    assert atoms.info["fitness_strategy"] == "diversity"
                    # Diversity fitness should be non-negative
                    assert atoms.info["fitness"] >= 0.0


@pytest.mark.slow
@pytest.mark.integration
def test_mixed_fitness_strategies(tmp_path, rng):
    """Test using different fitness strategies for BH and GA."""
    composition = ["Pt", "Pt", "Pt"]
    ref_dir = tmp_path / "ref"
    main_dir = tmp_path / "main"

    # First, create some reference structures under tmp_path
    params_ref = get_testing_params()
    params_ref["optimizer_params"]["bh"]["niter"] = 2
    params_ref["optimizer_params"]["bh"]["niter_local_relaxation"] = 2

    ref_results = _run_go_trials(
        composition,
        "gas_cluster",
        params=params_ref,
        seed=41,
        clean=True,
        output_dir=str(ref_dir),
    )
    assert len(ref_results) > 0

    # Run with mixed strategies using the reference databases we just created
    params = get_testing_params()

    # BH uses diversity, GA uses low_energy
    params["optimizer_params"]["bh"]["fitness_strategy"] = "diversity"
    params["optimizer_params"]["bh"]["diversity_reference_db"] = str(
        ref_dir / "**" / "*.db"
    )
    params["optimizer_params"]["bh"]["niter"] = 3

    params["optimizer_params"]["ga"]["fitness_strategy"] = "low_energy"
    params["optimizer_params"]["ga"]["niter"] = 2

    # Should work without errors
    results = _run_go_trials(
        composition,
        "gas_cluster",
        params=params,
        seed=42,
        verbosity=1,
        clean=False,
        output_dir=str(main_dir),
    )
    assert isinstance(results, list)


@pytest.mark.slow
@pytest.mark.integration
def test_campaign_database_handle_management(tmp_path, rng):
    """Test that database handles are properly closed in multi-composition campaigns.

    This test verifies that:
    1. Multiple compositions can be processed without file descriptor leaks
    2. Database connections are properly closed after each composition
    3. No lingering locks prevent subsequent operations
    """
    import psutil

    # Get initial file descriptor count
    process = psutil.Process()
    initial_fds = process.num_fds()

    # Run a campaign with multiple compositions (Pt2, Pt3, Pt4)
    params = get_testing_params()
    params["optimizer_params"]["ga"]["niter"] = 1
    params["optimizer_params"]["ga"]["population_size"] = 2
    params["optimizer_params"]["bh"]["niter"] = 1

    results = _run_go_campaign_compositions(
        build_one_element_compositions("Pt", 2, 4),
        system_type="gas_cluster",
        params=params,
        seed=42,
        verbosity=0,
        clean=True,
        output_dir=str(tmp_path / "campaign_fd_test"),
    )

    # Force garbage collection
    import gc

    gc.collect()

    # Check file descriptors haven't grown excessively
    # Allow some growth for normal operations but not hundreds of open files
    final_fds = process.num_fds()
    fd_growth = final_fds - initial_fds

    # Torch/MACE model internals (shared libraries, CUDA driver handles, model
    # file caches) legitimately retain FDs across a campaign.  CI environments
    # are particularly heavy because the entire test suite has already loaded
    # many libraries by this point.  Use a relaxed threshold in CI and a tighter
    # one locally so we still catch catastrophic leaks (hundreds of handles).
    fd_limit = 200 if os.environ.get("CI") else 50
    assert fd_growth < fd_limit, (
        f"File descriptor leak detected: {initial_fds} -> {final_fds} ({fd_growth} leaked)"
    )

    # Verify results structure
    assert isinstance(results, dict)
    assert "Pt2" in results
    assert "Pt3" in results
    assert "Pt4" in results

    # Verify we can write files without "too many open files" error
    # This would fail if database handles were still open
    test_file = tmp_path / "test_write.txt"
    for i in range(100):
        with open(test_file, "w") as f:
            f.write(f"test {i}\n")

    assert test_file.exists()
