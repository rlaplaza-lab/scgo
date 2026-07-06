"""Shared test helpers; re-exports selected names from tests.constants."""

import os
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT

import tests.constants as _tc
from scgo.constants import DEFAULT_ENERGY_TOLERANCE, DEFAULT_PAIR_COR_CUM_DIFF

# Re-exported for `from tests.test_utils import ...` (single source: _tc).
REPRODUCIBILITY_RTOL = _tc.REPRODUCIBILITY_RTOL
REPRODUCIBILITY_ATOL = _tc.REPRODUCIBILITY_ATOL
SMOKE_RTOL = _tc.SMOKE_RTOL
SMOKE_ATOL = _tc.SMOKE_ATOL
REPRODUCIBILITY_SEEDS = _tc.REPRODUCIBILITY_SEEDS
SMALL_SIZES = _tc.SMALL_SIZES
MEDIUM_SIZES = _tc.MEDIUM_SIZES
LARGE_SIZES = _tc.LARGE_SIZES
MIXED_COMPOSITIONS = _tc.MIXED_COMPOSITIONS
BATCH_TEST_SAMPLES = _tc.BATCH_TEST_SAMPLES
BATCH_TEST_SAMPLES_SLOW = _tc.BATCH_TEST_SAMPLES_SLOW
UNIQUENESS_THRESHOLD = _tc.UNIQUENESS_THRESHOLD
RNG_SEED_RANGE = _tc.RNG_SEED_RANGE
DIVERSITY_TEST_SAMPLES_SMALL = _tc.DIVERSITY_TEST_SAMPLES_SMALL
DIVERSITY_TEST_SAMPLES_MEDIUM = _tc.DIVERSITY_TEST_SAMPLES_MEDIUM
DIVERSITY_TEST_SAMPLES_LARGE = _tc.DIVERSITY_TEST_SAMPLES_LARGE
DIVERSITY_THRESHOLD_MIN = _tc.DIVERSITY_THRESHOLD_MIN
DIVERSITY_THRESHOLD_DEFAULT = _tc.DIVERSITY_THRESHOLD_DEFAULT
EMT_PT2_BOND_ANG = _tc.EMT_PT2_BOND_ANG
EMT_PT2_BOND_TOL_ANG = _tc.EMT_PT2_BOND_TOL_ANG
EMT_H2_BARRIER_EV = _tc.EMT_H2_BARRIER_EV
NN_DISTANCE_BAND = _tc.NN_DISTANCE_BAND
TS_FMAX_CONVERGED = _tc.TS_FMAX_CONVERGED
ADSORPTION_HEIGHT_TOLERANCE_ANG = _tc.ADSORPTION_HEIGHT_TOLERANCE_ANG
PT_O_DISTANCE_ANG = _tc.PT_O_DISTANCE_ANG


class MockRelaxer:
    """Minimal batch relaxer for GA tests (returns indexed energies)."""

    def __init__(self, max_steps: int | None = None):
        self.max_steps = max_steps

    def relax_batch(self, batch: list[Atoms]):
        return [(float(i) * 0.1, a.copy()) for i, a in enumerate(batch)]


def positions_equal(a: Atoms, b: Atoms, tolerance: float = 1e-6) -> bool:
    """Check if two Atoms objects have the same positions within tolerance.

    Args:
        a: First Atoms object to compare
        b: Second Atoms object to compare
        tolerance: Maximum allowed difference in positions (default: 1e-6)

    Returns:
        True if positions are equal within tolerance, False otherwise

    Note:
        This function only compares positions, not other properties like
        chemical symbols, cell, or other attributes.
    """
    if len(a) != len(b):
        return False
    return np.allclose(a.get_positions(), b.get_positions(), atol=tolerance)


def create_paired_rngs(seed: int):
    """Create two RNGs with the same seed for reproducibility testing.

    Args:
        seed: Integer seed for both RNGs

    Returns:
        Tuple of (rng1, rng2) both initialized with the same seed
    """
    return np.random.default_rng(seed), np.random.default_rng(seed)


def assert_serial_parallel_offspring_equal(
    tmp_path: Any,
    *,
    seed: int,
    ga_kwargs: dict[str, Any],
    rtol: float = REPRODUCIBILITY_RTOL,
    atol: float = REPRODUCIBILITY_ATOL,
    n_jobs_parallel: int = 2,
    output_suffix_serial: str = "offspring_serial",
    output_suffix_parallel: str = "offspring_parallel",
) -> None:
    """Assert GA offspring are identical for serial vs parallel workers (fixed seed)."""
    from scgo.algorithms import ga_go
    from scgo.utils.rng_helpers import create_child_rng, ensure_rng

    def _child_rng():
        return create_child_rng(ensure_rng(seed))

    with isolated_workflow_cwd(tmp_path / output_suffix_serial):
        minima_serial = ga_go(
            output_dir=str(tmp_path / output_suffix_serial),
            rng=_child_rng(),
            n_jobs_offspring=1,
            **ga_kwargs,
        )

    with isolated_workflow_cwd(tmp_path / output_suffix_parallel):
        minima_parallel = ga_go(
            output_dir=str(tmp_path / output_suffix_parallel),
            rng=_child_rng(),
            n_jobs_offspring=n_jobs_parallel,
            **ga_kwargs,
        )

    assert compare_minima_lists(minima_serial, minima_parallel, rtol=rtol, atol=atol), (
        "Serial and parallel offspring must match for fixed seed"
    )


def assert_batch_init_reproducible(
    *,
    composition: list[str],
    n_structures: int,
    seed: int,
    n_jobs_multi: int,
    mode: str = "smart",
) -> None:
    """Assert batch initialization is identical for n_jobs=1 vs parallel workers."""
    from scgo.initialization import create_initial_cluster_batch

    rng_single, rng_multi = create_paired_rngs(seed)

    batch_single = create_initial_cluster_batch(
        composition=composition,
        n_structures=n_structures,
        rng=rng_single,
        mode=mode,
        n_jobs=1,
    )
    batch_multi = create_initial_cluster_batch(
        composition=composition,
        n_structures=n_structures,
        rng=rng_multi,
        mode=mode,
        n_jobs=n_jobs_multi,
    )

    assert len(batch_single) == len(batch_multi) == n_structures
    for a_single, a_multi in zip(batch_single, batch_multi, strict=True):
        np.testing.assert_allclose(
            a_single.get_positions(),
            a_multi.get_positions(),
            rtol=REPRODUCIBILITY_RTOL,
            atol=REPRODUCIBILITY_ATOL,
        )
        assert a_single.get_chemical_symbols() == a_multi.get_chemical_symbols()


@contextmanager
def isolated_workflow_cwd(run_dir: Path) -> Iterator[Path]:
    """Run a workflow from an empty directory so seed discovery ignores repo dbs."""
    run_dir.mkdir(parents=True, exist_ok=True)
    previous = os.getcwd()
    os.chdir(run_dir)
    try:
        yield run_dir
    finally:
        os.chdir(previous)


@contextmanager
def seeded_globals(seed: int) -> Iterator[None]:
    """Seed Python and NumPy global RNGs; restore prior state on exit.

    Use only when ASE optimizers or legacy code paths consume global RNG state.
    Prefer explicit ``rng=`` arguments everywhere else.
    """
    import random

    py_state = random.getstate()
    np_state = np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


_MINIMUM_XYZ_RANK_RE = re.compile(r"_minimum_(\d+)_")


def compare_minima_lists(
    minima1,
    minima2,
    rtol=REPRODUCIBILITY_RTOL,
    atol=REPRODUCIBILITY_ATOL,
):
    """Compare two lists of minima (energy, atoms) tuples.

    Args:
        minima1: First list of (energy, atoms) tuples
        minima2: Second list of (energy, atoms) tuples
        rtol: Relative tolerance for energy comparison
        atol: Absolute tolerance for energy comparison

    Returns:
        True if lists are equal (order-independent), False otherwise
    """
    if len(minima1) != len(minima2):
        return False

    # Sort by energy to make comparison order-independent
    minima1_sorted = sorted(minima1, key=lambda x: x[0])
    minima2_sorted = sorted(minima2, key=lambda x: x[0])

    for (e1, a1), (e2, a2) in zip(minima1_sorted, minima2_sorted, strict=True):
        if not np.isclose(e1, e2, rtol=rtol, atol=atol):
            return False
        if not np.allclose(
            a1.get_positions(),
            a2.get_positions(),
            rtol=rtol,
            atol=atol,
        ):
            return False
        if not np.all(a1.get_atomic_numbers() == a2.get_atomic_numbers()):
            return False
    return True


def assert_minima_lists_equal(
    minima1,
    minima2,
    *,
    rtol: float = REPRODUCIBILITY_RTOL,
    atol: float = REPRODUCIBILITY_ATOL,
) -> None:
    """Assert two minima lists match within floating-point tolerance."""
    assert len(minima1) == len(minima2), (
        f"Minima count mismatch: {len(minima1)} != {len(minima2)}"
    )

    for index, ((e1, a1), (e2, a2)) in enumerate(
        zip(minima1, minima2, strict=True),
        start=1,
    ):
        np.testing.assert_allclose(
            e1,
            e2,
            rtol=rtol,
            atol=atol,
            err_msg=f"Energy mismatch at minimum {index}",
        )
        np.testing.assert_allclose(
            a1.get_positions(),
            a2.get_positions(),
            rtol=rtol,
            atol=atol,
            err_msg=f"Position mismatch at minimum {index}",
        )
        assert np.all(a1.get_atomic_numbers() == a2.get_atomic_numbers()), (
            f"Composition mismatch at minimum {index}"
        )


def xyz_files_by_minimum_rank(directory: Path) -> dict[int, Path]:
    """Map exported minimum rank (1-based) to XYZ path under ``directory``."""
    ranked: dict[int, Path] = {}
    for path in directory.glob("*.xyz"):
        match = _MINIMUM_XYZ_RANK_RE.search(path.name)
        if match is not None:
            ranked[int(match.group(1))] = path
    return dict(sorted(ranked.items()))


def assert_exported_minima_xyz_equal(
    directory_a: Path,
    directory_b: Path,
    *,
    rtol: float = REPRODUCIBILITY_RTOL,
    atol: float = REPRODUCIBILITY_ATOL,
) -> None:
    """Assert exported minima XYZ files match by minimum rank."""
    from ase.io import read

    ranked_a = xyz_files_by_minimum_rank(directory_a)
    ranked_b = xyz_files_by_minimum_rank(directory_b)
    assert ranked_a.keys() == ranked_b.keys(), (
        f"Exported minimum ranks differ: {list(ranked_a)} vs {list(ranked_b)}"
    )

    for rank in ranked_a:
        atoms_a = read(str(ranked_a[rank]))
        atoms_b = read(str(ranked_b[rank]))
        np.testing.assert_allclose(
            atoms_a.get_positions(),
            atoms_b.get_positions(),
            rtol=rtol,
            atol=atol,
            err_msg=f"XYZ position mismatch at minimum rank {rank:02d}",
        )
        assert list(atoms_a.get_atomic_numbers()) == list(
            atoms_b.get_atomic_numbers()
        ), f"XYZ composition mismatch at minimum rank {rank:02d}"

        score_a = atoms_a.info.get("key_value_pairs", {}).get("raw_score")
        score_b = atoms_b.info.get("key_value_pairs", {}).get("raw_score")
        if score_a is not None and score_b is not None:
            np.testing.assert_allclose(
                score_a,
                score_b,
                rtol=rtol,
                atol=atol,
                err_msg=f"XYZ raw_score mismatch at minimum rank {rank:02d}",
            )


def setup_test_atoms(atoms: Atoms, cell_size: float = 10.0, pbc: bool = False) -> Atoms:
    """Set up standard cell and pbc for test Atoms objects.

    Args:
        atoms: Atoms object to configure
        cell_size: Size of the cell (default: 10.0)
        pbc: Periodic boundary conditions (default: False)

    Returns:
        The same Atoms object with cell and pbc configured
    """
    atoms.set_cell([cell_size, cell_size, cell_size])
    atoms.set_pbc(pbc)
    return atoms


def create_test_atoms(
    composition: list[str] | str,
    *,
    positions: list[list[float]] | None = None,
    calc: Any = None,
    raw_score: float | None = None,
    trial: int | None = None,
    cell_size: float = 10.0,
    pbc: bool = False,
) -> Atoms:
    """Create test atoms with optional calculator and metadata.

    This is the unified factory for all test atom creation, consolidating
    create_atoms_with_calc() and create_atoms_with_info() functionality.

    Args:
        composition: Chemical composition (list or formula string)
        positions: Optional atom positions (uses defaults if None)
        calc: Optional calculator (uses EMT() if None and metadata provided)
        raw_score: Optional energy for metadata
        trial: Optional trial number for metadata
        cell_size: Cell size (default: 10.0)
        pbc: Periodic boundary conditions (default: False)

    Returns:
        Atoms object configured for testing
    """
    # Create atoms with positions if provided
    if positions is None:
        atoms = Atoms(composition)
    else:
        atoms = Atoms(composition, positions=positions)

    # Attach calculator if provided
    if calc is not None:
        atoms.calc = calc
    elif raw_score is not None or trial is not None:
        # If metadata provided but no calc, use EMT for consistency
        atoms.calc = EMT()

    # Set up test environment
    setup_test_atoms(atoms, cell_size=cell_size, pbc=pbc)

    # Add metadata if provided
    if raw_score is not None or trial is not None:
        if not hasattr(atoms, "info") or atoms.info is None:
            atoms.info = {}

        if raw_score is not None:
            from scgo.database.metadata import add_metadata

            add_metadata(atoms, raw_score=raw_score)

        if trial is not None:
            if "provenance" not in atoms.info:
                atoms.info["provenance"] = {}
            atoms.info["provenance"]["trial_id"] = trial

    return atoms


def run_algorithm_reproducibility_test(
    algorithm_func: Callable,
    composition: list[str],
    seed: int,
    tmp_path: Any,
    algorithm_params: dict[str, Any],
    output_suffix_1: str = "run1",
    output_suffix_2: str = "run2",
) -> tuple[list, list]:
    """Run algorithm reproducibility test by executing twice with the same NumPy seed.

    SCGO algorithms consume only the explicit ``rng`` argument; callers should
    not rely on Python's global ``random`` module for reproducibility.

    Args:
        algorithm_func: The algorithm function to test (bh_go or ga_go)
        composition: Chemical composition list
        seed: Random seed for reproducibility
        tmp_path: Pytest tmp_path fixture for output directories
        algorithm_params: Parameters to pass to the algorithm function
        output_suffix_1: Suffix for first run output directory
        output_suffix_2: Suffix for second run output directory

    Returns:
        Tuple of (minima1, minima2) from the two runs

    Example:
        >>> minima1, minima2 = run_algorithm_reproducibility_test(
        ...     bh_go, ["Pt", "Pt", "Pt"], 123, tmp_path,
        ...     {"niter": 3, "dr": 0.2, "temperature": 0.01}
        ... )
    """
    from scgo.algorithms import ga_go
    from scgo.initialization import create_initial_cluster

    is_ga_function = algorithm_func is ga_go

    with seeded_globals(seed):
        rng1, _ = create_paired_rngs(seed)
        if is_ga_function:
            with isolated_workflow_cwd(tmp_path / output_suffix_1):
                minima1 = algorithm_func(
                    composition,
                    calculator=EMT(),
                    output_dir=str(tmp_path / output_suffix_1),
                    rng=rng1,
                    **algorithm_params,
                )
        else:
            atoms1 = create_initial_cluster(composition, rng=rng1)
            atoms1.calc = EMT()
            minima1 = algorithm_func(
                atoms1,
                output_dir=str(tmp_path / output_suffix_1),
                rng=rng1,
                **algorithm_params,
            )

    with seeded_globals(seed):
        _, rng2 = create_paired_rngs(seed)
        if is_ga_function:
            with isolated_workflow_cwd(tmp_path / output_suffix_2):
                minima2 = algorithm_func(
                    composition,
                    calculator=EMT(),
                    output_dir=str(tmp_path / output_suffix_2),
                    rng=rng2,
                    **algorithm_params,
                )
        else:
            atoms2 = create_initial_cluster(composition, rng=rng2)
            atoms2.calc = EMT()
            minima2 = algorithm_func(
                atoms2,
                output_dir=str(tmp_path / output_suffix_2),
                rng=rng2,
                **algorithm_params,
            )

    return minima1, minima2


def create_preparedb(
    atoms: Atoms,
    db_path: Path | str,
    *,
    population_size: int | None = None,
    **prepare_kwargs,
):
    """Create a PrepareDB instance for testing.

    This helper reduces duplication in database tests by extracting the common
    pattern of creating a PrepareDB with the correct stoichiometry.

    Args:
        atoms: Atoms object to use as simulation cell and for stoichiometry
        db_path: Path to the database file (Path or string)
        population_size: Optional population size forwarded to PrepareDB
        **prepare_kwargs: Any additional keyword args forwarded to PrepareDB

    Returns:
        PrepareDB instance configured for the given atoms

    Example:
        >>> db = create_preparedb(pt3_atoms, tmp_path / "test.db", population_size=10)
        >>> db.add_unrelaxed_candidate(test_atoms, description="test")
    """
    from ase_ga.data import PrepareDB

    all_atom_numbers = [int(num) for num in atoms.get_atomic_numbers()]
    prepare_args = {
        "db_file_name": str(db_path),
        "simulation_cell": atoms,
        "stoichiometry": all_atom_numbers,
    }
    if population_size is not None:
        prepare_args["population_size"] = population_size
    prepare_args.update(prepare_kwargs)

    db = PrepareDB(**prepare_args)

    from scgo.database.registry import get_registry

    get_registry(Path(db_path).parent).register_database(
        Path(db_path), composition=list(atoms.symbols)
    )

    return db


def mark_test_minima_as_final(db_path: Path | str) -> None:
    """Mark all relaxed minima in a test DB as final_unique_minimum.

    TS runs require final-tagged minima from GO. Call this after add_relaxed_step
    in test fixtures that need run_transition_state_search to find minima.

    Uses ASE db.update() to keep index tables (number_key_values, etc.) in sync.
    Also adds scgo_metadata so DatabaseDiscovery.find_databases() includes this DB.

    Args:
        db_path: Path to the database file.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        return

    # Use ASE db.update() so number_key_values and other index tables stay in sync
    import ase.db

    with ase.db.connect(str(db_path)) as conn:
        for row in conn.select(relaxed=1):
            conn.update(row.id, final_unique_minimum=1, final_rank=1)

    from scgo.database.schema import stamp_scgo_database

    stamp_scgo_database(db_path)


def create_ga_comparator(n_top: int):
    """Create a SequentialComparator for GA testing.

    This helper reduces duplication in GA patch tests by extracting the common
    comparator configuration pattern.

    Args:
        n_top: Number of atoms in the cluster (for InteratomicDistanceComparator)

    Returns:
        SequentialComparator instance configured for GA testing

    Example:
        >>> comp = create_ga_comparator(len(pt3_atoms))
        >>> population = Population(..., comparator=comp, ...)
    """
    from ase_ga.standard_comparators import (
        InteratomicDistanceComparator,
        RawScoreComparator,
        SequentialComparator,
    )

    return SequentialComparator(
        methods=[
            RawScoreComparator(dist=DEFAULT_ENERGY_TOLERANCE),
            InteratomicDistanceComparator(
                n_top=n_top,
                mic=False,
                dE=DEFAULT_ENERGY_TOLERANCE,
                pair_cor_cum_diff=DEFAULT_PAIR_COR_CUM_DIFF,
            ),
        ],
    )


def get_structure_signature(atoms: Atoms, *, precision: int = 6) -> tuple[float, ...]:
    """Geometry signature for tests (delegates to production helper)."""
    from scgo.initialization.candidate_discovery import get_structure_signature as _sig

    return _sig(atoms, precision=precision)


def assert_cluster_valid(
    atoms: Atoms,
    expected_composition: list[str],
    min_distance_factor: float = _tc.MIN_DISTANCE_FACTOR_DEFAULT,
    connectivity_factor: float | None = None,
    check_connectivity: bool | None = None,
) -> None:
    """Assert that a cluster satisfies all standard invariants.

    This helper reduces duplication in tests by consolidating common validation
    patterns: composition match, connectivity, and no clashes.

    Args:
        atoms: Atoms object to validate
        expected_composition: Expected chemical composition list
        min_distance_factor: Minimum distance factor for clash checking
            (default: MIN_DISTANCE_FACTOR_DEFAULT, currently 0.4)
        connectivity_factor: Connectivity factor to use (default: None uses CONNECTIVITY_FACTOR)
        check_connectivity: Whether to check connectivity (default: None auto-detects from atom count)

    Raises:
        AssertionError: If any invariant is violated

    Example:
        >>> atoms = create_initial_cluster(["Pt", "Pt"], rng=rng)
        >>> assert_cluster_valid(atoms, ["Pt", "Pt"])
    """
    from scgo.initialization import is_cluster_connected
    from scgo.initialization.geometry_helpers import validate_cluster_structure
    from scgo.initialization.initialization_config import CONNECTIVITY_FACTOR
    from scgo.utils.helpers import get_composition_counts

    # Verify exact composition match
    assert get_composition_counts(
        atoms.get_chemical_symbols()
    ) == get_composition_counts(expected_composition), (
        f"Composition mismatch: expected {get_composition_counts(expected_composition)}, "
        f"got {get_composition_counts(atoms.get_chemical_symbols())}"
    )

    # Determine connectivity checking
    if connectivity_factor is None:
        connectivity_factor = CONNECTIVITY_FACTOR
    if check_connectivity is None:
        check_connectivity = len(atoms) > 1

    # Verify connectivity (for 2+ atoms)
    if check_connectivity:
        assert (
            is_cluster_connected(atoms, connectivity_factor=connectivity_factor) is True
        ), f"Cluster must be connected with connectivity_factor={connectivity_factor}"

    # Verify no clashes and connectivity via validate_cluster_structure
    is_valid, msg = validate_cluster_structure(
        atoms,
        min_distance_factor=min_distance_factor,
        connectivity_factor=connectivity_factor,
        check_clashes=True,
        check_connectivity=check_connectivity,
    )
    assert is_valid is True, f"Cluster validation failed: {msg}"


def assert_ts_result_valid(
    result: dict[str, Any],
    *,
    barrier_range: tuple[float, float] = EMT_H2_BARRIER_EV,
    require_interior_ts: bool = True,
) -> None:
    """Assert a NEB/TS result satisfies basic transition-state physics invariants."""
    barrier = result.get("barrier_height")
    assert barrier is not None, f"Missing barrier_height in result: {result}"
    lo, hi = barrier_range
    assert lo <= barrier <= hi, f"barrier_height {barrier} not in [{lo}, {hi}]"

    ts_energy = result.get("ts_energy")
    reactant_energy = result.get("reactant_energy")
    product_energy = result.get("product_energy")
    assert ts_energy is not None, f"Missing ts_energy in result: {result}"
    if reactant_energy is not None and product_energy is not None:
        assert ts_energy >= min(reactant_energy, product_energy) - 1e-6, (
            f"TS energy {ts_energy} below endpoint minimum "
            f"({reactant_energy}, {product_energy})"
        )

    if require_interior_ts:
        assert result.get("status") == "success", (
            f"Expected successful interior TS, got status={result.get('status')!r} "
            f"error={result.get('error')!r}"
        )
        ts_idx = result.get("ts_image_index")
        n_images = result.get("n_images")
        assert ts_idx is not None and n_images is not None
        assert 0 < int(ts_idx) < int(n_images), (
            f"TS image index {ts_idx} not interior for n_images={n_images}"
        )

    if result.get("neb_converged"):
        final_fmax = result.get("final_fmax")
        assert final_fmax is not None
        assert float(final_fmax) < TS_FMAX_CONVERGED, (
            f"final_fmax {final_fmax} >= {TS_FMAX_CONVERGED} despite neb_converged"
        )


def assert_nn_distances_in_band(
    atoms: Atoms,
    *,
    band: tuple[float, float] = NN_DISTANCE_BAND,
) -> None:
    """Assert each atom's nearest-neighbor distance is within a covalent-radii band."""
    from scgo.initialization.atomic_radii import get_covalent_radius

    symbols = atoms.get_chemical_symbols()
    n_atoms = len(atoms)
    if n_atoms < 2:
        return

    for i in range(n_atoms):
        ri = get_covalent_radius(symbols[i])
        best_j = None
        best_d = float("inf")
        for j in range(n_atoms):
            if i == j:
                continue
            d = atoms.get_distance(i, j, mic=True)
            if d < best_d:
                best_d = d
                best_j = j
        assert best_j is not None
        rj = get_covalent_radius(symbols[best_j])
        expected = ri + rj
        lo = band[0] * expected
        hi = band[1] * expected
        assert lo <= best_d <= hi, (
            f"NN distance {best_d:.3f} Å for {symbols[i]}-{symbols[best_j]} "
            f"outside [{lo:.3f}, {hi:.3f}] (expected sum radii {expected:.3f})"
        )


def assert_deposition_height_in_bounds(
    atoms: Atoms,
    slab: Atoms,
    h_min: float,
    h_max: float,
    *,
    n_slab: int,
    axis: int = 2,
    tolerance: float = ADSORPTION_HEIGHT_TOLERANCE_ANG,
    mobile_slice: slice | None = None,
) -> None:
    """Assert **initial placement** heights lie in the deposition sampler window.

    ``SurfaceSystemConfig.adsorption_height_min/max`` constrain how
    :func:`~scgo.surface.deposition.create_deposited_cluster` samples the
    cluster bottom above the slab top. They are not post-relaxation bounds:
    GA and NEB may move atoms outside this window while remaining bound to
    the surface. Use :func:`assert_supported_cluster_binding` after relaxation.
    """
    from scgo.surface.deposition import slab_surface_extreme

    slab_top = slab_surface_extreme(slab, axis, upper=True)
    positions = atoms.get_positions()
    mobile_positions = (
        positions[mobile_slice] if mobile_slice is not None else positions[n_slab:]
    )
    assert len(mobile_positions) > 0, "No mobile atoms to check"
    for pos in mobile_positions:
        height = float(pos[axis] - slab_top)
        assert h_min - tolerance <= height <= h_max + tolerance, (
            f"Deposition height {height:.3f} Å not in [{h_min}, {h_max}] ± {tolerance}"
        )


def assert_supported_cluster_binding(
    atoms: Atoms,
    surface_config,
    *,
    n_core_mobile: int | None = None,
    adsorbate_fragment_lengths: list[int] | None = None,
    allow_cluster_fragmentation: bool = False,
    allow_adsorbate_surface_detachment: bool = False,
    enforce_adsorbate_subgraph_integrity: bool = True,
) -> None:
    """Assert a relaxed slab+cluster remains chemisorbed and geometrically valid.

    Checks surface contact, no penetration below the slab top, connectivity,
    and (when fragment lengths are given) adsorbate fragment integrity.
    """
    from scgo.surface.validation import validate_supported_cluster_deposit

    ok, msg = validate_supported_cluster_deposit(
        atoms,
        len(surface_config.slab),
        surface_normal_axis=surface_config.surface_normal_axis,
        use_mic=surface_config.comparator_use_mic,
        n_core_mobile=n_core_mobile,
        adsorbate_fragment_lengths=adsorbate_fragment_lengths,
        allow_cluster_fragmentation=allow_cluster_fragmentation,
        allow_adsorbate_surface_detachment=allow_adsorbate_surface_detachment,
        enforce_adsorbate_subgraph_integrity=enforce_adsorbate_subgraph_integrity,
    )
    assert ok, msg


# Backward-compatible alias; prefer assert_deposition_height_in_bounds.
assert_adsorption_height_in_bounds = assert_deposition_height_in_bounds


def assert_pt_o_distance_reasonable(
    atoms: Atoms,
    pt_idx: int,
    o_idx: int,
    *,
    lo: float = PT_O_DISTANCE_ANG[0],
    hi: float = PT_O_DISTANCE_ANG[1],
) -> None:
    """Assert Pt–O separation is within a chemically reasonable window."""
    d = atoms.get_distance(pt_idx, o_idx, mic=True)
    assert lo <= d <= hi, f"Pt–O distance {d:.3f} Å outside [{lo}, {hi}]"


def assert_run_id_persisted(atoms: Atoms, expected_run_id: str) -> None:
    """Assert `run_id` is present in `Atoms.info` (`metadata` or `key_value_pairs`).

    This helper centralizes the run_id check across the test suite so tests
    don't branch on storage location.
    """
    md_run = atoms.info.get("metadata", {}).get("run_id")
    kv_run = atoms.info.get("key_value_pairs", {}).get("run_id")
    assert md_run == expected_run_id or kv_run == expected_run_id, (
        f"run_id not persisted (expected={expected_run_id!r}); "
        f"metadata.run_id={md_run!r}, key_value_pairs.run_id={kv_run!r}"
    )


def assert_db_final_row(db_path, expected_run_id, expect_final_id=True):
    """Assert at least one `systems` row is tagged as a final unique minimum.

    SCGO-created ASE databases store these flags in ``key_value_pairs`` JSON.
    If `expected_run_id` is not None, asserts at least one final-tagged row has
    ``run_id == expected_run_id`` in that JSON. If `expect_final_id` is True,
    asserts a tagged row contains a non-empty ``final_id`` value.
    """
    import json
    import sqlite3

    dbp = str(db_path)
    with sqlite3.connect(dbp) as conn:
        cur = conn.cursor()
        cur.execute("SELECT key_value_pairs FROM systems")
        rows = cur.fetchall()
        assert rows, "No rows found in DB"

        found_final = False
        found_runid_exact = False
        for (kvp_json,) in rows:
            kvp = json.loads(kvp_json) if kvp_json else {}

            if not kvp.get("final_unique_minimum"):
                continue

            found_final = True

            run_in_row = kvp.get("run_id")
            if expected_run_id is not None and run_in_row == expected_run_id:
                found_runid_exact = True

            if expect_final_id:
                final_id = kvp.get("final_id")
                assert final_id, "final_id not persisted for tagged final minima"

    assert found_final is True, "No final_unique_minimum flag found in database rows"
    if expected_run_id is not None:
        assert found_runid_exact is True, (
            f"No row with final_unique_minimum and expected run_id={expected_run_id!r}"
        )


def validate_structure_with_diagnostics(
    atoms: Atoms,
    min_distance_factor: float | None = None,
    connectivity_factor: float | None = None,
    context: str = "",
) -> None:
    """Validate structure and provide detailed diagnostics on failure.

    This helper reduces duplication in tests by consolidating validation
    with diagnostic information.

    Args:
        atoms: Atoms object to validate
        min_distance_factor: Minimum distance factor for clash checking
            (default: None uses MIN_DISTANCE_FACTOR_DEFAULT)
        connectivity_factor: Connectivity factor to use
            (default: None uses CONNECTIVITY_FACTOR)
        context: Optional context string for error messages

    Raises:
        pytest.fail: If validation fails, with detailed diagnostics
    """
    from scgo.initialization import get_structure_diagnostics
    from scgo.initialization.geometry_helpers import (
        _should_check_connectivity,
        validate_cluster_structure,
    )
    from scgo.initialization.initialization_config import (
        CONNECTIVITY_FACTOR,
        MIN_DISTANCE_FACTOR_DEFAULT,
    )

    if len(atoms) == 0:
        return  # Empty structures are trivially valid

    if min_distance_factor is None:
        min_distance_factor = MIN_DISTANCE_FACTOR_DEFAULT
    if connectivity_factor is None:
        connectivity_factor = CONNECTIVITY_FACTOR

    is_valid, error_msg = validate_cluster_structure(
        atoms,
        min_distance_factor,
        connectivity_factor,
        check_clashes=True,
        check_connectivity=_should_check_connectivity(atoms),
    )

    if not is_valid:
        diagnostics = get_structure_diagnostics(
            atoms, min_distance_factor, connectivity_factor
        )
        import pytest

        pytest.fail(
            f"Structure validation failed{' (' + context + ')' if context else ''}: "
            f"{error_msg}\n"
            f"Diagnostics: {diagnostics.summary}"
        )


def run_batch_connectivity_test(
    composition: list[str],
    mode: str,
    n_atoms: int,
    rng: np.random.Generator,
    create_atoms_func: Callable[[list[str], str, np.random.Generator], Atoms],
    n_samples: int = BATCH_TEST_SAMPLES,
    uniqueness_threshold: float = UNIQUENESS_THRESHOLD,
    connectivity_factor: float | None = None,
    composition_label: str = "",
) -> None:
    """Run batch connectivity test with uniqueness check.

    This helper reduces duplication in batch connectivity tests by consolidating
    the common pattern of generating multiple samples, checking connectivity,
    and verifying uniqueness.

    Args:
        composition: Target chemical composition
        mode: Initialization mode name (for error messages)
        n_atoms: Expected number of atoms
        rng: Random number generator (used to generate seeds, not mutated)
        create_atoms_func: Function that creates atoms: (comp, mode, rng) -> Atoms
        n_samples: Number of samples to generate (default: BATCH_TEST_SAMPLES)
        uniqueness_threshold: Minimum uniqueness ratio (default: UNIQUENESS_THRESHOLD)
        connectivity_factor: Connectivity factor to use (default: None uses CONNECTIVITY_FACTOR)
        composition_label: Label for composition type in error messages (e.g., "bimetallic")

    Raises:
        pytest.fail: If connectivity or uniqueness checks fail
    """
    import pytest

    from scgo.initialization import is_cluster_connected
    from scgo.initialization.geometry_helpers import analyze_disconnection
    from scgo.initialization.initialization_config import CONNECTIVITY_FACTOR
    from scgo.utils.helpers import get_composition_counts

    if connectivity_factor is None:
        connectivity_factor = CONNECTIVITY_FACTOR

    # Pre-generate seeds to avoid mutating parent RNG state
    seed_rng = np.random.default_rng(42)  # Fixed seed for reproducible test seeds
    seeds = [seed_rng.integers(*RNG_SEED_RANGE) for _ in range(n_samples)]

    failures = []
    signatures = []

    for i, seed in enumerate(seeds):
        sample_rng = np.random.default_rng(seed)
        atoms = create_atoms_func(composition, mode, sample_rng)

        assert len(atoms) == n_atoms, f"Sample {i}: Size mismatch"
        assert get_composition_counts(
            atoms.get_chemical_symbols()
        ) == get_composition_counts(composition), f"Sample {i}: Composition mismatch"

        is_connected = is_cluster_connected(
            atoms, connectivity_factor=connectivity_factor
        )
        if not is_connected:
            (
                disconnection_distance,
                suggested_factor,
                analysis_msg,
            ) = analyze_disconnection(atoms, connectivity_factor)
            failures.append(
                {
                    "sample": i,
                    "disconnection_distance": disconnection_distance,
                    "suggested_factor": suggested_factor,
                    "analysis": analysis_msg,
                }
            )

        signature = get_structure_signature(atoms)
        signatures.append(signature)

    unique_signatures = set(signatures)
    uniqueness_ratio = len(unique_signatures) / n_samples

    if failures:
        failure_summary = "\n".join(
            [
                f"  Sample {f['sample']}: gap={f['disconnection_distance']:.3f} Å, "
                f"suggested_factor={f['suggested_factor']:.2f}, {f['analysis']}"
                for f in failures
            ]
        )
        comp_label_str = f" {composition_label}" if composition_label else ""
        pytest.fail(
            f"Found {len(failures)}/{n_samples} disconnected{comp_label_str} clusters "
            f"in {mode} mode batch test (n_atoms={n_atoms}). "
            f"Failures:\n{failure_summary}"
        )

    assert uniqueness_ratio >= uniqueness_threshold, (
        f"Insufficient uniqueness{(' for ' + composition_label) if composition_label else ''}: "
        f"only {len(unique_signatures)}/{n_samples} "
        f"({uniqueness_ratio:.1%}) unique structures in {mode} mode batch test"
    )
