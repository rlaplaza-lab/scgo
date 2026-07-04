"""TorchSim-enhanced Genetic Algorithm global optimization for clusters.

Genetic Algorithm global optimization with batched relaxations (TorchSim for MLIPs,
ASE sequential batch relaxer for classical calculators). Database interaction
remains single-threaded to protect against SQLite locking issues.
"""

from __future__ import annotations

import copy
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import suppress
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.optimize import FIRE
from ase.optimize.optimize import Optimizer
from ase_ga.data import DataConnection
from ase_ga.utilities import get_all_atom_types
from tqdm import tqdm

from scgo.algorithms.ga_common import (
    ClusterStartGenerator,
    SurfaceClusterStartGenerator,
    create_ga_pairing,
    create_mutation_operators,
    create_structure_comparator,
    ga_run_metadata_extras,
    log_early_stopping_info,
    maybe_apply_mobile_core_ads_tags,
    reseed_mutation_operator_rngs,
    select_population_class,
    setup_diversity_scorer,
    sort_minima_by_fitness,
    update_early_stopping_state_unified,
    update_mutation_weights,
    validate_ga_common_params,
)
from scgo.ase_ga_patches.population import Population
from scgo.calculators.ase_batch_relaxer import AseBatchRelaxer
from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer
from scgo.cluster_adsorbate.config import ClusterAdsorbateConfig
from scgo.cluster_adsorbate.constraints import (
    attach_adsorbate_internal_geometry_constraints,
)
from scgo.cluster_adsorbate.rigid import enforce_frozen_adsorbate_geometry
from scgo.constants import DEFAULT_ENERGY_TOLERANCE
from scgo.database import (
    HPC_DATABASE_EXCEPTIONS,
    RetryConfig,
    close_data_connection,
    database_retry,
    setup_database,
)
from scgo.database.constants import SYSTEMS_JSON_COLUMN
from scgo.database.metadata import (
    add_metadata,
    filter_by_metadata,
    get_metadata,
    update_metadata,
)
from scgo.initialization import compute_cell_side
from scgo.initialization.atomic_radii import build_blmin_from_zs
from scgo.initialization.geometry_helpers import reorder_cluster_to_composition
from scgo.initialization.initialization_config import BLMIN_RATIO_DEFAULT
from scgo.surface.config import SurfaceSystemConfig
from scgo.surface.constraints import attach_slab_constraints
from scgo.system_types import (
    AdsorbateDefinition,
    AdsorbateFragmentInput,
    SystemType,
    uses_surface,
    validate_structure_for_system_type,
    validate_system_type_settings,
)
from scgo.utils.fitness_strategies import (
    FitnessStrategy,
    ensure_fitness_strategy_resolved,
)
from scgo.utils.helpers import (
    canonicalize_relaxed_for_storage,
    extract_minima_from_database,
)
from scgo.utils.logging import get_logger, should_show_progress
from scgo.utils.mutation_weights import get_adaptive_mutation_config
from scgo.utils.parallel_workers import resolve_n_jobs_to_workers
from scgo.utils.rng_helpers import ensure_rng_or_create, offspring_rng_triple
from scgo.utils.timing_report import (
    cpu_non_relax_seconds_from_timings,
    ga_relax_seconds_from_timings,
    log_timing_summary,
    write_timing_file,
)
from scgo.utils.torchsim_policy import is_ml_calculator
from scgo.utils.validation import validate_composition


def _resolve_parallel_worker_count(n_jobs: int, n_tasks: int) -> int:
    """Resolve worker count from initialization-style semantics."""
    if n_tasks <= 1:
        return 1
    requested = resolve_n_jobs_to_workers(n_jobs)
    return max(1, min(requested, n_tasks))


def _sorted_unrelaxed_gaids(da: DataConnection) -> list[int]:
    """Return unrelaxed configuration IDs in deterministic ascending order."""
    all_unrelaxed = {row.gaid for row in da.c.select(relaxed=0)}
    all_relaxed = {row.gaid for row in da.c.select(relaxed=1)}
    all_queued = {row.gaid for row in da.c.select(queued=1)}
    return sorted(
        gaid
        for gaid in all_unrelaxed
        if gaid not in all_relaxed and gaid not in all_queued
    )


def _load_unrelaxed_by_gaid(da: DataConnection, gaid: int) -> Atoms:
    """Load the latest trajectory for an unrelaxed configuration ID."""
    rows = list(da.c.select(gaid=gaid))
    rows.sort(key=lambda row: row.mtime)
    atoms = da.get_atoms(rows[-1].id)
    atoms.info["confid"] = gaid
    atoms.info.setdefault("data", {})
    return atoms


def _count_relaxed_candidates(da: DataConnection) -> int:
    """Count relaxed candidates without materializing all candidate atoms."""
    kvp = SYSTEMS_JSON_COLUMN
    with da.c.managed_connection() as conn:
        row = conn.execute(
            f"SELECT COUNT(*) FROM systems "
            f"WHERE CAST(json_extract({kvp}, '$.relaxed') AS INTEGER)=1"
        ).fetchone()
    return int(row[0]) if row else 0


def _fails_fast_geometric_prefilter(atoms: Atoms, blmin: dict) -> bool:
    """Return True when a severe clash is detected quickly."""
    if len(atoms) < 2:
        return False
    numbers = atoms.get_atomic_numbers()
    positions = atoms.get_positions()
    for i in range(len(atoms)):
        zi = int(numbers[i])
        for j in range(i + 1, len(atoms)):
            zj = int(numbers[j])
            distance = float(np.linalg.norm(positions[i] - positions[j]))
            min_allowed = float(blmin.get((zi, zj), blmin.get((zj, zi), 0.0)))
            if min_allowed > 0.0 and distance < 0.55 * min_allowed:
                return True
    return False


def _picklable_atoms_copy(atoms: Atoms | None) -> Atoms | None:
    """Return an Atoms copy safe for process-pool pickling (no calculator)."""
    if atoms is None:
        return None
    copy = atoms.copy()
    copy.calc = None
    return copy


def _picklable_fragment_templates(
    templates: AdsorbateFragmentInput | None,
) -> list[Atoms] | None:
    if templates is None:
        return None
    if isinstance(templates, Atoms):
        copied = _picklable_atoms_copy(templates)
        return [copied] if copied is not None else None
    out: list[Atoms] = []
    for frag in templates:
        copied = _picklable_atoms_copy(frag)
        if copied is not None:
            out.append(copied)
    return out or None


@dataclass(frozen=True)
class OffspringBuildContext:
    """Picklable snapshot of per-generation offspring build inputs."""

    atoms_template: Atoms
    n_to_optimize: int
    composition: list[str]
    blmin: dict
    system_type: SystemType
    n_slab: int
    slab_for_pairing: Atoms | None
    surface_normal_axis: int
    adsorbate_definition: AdsorbateDefinition | None
    connectivity_factor: float | None
    allow_cluster_fragmentation: bool
    allow_adsorbate_surface_detachment: bool
    enforce_adsorbate_subgraph_integrity: bool
    freeze_adsorbate_internal_geometry: bool
    adsorbate_fragment_templates: list[Atoms] | None
    surface_config: SurfaceSystemConfig | None
    adaptive_config: dict[str, Any]
    current_mutation_probability: float
    operators_list: list
    name_map: dict[str, int]


def _offspring_worker_init() -> None:
    """Limit BLAS threading in process-pool offspring workers."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def _build_offspring_worker(
    job: dict[str, Any],
    ctx: OffspringBuildContext,
) -> dict[str, Any]:
    """Build one GA offspring (crossover + optional mutation) in an isolated worker."""
    worker_logger = get_logger(__name__)
    pairing_rng, operator_rng, decision_rng = offspring_rng_triple(job["task_seed"])
    setup_t0 = perf_counter()
    local_pairing = create_ga_pairing(
        ctx.atoms_template,
        ctx.n_to_optimize,
        pairing_rng,
        slab_atoms=ctx.slab_for_pairing,
        system_type=ctx.system_type,
        composition=ctx.composition,
        adsorbate_definition=ctx.adsorbate_definition,
    )
    local_ops = copy.deepcopy(ctx.operators_list)
    reseed_mutation_operator_rngs(local_ops, operator_rng)
    local_mutations = update_mutation_weights(
        operators_list=local_ops,
        name_map=ctx.name_map,
        adaptive_config=ctx.adaptive_config,
        rng=decision_rng,
    )
    operator_setup_s = perf_counter() - setup_t0
    crossover_t0 = perf_counter()
    child, desc = local_pairing.get_new_individual([job["a1"], job["a2"]])
    crossover_s = perf_counter() - crossover_t0
    mutation_s = 0.0
    if child is None:
        return {
            "index": job["index"],
            "child": None,
            "desc": None,
            "failure_reason": "pairing_failed",
            "operator_setup_s": operator_setup_s,
            "crossover_s": crossover_s,
            "mutation_s": mutation_s,
        }
    if _fails_fast_geometric_prefilter(child, ctx.blmin):
        return {
            "index": job["index"],
            "child": None,
            "desc": desc,
            "failure_reason": "too_close_prefilter",
            "operator_setup_s": operator_setup_s,
            "crossover_s": crossover_s,
            "mutation_s": mutation_s,
        }
    if decision_rng.random() < ctx.current_mutation_probability:
        mutation_t0 = perf_counter()
        mutated = local_mutations.get_operator().mutate(child)
        mutation_s = perf_counter() - mutation_t0
        if mutated is not None:
            child = mutated
    if ctx.freeze_adsorbate_internal_geometry:
        enforce_frozen_adsorbate_geometry(
            child,
            n_slab=ctx.n_slab,
            adsorbate_definition=ctx.adsorbate_definition,
            fragment_templates=ctx.adsorbate_fragment_templates,
        )
    maybe_apply_mobile_core_ads_tags(
        child,
        ctx.n_slab,
        ctx.composition,
        ctx.adsorbate_definition,
        ctx.system_type,
    )
    try:
        validate_structure_for_system_type(
            child,
            system_type=ctx.system_type,
            surface_config=ctx.surface_config,
            n_slab=ctx.n_slab,
            adsorbate_definition=ctx.adsorbate_definition,
            connectivity_factor=ctx.connectivity_factor,
            allow_cluster_fragmentation=ctx.allow_cluster_fragmentation,
            allow_adsorbate_surface_detachment=ctx.allow_adsorbate_surface_detachment,
            enforce_adsorbate_subgraph_integrity=ctx.enforce_adsorbate_subgraph_integrity,
        )
    except ValueError as exc:
        worker_logger.debug("Offspring rejected by system_type validation: %s", exc)
        return {
            "index": job["index"],
            "child": None,
            "desc": desc,
            "failure_reason": "validation_failed",
            "operator_setup_s": operator_setup_s,
            "crossover_s": crossover_s,
            "mutation_s": mutation_s,
        }
    return {
        "index": job["index"],
        "child": child,
        "desc": desc,
        "failure_reason": None,
        "operator_setup_s": operator_setup_s,
        "crossover_s": crossover_s,
        "mutation_s": mutation_s,
    }


def _torchsim_prepare_relaxed_copy(
    cand: Atoms,
    surface_config: SurfaceSystemConfig | None,
    n_slab: int,
    *,
    surface_mode: bool = False,
    freeze_adsorbate_internal_geometry: bool = False,
    adsorbate_definition: AdsorbateDefinition | None = None,
    adsorbate_fragment_templates: AdsorbateFragmentInput | None = None,
) -> Atoms:
    """Copy candidate and attach slab constraints before TorchSim relaxation."""
    if surface_config is not None and n_slab > 0 and not surface_mode:
        surface_mode = True
    c = cand.copy()
    if freeze_adsorbate_internal_geometry:
        enforce_frozen_adsorbate_geometry(
            c,
            n_slab=(n_slab if surface_mode else 0),
            adsorbate_definition=adsorbate_definition,
            fragment_templates=adsorbate_fragment_templates,
        )
    if surface_mode and surface_config is not None and n_slab > 0:
        attach_slab_constraints(
            c,
            n_slab,
            fix_all_slab_atoms=surface_config.fix_all_slab_atoms,
            n_fix_bottom_slab_layers=surface_config.n_fix_bottom_slab_layers,
            n_relax_top_slab_layers=surface_config.n_relax_top_slab_layers,
            surface_normal_axis=surface_config.surface_normal_axis,
        )
    if freeze_adsorbate_internal_geometry:
        attach_adsorbate_internal_geometry_constraints(
            c,
            n_slab=(n_slab if surface_mode else 0),
            adsorbate_definition=adsorbate_definition,
        )
    return c


def _record_relax_batch_steps(
    relaxer: TorchSimBatchRelaxer,
    profiling: dict[str, float] | None,
    counters: dict[str, int] | None,
    n_structures: int,
) -> None:
    steps_list = getattr(relaxer, "last_batch_relax_steps", None) or []
    if not steps_list or profiling is None:
        return
    step_val = steps_list[0]
    profiling["relax_steps_sum"] = profiling.get("relax_steps_sum", 0.0) + float(
        step_val * n_structures
    )
    profiling["relax_steps_max"] = max(
        float(profiling.get("relax_steps_max", 0.0)), float(step_val)
    )
    if counters is not None:
        counters["relax_batches"] = counters.get("relax_batches", 0) + 1
        counters["relax_structures"] = counters.get("relax_structures", 0) + int(
            n_structures
        )


def _relax_unrelaxed_candidates(
    da: DataConnection,
    relaxer: TorchSimBatchRelaxer,
    *,
    population: Population | None = None,
    max_batch: int | None = None,
    force: bool = False,
    generation: int | None = None,
    run_id: str | None = None,
    surface_config: SurfaceSystemConfig | None = None,
    n_slab: int = 0,
    system_type: SystemType = "gas_cluster",
    profiling: dict[str, float] | None = None,
    counters: dict[str, int] | None = None,
    composition: list[str] | None = None,
    adsorbate_definition: AdsorbateDefinition | None = None,
    connectivity_factor: float | None = None,
    allow_cluster_fragmentation: bool = False,
    allow_adsorbate_surface_detachment: bool = False,
    enforce_adsorbate_subgraph_integrity: bool = True,
    freeze_adsorbate_internal_geometry: bool = False,
    adsorbate_fragment_templates: AdsorbateFragmentInput | None = None,
) -> int:
    """Relax unrelaxed candidates in batches and commit them to the database."""
    available = database_retry(
        da.get_number_of_unrelaxed_candidates,
        config=RetryConfig(max_retries=5),
        operation_name="get_unrelaxed_candidates_count",
    )

    if available == 0:
        return 0
    if not force and max_batch is not None and available < max_batch:
        return 0

    to_take = available if force or max_batch is None else min(available, max_batch)

    # Batch read candidates under a single database connection
    def _read_batch_under_connection():
        """Read batch of candidates under a single connection in sorted gaid order."""
        with da.c:
            gaids = _sorted_unrelaxed_gaids(da)[:to_take]
            return [_load_unrelaxed_by_gaid(da, gaid) for gaid in gaids]

    t0 = perf_counter()
    batch = database_retry(
        _read_batch_under_connection,
        config=RetryConfig(max_retries=5),
        operation_name="read_candidate_batch",
    )
    if profiling is not None:
        profiling["db_read_s"] = profiling.get("db_read_s", 0.0) + (perf_counter() - t0)

    if not batch:
        return 0

    t0 = perf_counter()
    surface_mode = uses_surface(system_type)
    relaxed_results = relaxer.relax_batch(
        [
            _torchsim_prepare_relaxed_copy(
                cand,
                surface_config,
                n_slab,
                surface_mode=surface_mode,
                freeze_adsorbate_internal_geometry=freeze_adsorbate_internal_geometry,
                adsorbate_definition=adsorbate_definition,
                adsorbate_fragment_templates=adsorbate_fragment_templates,
            )
            for cand in batch
        ]
    )
    if profiling is not None:
        profiling["relax_batch_s"] = profiling.get("relax_batch_s", 0.0) + (
            perf_counter() - t0
        )
    _record_relax_batch_steps(relaxer, profiling, counters, len(batch))
    if len(relaxed_results) != len(batch):
        raise RuntimeError("TorchSim relaxer returned mismatched batch size")

    # Batch write results under a single database connection.
    # Disconnected structures are persisted but marked ineligible for GA evolution.
    successful_count = 0
    ineligible_count = 0
    logger = get_logger(__name__)

    def _write_batch_under_connection():
        """Write relaxed results under a single connection."""
        nonlocal ineligible_count, successful_count
        with da.c:
            for idx, (original, (energy, relaxed)) in enumerate(
                zip(batch, relaxed_results, strict=True)
            ):
                original.set_cell(relaxed.get_cell(), scale_atoms=True)
                original.set_pbc(relaxed.get_pbc())
                original.set_positions(relaxed.get_positions())
                if composition is not None:
                    maybe_apply_mobile_core_ads_tags(
                        original,
                        n_slab,
                        composition,
                        adsorbate_definition,
                        system_type,
                    )
                canonicalize_relaxed_for_storage(
                    original,
                    surface_mode=surface_mode,
                    n_slab=n_slab,
                )
                validation_error: str | None = None
                try:
                    validate_structure_for_system_type(
                        original,
                        system_type=system_type,
                        surface_config=surface_config,
                        n_slab=n_slab if surface_mode else None,
                        adsorbate_definition=adsorbate_definition,
                        connectivity_factor=connectivity_factor,
                        allow_cluster_fragmentation=allow_cluster_fragmentation,
                        allow_adsorbate_surface_detachment=allow_adsorbate_surface_detachment,
                        enforce_adsorbate_subgraph_integrity=enforce_adsorbate_subgraph_integrity,
                    )
                except ValueError as exc:
                    ineligible_count += 1
                    validation_error = str(exc)
                    logger.warning(
                        "Offspring %d/%d disconnected after relaxation; storing but excluding from GA population: %s",
                        idx + 1,
                        len(batch),
                        exc,
                    )

                # Copy forces if available (already converted to float64 by relaxer)
                if "forces" in relaxed.arrays:
                    original.arrays["forces"] = relaxed.arrays["forces"].copy()

                original.info.setdefault("key_value_pairs", {})
                update_metadata(
                    original,
                    **relaxed.info.get(
                        "key_value_pairs",
                        {"potential_energy": energy, "raw_score": -energy},
                    ),
                )
                update_metadata(
                    original,
                    ga_eligible=(validation_error is None),
                )
                original.info.setdefault("key_value_pairs", {})["ga_eligible"] = (
                    validation_error is None
                )
                if validation_error is not None:
                    update_metadata(
                        original,
                        ga_ineligible_reason=validation_error,
                    )
                    original.info.setdefault("key_value_pairs", {})[
                        "ga_ineligible_reason"
                    ] = validation_error
                comp_meta = list(composition) if composition is not None else []
                extra = ga_run_metadata_extras(
                    surface_config,
                    n_slab,
                    system_type,
                    comp_meta,
                    adsorbate_definition=adsorbate_definition,
                )
                if generation is not None:
                    add_metadata(
                        original,
                        generation=generation,
                        run_id=run_id,
                        **extra,
                    )
                elif run_id is not None:
                    add_metadata(original, run_id=run_id, **extra)

                original.calc = SinglePointCalculator(original, energy=energy)
                da.add_relaxed_step(original)
                if validation_error is None:
                    successful_count += 1

    t0 = perf_counter()
    database_retry(
        _write_batch_under_connection,
        config=RetryConfig(max_retries=5),
        operation_name="write_relaxed_batch",
    )
    if ineligible_count > 0:
        logger.info(
            "Relaxation batch: %d/%d individuals are GA-eligible, %d persisted as ineligible",
            len(batch) - ineligible_count,
            len(batch),
            ineligible_count,
        )
    if profiling is not None:
        profiling["db_write_s"] = profiling.get("db_write_s", 0.0) + (
            perf_counter() - t0
        )

    if population is not None:
        t0 = perf_counter()
        population.update()
        if profiling is not None:
            profiling["population_update_s"] = profiling.get(
                "population_update_s", 0.0
            ) + (perf_counter() - t0)

    return successful_count


def ga_go(
    composition: list[str],
    output_dir: str,
    rng: np.random.Generator | None,
    calculator: Any,
    *,
    niter: int = 10,
    fmax: float = 0.05,
    niter_local_relaxation: int = 250,
    optimizer: type[Optimizer] = FIRE,
    energy_tolerance: float = DEFAULT_ENERGY_TOLERANCE,
    mutation_probability: float = 0.4,
    population_size: int = 10,
    offspring_fraction: float = 0.5,
    n_jobs_population_init: int = -2,
    n_jobs_offspring: int = -2,
    vacuum: float = 10.0,
    previous_search_glob: str = "**/*.db",
    use_adaptive_mutations: bool = True,
    stagnation_trigger: int = 4,
    stagnation_full_trigger: int = 8,
    recovery_window: int = 2,
    aggressive_burst_multiplier: float = 1.8,
    max_mutation_probability: float = 0.65,
    early_stopping_niter: int = 10,
    relaxer: TorchSimBatchRelaxer | None = None,
    batch_size: int | None = None,
    verbosity: int = 1,
    elite_fraction: float = 0.1,
    run_id: str | None = None,
    clean: bool = False,
    fitness_strategy: str = "low_energy",
    diversity_reference_db: str | None = None,
    diversity_max_references: int = 100,
    diversity_update_interval: int = 5,
    surface_config: SurfaceSystemConfig | None = None,
    system_type: SystemType = "gas_cluster",
    write_timing_json: bool = False,
    detailed_timing: bool = False,
    timing_output_dir: str | None = None,
    timing_collector: list[dict[str, Any]] | None = None,
    adsorbate_definition: AdsorbateDefinition | None = None,
    adsorbate_fragment_template: AdsorbateFragmentInput | None = None,
    cluster_adsorbate_config: ClusterAdsorbateConfig | None = None,
    connectivity_factor: float | None = None,
    allow_cluster_fragmentation: bool = False,
    allow_adsorbate_surface_detachment: bool = False,
    enforce_adsorbate_subgraph_integrity: bool = True,
    freeze_adsorbate_internal_geometry: bool = False,
    ga_adaptive_retry_enabled: bool = True,
    ga_retry_floor_multiplier: int = 4,
    ga_retry_ceiling_multiplier: int = 15,
    ga_fast_prefilter_enabled: bool = True,
    db_enable_expression_indexes: bool = False,
) -> list[tuple[float, Atoms]]:
    """Run the GA using TorchSim for batched relaxations.

    Genetic algorithm with batched relaxations (TorchSim for MLIPs, ASE batch otherwise).
    The ``relaxer`` argument controls TorchSim batching; when omitted the
    function instantiates a default :class:`TorchSimBatchRelaxer` using the
    provided ``fmax`` as a force tolerance.

    Args:
        composition: List of element symbols defining the cluster composition.
        calculator: ASE calculator for energy/force evaluations.
        previous_search_glob: Glob pattern used to discover previous database
            files for seed-based initialization.
        early_stopping_niter: Number of consecutive generations with no improvement
                              before stopping early. Uses fitness for non-low_energy
                              strategies, energy for low_energy. If 0, no early stopping
                              is applied. Default 10.
        verbosity: Verbosity level (0=quiet, 1=normal, 2=debug, 3=trace). Defaults to 1.
        elite_fraction: Fraction of population to preserve as elite candidates
                         (top performers by fitness). Default 0.1 (top 10%).
        run_id: Optional run ID for tracking.
        clean: If True, start fresh (ignore previous databases).
        fitness_strategy: Fitness strategy to use. One of: "low_energy", "high_energy", "diversity".
            Defaults to "low_energy" (minimize energy).
        diversity_reference_db: Glob pattern for reference structure databases (for diversity strategy).
            Required when fitness_strategy="diversity", ignored otherwise.
        diversity_max_references: Maximum number of reference structures to load (for performance).
        diversity_update_interval: Number of generations between reference updates (for diversity strategy).
        surface_config: Optional slab + adsorbate configuration for surface GA runs.
        write_timing_json: If True, write ``timing.json`` (see ``timing_output_dir``).
            Set in ``optimizer_params['ga']`` inside ``go_params``/``params``.
        detailed_timing: If True, include ``per_generation`` rows in ``timing.json``.
            Requires ``write_timing_json=True``.
        timing_output_dir: Directory for ``timing.json`` (defaults to ``output_dir``).
            ``run_trials`` sets this to the run directory alongside ``metadata.json``.
        timing_collector: Optional list appended with the timing payload after the run.
    """
    logger = get_logger(__name__)
    profile_t0 = perf_counter()
    profile_timings: dict[str, float] = {}
    profile_counters: dict[str, int] = {
        "offspring_created": 0,
        "offspring_relaxed": 0,
        "offspring_worker_failures": 0,
        "offspring_attempts_total": 0,
    }
    profile_retry_failures: dict[str, int] = {}
    per_generation: list[dict[str, Any]] | None = [] if detailed_timing else None

    from scgo.system_types import resolve_connectivity_factor

    connectivity_factor = resolve_connectivity_factor(
        connectivity_factor,
        cluster_adsorbate_config=cluster_adsorbate_config,
        surface_config=surface_config,
    )
    validate_composition(composition, allow_empty=False, allow_tuple=False)
    validate_system_type_settings(
        system_type=system_type, surface_config=surface_config
    )
    validate_ga_common_params(
        niter=niter,
        population_size=population_size,
        n_jobs_population_init=n_jobs_population_init,
        calculator=calculator,
        mutation_probability=mutation_probability,
        offspring_fraction=offspring_fraction,
        vacuum=vacuum,
        fmax=fmax,
    )
    if n_jobs_offspring not in (-1, -2) and n_jobs_offspring < 1:
        raise ValueError(
            f"n_jobs_offspring must be -1, -2, or >= 1, got {n_jobs_offspring}"
        )

    # Validate and normalize fitness strategy (coerce to Enum)
    fitness_strategy = FitnessStrategy(
        ensure_fitness_strategy_resolved(fitness_strategy)
    )

    if batch_size is not None and batch_size <= 0:
        batch_size = None

    # Normalize RNG early and enforce Generator-only policy
    rng = ensure_rng_or_create(rng)

    if relaxer is None:
        if is_ml_calculator(calculator):
            relaxer = TorchSimBatchRelaxer(
                force_tol=fmax,
                mace_model_name="mace_matpes_0",
                max_steps=niter_local_relaxation,
            )
        else:
            relaxer = AseBatchRelaxer(
                calculator,
                optimizer=optimizer,
                force_tol=fmax,
                max_steps=niter_local_relaxation,
            )
    elif (
        isinstance(niter_local_relaxation, int) and niter_local_relaxation > 0
    ) or relaxer.max_steps is None:
        relaxer.max_steps = niter_local_relaxation

    n_to_optimize = len(composition)

    surface_mode = uses_surface(system_type)
    if surface_mode:
        if not isinstance(surface_config, SurfaceSystemConfig):
            raise TypeError(
                "surface_config must be a SurfaceSystemConfig instance or None"
            )
        slab_ref = surface_config.slab.copy()
        n_slab = len(slab_ref)
        dummy_top = [[0.0, 0.0, 0.0] for _ in range(n_to_optimize)]
        atoms_template = Atoms(
            symbols=list(slab_ref.get_chemical_symbols()) + list(composition),
            positions=np.vstack([slab_ref.get_positions(), np.asarray(dummy_top)]),
            cell=slab_ref.get_cell(),
            pbc=slab_ref.get_pbc(),
        )
    else:
        n_slab = 0
        slab_ref = None
        cell_side = compute_cell_side(composition, vacuum=vacuum)
        atoms_template = Atoms(
            symbols=composition,
            positions=[[0, 0, 0] for _ in range(n_to_optimize)],  # Dummy positions
            cell=[cell_side] * 3,
            pbc=False,
        )
    atoms_template.calc = calculator

    # Load reference structures and create DiversityScorer for diversity strategy
    diversity_scorer = setup_diversity_scorer(
        fitness_strategy=fitness_strategy,
        diversity_reference_db=diversity_reference_db,
        composition=composition,
        n_to_optimize=n_to_optimize,
        diversity_max_references=diversity_max_references,
        logger=logger,
        base_dir=output_dir,
    )

    slab_for_pairing = slab_ref if surface_mode else None
    _ = create_ga_pairing(
        atoms_template,
        n_to_optimize,
        rng,
        slab_atoms=slab_for_pairing,
        system_type=system_type,
        composition=composition,
        adsorbate_definition=adsorbate_definition,
    )

    adaptive_config = get_adaptive_mutation_config(
        composition=composition,
        current_generation=0,
        total_generations=niter,
        use_adaptive=use_adaptive_mutations,
        generations_without_improvement=0,
        stagnation_trigger=stagnation_trigger,
        stagnation_full_trigger=stagnation_full_trigger,
        recovery_window=recovery_window,
        aggressive_burst_multiplier=aggressive_burst_multiplier,
        max_mutation_probability=max_mutation_probability,
    )

    idx_top = (
        range(n_slab, n_slab + n_to_optimize) if surface_mode else range(n_to_optimize)
    )
    top_z = list({int(atoms_template[i].number) for i in idx_top})
    all_atom_types = get_all_atom_types(atoms_template, top_z)
    blmin = build_blmin_from_zs(all_atom_types, ratio=BLMIN_RATIO_DEFAULT)

    operators_list, name_map = create_mutation_operators(
        composition=composition,
        n_to_optimize=n_to_optimize,
        blmin=blmin,
        rng=rng,
        use_adaptive=use_adaptive_mutations,
        system_type=system_type,
        n_slab=n_slab,
        surface_normal_axis=(surface_config.surface_normal_axis if surface_mode else 2),
        adsorbate_definition=adsorbate_definition,
        freeze_adsorbate_internal_geometry=freeze_adsorbate_internal_geometry,
        adsorbate_fragment_template=adsorbate_fragment_template,
        cluster_adsorbate_config=cluster_adsorbate_config,
    )

    _ = update_mutation_weights(
        operators_list=operators_list,
        name_map=name_map,
        adaptive_config=adaptive_config,
        rng=rng,
    )
    # Use user-provided mutation_probability when adaptive mutations are disabled
    current_mutation_probability = (
        mutation_probability
        if not use_adaptive_mutations
        else adaptive_config["mutation_probability"]
    )

    comp_mic = bool(surface_config.comparator_use_mic) if surface_mode else False
    comp = create_structure_comparator(n_to_optimize, energy_tolerance, mic=comp_mic)

    t0_batch_build = perf_counter()
    if surface_mode:
        assert slab_ref is not None
        start_generator = SurfaceClusterStartGenerator(
            composition,
            slab_ref,
            surface_config,
            blmin,
            rng=rng,
            calculator=None,
            population_size=population_size,
            previous_search_glob=previous_search_glob,
            n_jobs=n_jobs_population_init,
            adsorbate_definition=adsorbate_definition,
            adsorbate_fragment_template=adsorbate_fragment_template,
            cluster_adsorbate_config=cluster_adsorbate_config,
        )
    else:
        start_generator = ClusterStartGenerator(
            composition,
            vacuum,
            rng=rng,
            calculator=None,  # Do not attach calculator to initial population to avoid pickling issues
            population_size=population_size,
            mode="smart",
            previous_search_glob=previous_search_glob,
            n_jobs=n_jobs_population_init,
            system_type=system_type,
            adsorbate_definition=adsorbate_definition,
            adsorbate_fragment_template=adsorbate_fragment_template,
            cluster_adsorbate_config=cluster_adsorbate_config,
        )
    profile_timings["initial_population_batch_build_s"] = (
        perf_counter() - t0_batch_build
    )
    t0 = perf_counter()
    initial_population = [
        start_generator.get_new_candidate() for _ in range(population_size)
    ]
    profile_timings["initial_population_generation_s"] = perf_counter() - t0

    if verbosity >= 1:
        n_workers = (
            "all CPUs"
            if n_jobs_population_init == -1
            else "all but one CPU"
            if n_jobs_population_init == -2
            else f"{n_jobs_population_init} workers"
        )
        logger.info(
            f"Generated initial population of {population_size} candidates "
            f"(batched, parallel n_jobs={n_workers})"
        )

    # Do not pass initial_population to SetupDB (avoids formula keys in key_value_pairs).
    # Insert unrelaxed starters via the low-level API, then batch-relax with TorchSim and tag generation=0.
    da = setup_database(
        output_dir=output_dir,
        db_filename="ga_go.db",
        atoms_template=atoms_template,
        initial_population=None,
        remove_existing=clean,
        remove_aux_files=clean,
        enable_expression_indexes=db_enable_expression_indexes,
        run_id=run_id,
    )

    try:
        if verbosity >= 1:
            logger.info(
                f"Relaxing initial population of {population_size} candidates..."
            )

        logger.debug(
            "Using GA database at %s",
            os.path.join(output_dir, "ga_go.db"),
        )

        initial_pop_count = 0
        initial_discarded_count = 0
        initial_ineligible_relaxed_count = 0

        def _insert_unrelaxed(cand):
            cand.info.setdefault("key_value_pairs", {})
            cand.info.setdefault("data", {})
            gaid = da.c.write(
                cand,
                origin="StartingCandidateUnrelaxed",
                relaxed=0,
                generation=0,
                extinct=0,
                description="initial",
            )
            da.c.update(gaid, gaid=gaid)
            cand.info["confid"] = gaid

        t0 = perf_counter()
        with da.c:
            for cand in initial_population:
                if adsorbate_definition is None and not surface_mode:
                    cand = reorder_cluster_to_composition(cand, list(composition))
                maybe_apply_mobile_core_ads_tags(
                    cand,
                    n_slab,
                    composition,
                    adsorbate_definition,
                    system_type,
                )
                if freeze_adsorbate_internal_geometry:
                    enforce_frozen_adsorbate_geometry(
                        cand,
                        n_slab=n_slab,
                        adsorbate_definition=adsorbate_definition,
                        fragment_templates=adsorbate_fragment_template,
                    )
                try:
                    validate_structure_for_system_type(
                        cand,
                        system_type=system_type,
                        surface_config=surface_config,
                        n_slab=n_slab,
                        adsorbate_definition=adsorbate_definition,
                        connectivity_factor=connectivity_factor,
                        allow_cluster_fragmentation=allow_cluster_fragmentation,
                        allow_adsorbate_surface_detachment=allow_adsorbate_surface_detachment,
                        enforce_adsorbate_subgraph_integrity=enforce_adsorbate_subgraph_integrity,
                    )
                except ValueError as exc:
                    initial_discarded_count += 1
                    logger.warning(
                        "Discarding disconnected initial candidate before DB insert: %s",
                        exc,
                    )
                    continue
                database_retry(
                    lambda _cand=cand: _insert_unrelaxed(_cand),
                    config=RetryConfig(max_retries=5),
                    operation_name="insert_unrelaxed_candidate",
                )
        profile_timings["initial_unrelaxed_insert_s"] = perf_counter() - t0

        # Helper to write a relaxed batch into the database under a single connection
        def _write_relaxed_batch(batch, relaxed_results):
            nonlocal initial_ineligible_relaxed_count
            with da.c:
                for original, (energy, relaxed) in zip(
                    batch, relaxed_results, strict=True
                ):
                    original.set_cell(relaxed.get_cell(), scale_atoms=True)
                    original.set_pbc(relaxed.get_pbc())
                    original.set_positions(relaxed.get_positions())
                    maybe_apply_mobile_core_ads_tags(
                        original,
                        n_slab,
                        composition,
                        adsorbate_definition,
                        system_type,
                    )
                    canonicalize_relaxed_for_storage(
                        original,
                        surface_mode=surface_mode,
                        n_slab=n_slab,
                    )
                    validation_error: str | None = None
                    try:
                        validate_structure_for_system_type(
                            original,
                            system_type=system_type,
                            surface_config=surface_config,
                            n_slab=n_slab if surface_mode else None,
                            adsorbate_definition=adsorbate_definition,
                            connectivity_factor=connectivity_factor,
                            allow_cluster_fragmentation=allow_cluster_fragmentation,
                            allow_adsorbate_surface_detachment=allow_adsorbate_surface_detachment,
                            enforce_adsorbate_subgraph_integrity=enforce_adsorbate_subgraph_integrity,
                        )
                    except ValueError as exc:
                        validation_error = str(exc)
                        initial_ineligible_relaxed_count += 1
                        logger.warning(
                            "Initial candidate disconnected after relaxation; storing but excluding from GA population: %s",
                            exc,
                        )

                    # Copy forces if available
                    if "forces" in relaxed.arrays:
                        original.arrays["forces"] = relaxed.arrays["forces"].copy()

                    original.info.setdefault("key_value_pairs", {})
                    update_metadata(
                        original,
                        **relaxed.info.get(
                            "key_value_pairs",
                            {"potential_energy": energy, "raw_score": -energy},
                        ),
                    )
                    update_metadata(
                        original,
                        ga_eligible=(validation_error is None),
                    )
                    original.info.setdefault("key_value_pairs", {})["ga_eligible"] = (
                        validation_error is None
                    )
                    if validation_error is not None:
                        update_metadata(
                            original,
                            ga_ineligible_reason=validation_error,
                        )
                        original.info.setdefault("key_value_pairs", {})[
                            "ga_ineligible_reason"
                        ] = validation_error
                    add_metadata(
                        original,
                        generation=0,
                        run_id=run_id,
                        **ga_run_metadata_extras(
                            surface_config,
                            n_slab,
                            system_type,
                            composition,
                            adsorbate_definition=adsorbate_definition,
                        ),
                    )
                    original.calc = SinglePointCalculator(original, energy=energy)
                    da.add_relaxed_step(original)

        # Process starting population in batches
        batch_size_internal = batch_size or len(initial_population)
        t0_relax = 0.0
        t0_write = 0.0
        for i in range(0, len(initial_population), batch_size_internal):
            batch = initial_population[i : i + batch_size_internal]
            t_start = perf_counter()
            relaxed_results = relaxer.relax_batch(
                [
                    _torchsim_prepare_relaxed_copy(
                        c,
                        surface_config,
                        n_slab,
                        surface_mode=surface_mode,
                        freeze_adsorbate_internal_geometry=freeze_adsorbate_internal_geometry,
                        adsorbate_definition=adsorbate_definition,
                    )
                    for c in batch
                ]
            )
            t0_relax += perf_counter() - t_start
            _record_relax_batch_steps(
                relaxer, profile_timings, profile_counters, len(batch)
            )
            if len(relaxed_results) != len(batch):
                raise RuntimeError("TorchSim relaxer returned mismatched batch size")

            t_start = perf_counter()
            database_retry(
                lambda _batch=batch, _results=relaxed_results: _write_relaxed_batch(
                    _batch, _results
                ),
                config=RetryConfig(max_retries=5),
                operation_name="write_initial_relaxed_batch",
            )
            t0_write += perf_counter() - t_start

            initial_pop_count += len(batch)
        profile_timings["initial_relax_batch_s"] = t0_relax
        profile_timings["initial_relaxed_write_s"] = t0_write

        if initial_pop_count > 0:
            logger.debug(
                "Tagged %s GA population members with generation=0",
                initial_pop_count,
            )
        if initial_discarded_count > 0:
            logger.info(
                "Discarded %d disconnected initial candidates before DB insert",
                initial_discarded_count,
            )
        if initial_ineligible_relaxed_count > 0:
            logger.info(
                "Stored %d initial relaxed candidates as GA-ineligible",
                initial_ineligible_relaxed_count,
            )

        log_file = os.path.join(output_dir, "population.log")

        with suppress(FileNotFoundError):
            os.remove(log_file)

        # Select appropriate Population class based on fitness strategy
        PopulationClass, population_kwargs = select_population_class(
            fitness_strategy=fitness_strategy,
            diversity_scorer=diversity_scorer,
            diversity_update_interval=diversity_update_interval,
            logger=logger,
        )

        population = PopulationClass(
            data_connection=da,
            population_size=population_size,
            comparator=comp,
            logfile=log_file,
            rng=rng,  # type: ignore[arg-type]
            elite_fraction=elite_fraction,
            run_id=run_id,
            **population_kwargs,
        )
        population._write_log()
        logger.debug(
            "Initial Population created: size=%d, confids=%s",
            len(population.pop),
            [a.info.get("confid") for a in population.pop],
        )

        log_early_stopping_info(
            verbosity=verbosity,
            fitness_strategy=fitness_strategy,
            early_stopping_niter=early_stopping_niter,
            niter=niter,
            logger=logger,
        )

        # Track best value for early stopping (energy or fitness)
        best_value = None  # Energy for low_energy, fitness for others
        generations_without_improvement = 0
        recent_acceptance_ratios: list[float] = []

        for generation in tqdm(
            range(niter),
            desc=f"  GA generations for {len(composition)} atoms",
            disable=not should_show_progress(verbosity),
        ):
            if use_adaptive_mutations:
                adaptive_config = get_adaptive_mutation_config(
                    composition=composition,
                    current_generation=generation,
                    total_generations=niter,
                    use_adaptive=True,
                    generations_without_improvement=generations_without_improvement,
                    stagnation_trigger=stagnation_trigger,
                    stagnation_full_trigger=stagnation_full_trigger,
                    recovery_window=recovery_window,
                    aggressive_burst_multiplier=aggressive_burst_multiplier,
                    max_mutation_probability=max_mutation_probability,
                )
                _ = update_mutation_weights(
                    operators_list=operators_list,
                    name_map=name_map,
                    adaptive_config=adaptive_config,
                    rng=rng,
                )
                current_mutation_probability = adaptive_config["mutation_probability"]

            # Create up to `n_offspring` unrelaxed candidates for this generation;
            # TorchSim will handle batching/relaxation later.
            n_offspring = max(1, math.ceil(population_size * offspring_fraction))
            created = 0
            attempts = 0
            max_attempts = max(10, n_offspring * 10)
            if ga_adaptive_retry_enabled:
                recent_ratio = (
                    float(np.mean(recent_acceptance_ratios[-5:]))
                    if recent_acceptance_ratios
                    else 0.35
                )
                target_ratio = max(0.05, min(0.95, recent_ratio))
                estimated_needed = int(math.ceil(n_offspring / target_ratio))
                floor_attempts = max(10, n_offspring * int(ga_retry_floor_multiplier))
                ceil_attempts = max(
                    floor_attempts, n_offspring * int(ga_retry_ceiling_multiplier)
                )
                max_attempts = max(floor_attempts, min(estimated_needed, ceil_attempts))

            t_loop = perf_counter()
            t_parent_select_gen = 0.0
            t_operator_setup_gen = 0.0
            t_crossover_gen = 0.0
            t_mutation_gen = 0.0
            t_db_unrelaxed_gen = 0.0
            t_offspring_parallel_wall_gen = 0.0
            worker_failures_gen = 0
            worker_failure_types_gen: dict[str, int] = {}
            retry_failure_reasons_gen: dict[str, int] = {}
            while created < n_offspring and attempts < max_attempts:
                attempts_remaining = max_attempts - attempts
                if attempts_remaining <= 0:
                    break
                jobs_target = min(n_offspring - created, attempts_remaining)
                jobs: list[dict[str, Any]] = []
                for _ in range(jobs_target):
                    attempts += 1
                    t0 = perf_counter()
                    candidates = population.get_two_candidates()
                    t_parent_select_gen += perf_counter() - t0
                    if candidates is None:
                        continue
                    a1, a2 = candidates
                    task_seed = int(rng.integers(0, 2**31 - 1))
                    jobs.append(
                        {
                            "index": len(jobs),
                            "a1": a1.copy(),
                            "a2": a2.copy(),
                            "task_seed": task_seed,
                        }
                    )
                if not jobs:
                    continue

                n_workers = _resolve_parallel_worker_count(n_jobs_offspring, len(jobs))
                offspring_ctx = OffspringBuildContext(
                    atoms_template=_picklable_atoms_copy(atoms_template),
                    n_to_optimize=n_to_optimize,
                    composition=composition,
                    blmin=blmin if ga_fast_prefilter_enabled else {},
                    system_type=system_type,
                    n_slab=n_slab,
                    slab_for_pairing=_picklable_atoms_copy(slab_for_pairing),
                    surface_normal_axis=(
                        surface_config.surface_normal_axis if surface_mode else 2
                    ),
                    adsorbate_definition=adsorbate_definition,
                    connectivity_factor=connectivity_factor,
                    allow_cluster_fragmentation=allow_cluster_fragmentation,
                    allow_adsorbate_surface_detachment=allow_adsorbate_surface_detachment,
                    enforce_adsorbate_subgraph_integrity=enforce_adsorbate_subgraph_integrity,
                    freeze_adsorbate_internal_geometry=freeze_adsorbate_internal_geometry,
                    adsorbate_fragment_templates=_picklable_fragment_templates(
                        adsorbate_fragment_template
                    ),
                    surface_config=surface_config,
                    adaptive_config=adaptive_config,
                    current_mutation_probability=current_mutation_probability,
                    operators_list=copy.deepcopy(operators_list),
                    name_map=name_map,
                )

                t_parallel = perf_counter()
                job_results: dict[int, dict[str, Any]] = {}
                worker_exceptions: list[BaseException] = []
                if n_workers == 1:
                    for job in jobs:
                        try:
                            result = _build_offspring_worker(job, offspring_ctx)
                        except (RuntimeError, ValueError, TypeError) as exc:
                            worker_failures_gen += 1
                            err_name = type(exc).__name__
                            worker_failure_types_gen[err_name] = (
                                worker_failure_types_gen.get(err_name, 0) + 1
                            )
                            reason = f"worker_exception_{err_name}"
                            retry_failure_reasons_gen[reason] = (
                                retry_failure_reasons_gen.get(reason, 0) + 1
                            )
                            worker_exceptions.append(exc)
                            logger.exception(
                                "Offspring crossover/mutation worker failed (%s)",
                                err_name,
                            )
                            continue
                        job_results[result["index"]] = result
                else:
                    with ProcessPoolExecutor(
                        max_workers=n_workers,
                        initializer=_offspring_worker_init,
                    ) as executor:
                        futures = [
                            executor.submit(_build_offspring_worker, job, offspring_ctx)
                            for job in jobs
                        ]
                        for future in as_completed(futures):
                            try:
                                result = future.result()
                            except (RuntimeError, ValueError, TypeError) as exc:
                                worker_failures_gen += 1
                                err_name = type(exc).__name__
                                worker_failure_types_gen[err_name] = (
                                    worker_failure_types_gen.get(err_name, 0) + 1
                                )
                                reason = f"worker_exception_{err_name}"
                                retry_failure_reasons_gen[reason] = (
                                    retry_failure_reasons_gen.get(reason, 0) + 1
                                )
                                worker_exceptions.append(exc)
                                logger.exception(
                                    "Offspring crossover/mutation worker failed (%s)",
                                    err_name,
                                )
                                continue
                            job_results[result["index"]] = result
                if len(jobs) > 0 and len(job_results) == 0 and worker_exceptions:
                    first = worker_exceptions[0]
                    if not all(isinstance(e, ValueError) for e in worker_exceptions):
                        raise RuntimeError(
                            f"All {len(jobs)} parallel offspring workers failed"
                        ) from first
                t_offspring_parallel_wall_gen += perf_counter() - t_parallel
                if worker_failures_gen:
                    profile_counters["offspring_worker_failures"] += worker_failures_gen
                    failure_limit = max(3, len(jobs) // 2)
                    if worker_failures_gen >= failure_limit:
                        logger.warning(
                            "Generation %s offspring worker failures: %d/%d (%s)",
                            generation,
                            worker_failures_gen,
                            len(jobs),
                            worker_failure_types_gen,
                        )

                pending_inserts: list[tuple[Atoms, str]] = []
                for idx in range(len(jobs)):
                    if created >= n_offspring:
                        break
                    result = job_results.get(idx)
                    if result is None:
                        continue
                    t_operator_setup_gen += float(result["operator_setup_s"])
                    t_crossover_gen += float(result["crossover_s"])
                    t_mutation_gen += float(result["mutation_s"])
                    child = result["child"]
                    if child is None:
                        reason = result.get("failure_reason") or "unknown"
                        retry_failure_reasons_gen[reason] = (
                            retry_failure_reasons_gen.get(reason, 0) + 1
                        )
                        continue
                    pending_inserts.append((child, result["desc"]))
                if pending_inserts:
                    t0 = perf_counter()
                    with da.c:
                        for child, desc in pending_inserts:
                            database_retry(
                                lambda _a3=child, _desc=desc: (
                                    da.add_unrelaxed_candidate(_a3, description=_desc)
                                ),
                                config=RetryConfig(max_retries=5),
                                operation_name="add_unrelaxed_offspring",
                            )
                            created += 1
                    t_db_unrelaxed_gen += perf_counter() - t0
            generation_acceptance = created / max(attempts, 1)
            recent_acceptance_ratios.append(generation_acceptance)
            profile_counters["offspring_attempts_total"] += attempts
            for reason, count in retry_failure_reasons_gen.items():
                profile_retry_failures[reason] = (
                    profile_retry_failures.get(reason, 0) + count
                )
            profile_timings["offspring_mutation_queue_s"] = profile_timings.get(
                "offspring_mutation_queue_s", 0.0
            ) + (perf_counter() - t_loop)
            profile_timings["offspring_parent_select_s"] = (
                profile_timings.get("offspring_parent_select_s", 0.0)
                + t_parent_select_gen
            )
            profile_timings["offspring_operator_setup_s"] = (
                profile_timings.get("offspring_operator_setup_s", 0.0)
                + t_operator_setup_gen
            )
            profile_timings["offspring_crossover_s"] = (
                profile_timings.get("offspring_crossover_s", 0.0) + t_crossover_gen
            )
            profile_timings["offspring_mutation_s"] = (
                profile_timings.get("offspring_mutation_s", 0.0) + t_mutation_gen
            )
            profile_timings["offspring_unrelaxed_insert_s"] = (
                profile_timings.get("offspring_unrelaxed_insert_s", 0.0)
                + t_db_unrelaxed_gen
            )
            profile_timings["offspring_parallel_wall_s"] = (
                profile_timings.get("offspring_parallel_wall_s", 0.0)
                + t_offspring_parallel_wall_gen
            )
            profile_counters["offspring_created"] += created

            # Emit a concise per-generation summary at DEBUG level (one line)
            logger.debug(
                "Generation %s offspring loop: n_offspring=%d, created=%d, attempts=%d",
                generation,
                n_offspring,
                created,
                attempts,
            )

            # Ask TorchSim relaxer to process available unrelaxed candidates now.
            # Enforce a per-generation limit: when `batch_size` is None, treat the
            # per-call limit as the GA `n_offspring` so a single relax call does not
            # drain an unrelated backlog and make logs look cumulative.
            per_gen_max = batch_size if batch_size is not None else n_offspring
            pre_db_read = float(profile_timings.get("db_read_s", 0.0))
            pre_relax = float(profile_timings.get("relax_batch_s", 0.0))
            pre_db_write = float(profile_timings.get("db_write_s", 0.0))
            pre_pop_update = float(profile_timings.get("population_update_s", 0.0))
            t0_relax_call = perf_counter()
            offspring_count = _relax_unrelaxed_candidates(
                da,
                relaxer,
                population=population,
                max_batch=per_gen_max,
                generation=generation,
                run_id=run_id,
                surface_config=surface_config,
                n_slab=n_slab,
                system_type=system_type,
                profiling=profile_timings,
                counters=profile_counters,
                composition=composition,
                adsorbate_definition=adsorbate_definition,
                connectivity_factor=connectivity_factor,
                allow_cluster_fragmentation=allow_cluster_fragmentation,
                allow_adsorbate_surface_detachment=allow_adsorbate_surface_detachment,
                enforce_adsorbate_subgraph_integrity=enforce_adsorbate_subgraph_integrity,
                freeze_adsorbate_internal_geometry=freeze_adsorbate_internal_geometry,
                adsorbate_fragment_templates=adsorbate_fragment_template,
            )
            relax_call_wall_s = perf_counter() - t0_relax_call
            post_db_read = float(profile_timings.get("db_read_s", 0.0))
            post_relax = float(profile_timings.get("relax_batch_s", 0.0))
            post_db_write = float(profile_timings.get("db_write_s", 0.0))
            post_pop_update = float(profile_timings.get("population_update_s", 0.0))
            gen_db_read_s = max(0.0, post_db_read - pre_db_read)
            gen_relax_s = max(0.0, post_relax - pre_relax)
            gen_db_write_s = max(0.0, post_db_write - pre_db_write)
            gen_pop_update_s_from_relax = max(0.0, post_pop_update - pre_pop_update)
            pop_update_s = gen_pop_update_s_from_relax
            if offspring_count > 0:
                profile_counters["offspring_relaxed"] += int(offspring_count)
                # Attempt to report a concise triple: created_this_gen / relaxed_this_call / total_relaxed
                try:
                    total_relaxed_cnt = database_retry(
                        lambda: _count_relaxed_candidates(da),
                        config=RetryConfig(max_retries=5),
                        operation_name="count_relaxed_candidates",
                    )
                except HPC_DATABASE_EXCEPTIONS:
                    total_relaxed_cnt = None

                if total_relaxed_cnt is not None:
                    logger.debug(
                        "Generation %s: created=%d, relaxed_this_call=%d, total_relaxed=%d",
                        generation,
                        created,
                        offspring_count,
                        total_relaxed_cnt,
                    )
                else:
                    logger.debug(
                        "Generation %s: created=%d, relaxed_this_call=%d",
                        generation,
                        created,
                        offspring_count,
                    )

            if per_generation is not None:
                per_generation.append(
                    {
                        "generation": int(generation),
                        "n_offspring_target": int(n_offspring),
                        "offspring_created": int(created),
                        "attempts": int(attempts),
                        "acceptance_ratio": float(generation_acceptance),
                        "offspring_relaxed_this_call": int(offspring_count),
                        "retry_failures": dict(retry_failure_reasons_gen),
                        "timings_s": {
                            "parent_select_s": t_parent_select_gen,
                            "operator_setup_s": t_operator_setup_gen,
                            "crossover_s": t_crossover_gen,
                            "mutation_s": t_mutation_gen,
                            "db_unrelaxed_insert_s": t_db_unrelaxed_gen,
                            "offspring_parallel_wall_s": t_offspring_parallel_wall_gen,
                            "torchsim_db_read_s": gen_db_read_s,
                            "torchsim_relax_s": gen_relax_s,
                            "torchsim_db_write_s": gen_db_write_s,
                            "torchsim_relax_call_wall_s": relax_call_wall_s,
                            "population_update_s": pop_update_s,
                            "population_update_s_from_relax": gen_pop_update_s_from_relax,
                            "offspring_loop_wall_s": perf_counter() - t_loop,
                        },
                    }
                )

            if early_stopping_niter > 0:
                best_value, generations_without_improvement, should_stop = (
                    update_early_stopping_state_unified(
                        population=population,
                        fitness_strategy=fitness_strategy,
                        best_value=best_value,
                        generations_without_improvement=generations_without_improvement,
                        early_stopping_niter=early_stopping_niter,
                    )
                )
                if should_stop:
                    if verbosity >= 1:
                        stopping_metric = (
                            "fitness"
                            if fitness_strategy != FitnessStrategy.LOW_ENERGY
                            else "energy"
                        )
                        logger.info(
                            f"Early stopping triggered: no {stopping_metric} improvement for "
                            f"{generations_without_improvement} generations "
                            f"(best {stopping_metric}: {best_value:.6f})"
                        )
                    break

        _relax_unrelaxed_candidates(
            da,
            relaxer,
            population=population,
            max_batch=batch_size,
            force=True,
            run_id=run_id,
            surface_config=surface_config,
            n_slab=n_slab,
            system_type=system_type,
            profiling=profile_timings,
            counters=profile_counters,
            composition=composition,
            adsorbate_definition=adsorbate_definition,
            connectivity_factor=connectivity_factor,
            allow_cluster_fragmentation=allow_cluster_fragmentation,
            allow_adsorbate_surface_detachment=allow_adsorbate_surface_detachment,
            enforce_adsorbate_subgraph_integrity=enforce_adsorbate_subgraph_integrity,
            freeze_adsorbate_internal_geometry=freeze_adsorbate_internal_geometry,
            adsorbate_fragment_templates=adsorbate_fragment_template,
        )

        all_candidates = database_retry(
            da.get_all_relaxed_candidates,
            config=RetryConfig(max_retries=5),
            operation_name="get_final_all_relaxed_candidates",
        )
        if run_id is not None:
            all_candidates = filter_by_metadata(all_candidates, run_id=run_id)
        all_candidates = [
            cand
            for cand in all_candidates
            if bool(get_metadata(cand, "ga_eligible", default=True))
        ]
        all_minima = extract_minima_from_database(all_candidates)

        if verbosity >= 1:
            logger.info(
                f"GA evolution complete. Found {len(all_minima)} unique minima."
            )

        # Sort by fitness (highest first) for non-default strategies
        sort_minima_by_fitness(
            all_minima=all_minima,
            fitness_strategy=fitness_strategy,
            logger=logger,
        )
        profile_timings["total_wall_s"] = perf_counter() - profile_t0
        relax_total = ga_relax_seconds_from_timings(profile_timings)
        profile_timings["relax_total_s"] = relax_total
        profile_timings["cpu_non_relax_s"] = cpu_non_relax_seconds_from_timings(
            profile_timings
        )
        log_timing_summary(logger, "torchsim_ga", profile_timings, verbosity=verbosity)
        out_payload: dict[str, Any] = {
            "backend": "torchsim_ga",
            "timings_s": profile_timings,
            "counters": profile_counters,
            "retry_failures": profile_retry_failures,
        }
        if per_generation is not None:
            out_payload["per_generation"] = per_generation
        if timing_collector is not None:
            timing_collector.append(out_payload)
        if write_timing_json:
            if timing_output_dir is not None:
                write_timing_file(timing_output_dir, out_payload)
            elif timing_collector is None:
                write_timing_file(output_dir, out_payload)

        return all_minima

    finally:
        close_data_connection(da, log_errors=False)
