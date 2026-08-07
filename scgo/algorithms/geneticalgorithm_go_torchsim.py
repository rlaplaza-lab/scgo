"""TorchSim-enhanced Genetic Algorithm global optimization for clusters.

Genetic Algorithm global optimization with batched relaxations (TorchSim for MLIPs,
ASE sequential batch relaxer for classical calculators). Database interaction
remains single-threaded to protect against SQLite locking issues.
"""

from __future__ import annotations

import copy
import functools
import math
import os
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import suppress
from dataclasses import dataclass
from time import perf_counter
from typing import Any, cast

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.optimize import FIRE
from ase.optimize.optimize import Optimizer
from ase_ga.data import DataConnection
from ase_ga.utilities import get_all_atom_types
from scipy.spatial.distance import cdist
from tqdm import tqdm

from scgo.algorithms.ga_common import (
    ClusterStartGenerator,
    SurfaceClusterStartGenerator,
    SurfaceSlabStartGenerator,
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
    validate_structure_for_ga_storage,
)
from scgo.algorithms.run_context import validate_and_resolve_run_context
from scgo.ase_ga_patches.cutandsplicepairing import (
    CutAndSplicePairing,
    DualCutAndSplicePairing,
)
from scgo.ase_ga_patches.population import Population
from scgo.calculators.ase_batch_relaxer import AseBatchRelaxer
from scgo.calculators.torchsim_helpers import (
    TorchSimBatchRelaxer,
    build_torchsim_relaxer,
)
from scgo.cluster_adsorbate.config import ClusterAdsorbateConfig
from scgo.cluster_adsorbate.constraints import (
    attach_adsorbate_internal_geometry_constraints,
)
from scgo.cluster_adsorbate.rigid import enforce_frozen_adsorbate_geometry
from scgo.constants import DEFAULT_ENERGY_TOLERANCE
from scgo.database import (
    RetryConfig,
    close_data_connection,
    database_retry,
    setup_database,
)
from scgo.exceptions import SCGORuntimeError, SCGOValidationError
from scgo.initialization import compute_cell_side
from scgo.initialization.atomic_radii import build_blmin_from_zs
from scgo.initialization.geometry_helpers import reorder_cluster_to_composition
from scgo.initialization.initialization_config import BLMIN_RATIO_DEFAULT
from scgo.initialization.steric_scoring import build_blmin_threshold_matrix
from scgo.metadata.atoms import filter_by_tags, get_tag, get_tags, set_tags
from scgo.surface.config import SurfaceSystemConfig
from scgo.surface.constraints import attach_slab_constraints
from scgo.system_types import (
    AdsorbateDefinition,
    AdsorbateFragmentInput,
    SystemType,
    resolve_search_mobile_composition,
    resolve_structure_mic,
    uses_surface,
    validate_structure_for_system_type,
)
from scgo.utils.fitness_strategies import (
    FitnessStrategy,
)
from scgo.utils.helpers import (
    extract_minima_from_database,
)
from scgo.utils.logging import (
    get_logger,
    log_debug_v,
    log_exception_v,
    log_info_v,
    should_show_progress,
)
from scgo.utils.mutation_weights import get_adaptive_mutation_config
from scgo.utils.parallel_workers import resolve_n_jobs_to_workers
from scgo.utils.phase_logging import (
    log_generation_offspring_summaries,
    log_phase_subheader,
)
from scgo.utils.rng_helpers import (
    create_child_rng,
    ensure_rng_or_create,
    offspring_rng_triple,
)
from scgo.utils.timing_report import (
    build_timing_payload,
    cpu_non_relax_seconds_from_timings,
    ga_relax_seconds_from_timings,
    log_timing_summary,
    write_timing_file,
)
from scgo.utils.torchsim_policy import (
    is_ml_calculator,
)
from scgo.utils.validation import validate_composition


def _resolve_parallel_worker_count(n_jobs: int, n_tasks: int) -> int:
    """Resolve worker count from initialization-style semantics."""
    if n_tasks <= 1:
        return 1
    return min(resolve_n_jobs_to_workers(n_jobs), n_tasks)


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


_PREFILTER_BLMIN_FACTOR = 0.55

_MAX_FINAL_FLUSH_BATCHES = 1000
"""Safety bound on the number of batches used to drain the final relax backlog."""

BatchRelaxer = TorchSimBatchRelaxer | AseBatchRelaxer
"""Either batched relaxer backend accepted by the GA relaxation helpers."""


def _resolve_relax_batch_target(
    relax_batch_target: int | str | None,
    *,
    population_size: int,
    n_offspring: int,
    batch_size: int | None,
) -> int:
    """Resolve the per-call relaxation batch target for the generation loop.

    ``"auto"`` accumulates a full population's worth of offspring before calling
    the relaxer, which keeps MLIP/GPU batches large instead of relaxing the
    handful of offspring produced by a single generation. ``None`` or ``0``
    restores the legacy one-batch-per-generation behavior.

    Args:
        relax_batch_target: ``"auto"``, a positive integer, ``None`` or ``0``.
        population_size: GA population size (the ``"auto"`` target).
        n_offspring: Offspring created per generation (the legacy target).
        batch_size: Optional hard cap on structures per relax call.

    Returns:
        Positive number of structures to accumulate (and cap) per relax call.
    """
    if isinstance(relax_batch_target, str):
        if relax_batch_target.strip().lower() != "auto":
            raise SCGOValidationError(
                f"relax_batch_target must be 'auto', None, or a positive int, "
                f"got {relax_batch_target!r}"
            )
        target = int(population_size)
    elif relax_batch_target is None or int(relax_batch_target) <= 0:
        # Legacy behavior: one relax call per generation.
        target = int(batch_size) if batch_size is not None else int(n_offspring)
    else:
        target = int(relax_batch_target)

    if batch_size is not None:
        target = min(target, int(batch_size))
    return max(1, target)


def _blmin_threshold_matrix(
    atomic_numbers: np.ndarray, blmin: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Map atomic numbers to a dense Z-pair clash-threshold matrix."""
    return build_blmin_threshold_matrix(
        atomic_numbers,
        blmin,
        factor=_PREFILTER_BLMIN_FACTOR,
        default=0.0,
    )


def _fails_fast_geometric_prefilter(
    atoms: Atoms, blmin: dict, *, n_slab: int = 0
) -> bool:
    """Return True when a severe clash is detected quickly.

    Only mobile atoms (indices ``n_slab:``) participate: mobile–mobile and
    mobile–slab pairs are checked; slab–slab pairs are skipped.
    """
    n_atoms = len(atoms)
    if n_atoms < 2:
        return False
    n_slab_i = max(0, min(int(n_slab), n_atoms))
    n_mobile = n_atoms - n_slab_i
    if n_mobile < 1:
        return False

    numbers = atoms.get_atomic_numbers()
    positions = atoms.get_positions()
    thresh, z_index = _blmin_threshold_matrix(numbers, blmin)
    mobile_pos = positions[n_slab_i:]
    mobile_idx = z_index[n_slab_i:]

    # Mobile–mobile pairs (upper triangle).
    if n_mobile >= 2:
        mm = cdist(mobile_pos, mobile_pos)
        pair_thresh = thresh[np.ix_(mobile_idx, mobile_idx)]
        upper = np.triu(np.ones((n_mobile, n_mobile), dtype=bool), k=1)
        if np.any(upper & (pair_thresh > 0.0) & (mm < pair_thresh)):
            return True

    # Mobile–slab pairs.
    if n_slab_i > 0:
        slab_pos = positions[:n_slab_i]
        slab_idx = z_index[:n_slab_i]
        ms = cdist(mobile_pos, slab_pos)
        pair_thresh = thresh[np.ix_(mobile_idx, slab_idx)]
        if np.any((pair_thresh > 0.0) & (ms < pair_thresh)):
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
    operators_epoch: int


_OFFSPRING_WORKER_STATE: dict[str, Any] = {}


def _reseed_pairing_rng(pairing: Any, rng: np.random.Generator) -> None:
    if isinstance(pairing, DualCutAndSplicePairing):
        pairing.rng = create_child_rng(rng)
        pairing.primary.rng = create_child_rng(rng)
        pairing.exploratory.rng = create_child_rng(rng)
        return
    if isinstance(pairing, CutAndSplicePairing):
        pairing.rng = create_child_rng(rng)


def _load_offspring_worker_state(ctx: OffspringBuildContext) -> None:
    """Build pairing and operators once per worker process / generation."""
    placeholder_rng = np.random.default_rng(0)
    pairing = create_ga_pairing(
        ctx.atoms_template,
        ctx.n_to_optimize,
        placeholder_rng,
        slab_atoms=ctx.slab_for_pairing,
        system_type=ctx.system_type,
        composition=ctx.composition,
        adsorbate_definition=ctx.adsorbate_definition,
    )
    _OFFSPRING_WORKER_STATE["operators_epoch"] = ctx.operators_epoch
    _OFFSPRING_WORKER_STATE["pairing"] = pairing
    _OFFSPRING_WORKER_STATE["operators"] = copy.deepcopy(ctx.operators_list)


def _offspring_worker_bootstrap_init(ctx: OffspringBuildContext) -> None:
    _offspring_worker_init()
    _load_offspring_worker_state(ctx)


def _ensure_offspring_worker_state(ctx: OffspringBuildContext) -> None:
    if _OFFSPRING_WORKER_STATE.get("operators_epoch") != ctx.operators_epoch:
        _load_offspring_worker_state(ctx)


def _offspring_worker_has_cached_state(ctx: OffspringBuildContext) -> bool:
    return (
        _OFFSPRING_WORKER_STATE.get("operators_epoch") == ctx.operators_epoch
        and "pairing" in _OFFSPRING_WORKER_STATE
    )


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
    pairing_rng, operator_rng, decision_rng = offspring_rng_triple(job["task_seed"])
    setup_t0 = perf_counter()
    if _offspring_worker_has_cached_state(ctx):
        local_pairing = _OFFSPRING_WORKER_STATE["pairing"]
        _reseed_pairing_rng(local_pairing, pairing_rng)
        local_ops = _OFFSPRING_WORKER_STATE["operators"]
        reseed_mutation_operator_rngs(local_ops, operator_rng)
    else:
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
    mutation_applied = False
    if child is None:
        return {
            "index": job["index"],
            "child": None,
            "desc": None,
            "failure_reason": "pairing_failed",
            "mutation_applied": False,
            "operator_setup_s": operator_setup_s,
            "crossover_s": crossover_s,
            "mutation_s": mutation_s,
        }
    if _fails_fast_geometric_prefilter(child, ctx.blmin, n_slab=ctx.n_slab):
        return {
            "index": job["index"],
            "child": None,
            "desc": desc,
            "failure_reason": "too_close_prefilter",
            "mutation_applied": False,
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
            mutation_applied = True
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
        # Pre-relax geometric screen (raw frame); eligibility is decided post-relax
        # via validate_structure_for_ga_storage after canonicalization.
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
    except (ValueError, SCGOValidationError) as exc:
        return {
            "index": job["index"],
            "child": None,
            "desc": desc,
            "failure_reason": "validation_failed",
            "validation_error": str(exc),
            "mutation_applied": mutation_applied,
            "operator_setup_s": operator_setup_s,
            "crossover_s": crossover_s,
            "mutation_s": mutation_s,
        }
    return {
        "index": job["index"],
        "child": child,
        "desc": desc,
        "failure_reason": None,
        "mutation_applied": mutation_applied,
        "operator_setup_s": operator_setup_s,
        "crossover_s": crossover_s,
        "mutation_s": mutation_s,
    }


def _torchsim_prepare_relaxed_copy(
    cand: Atoms,
    surface_config: SurfaceSystemConfig | None,
    n_slab: int,
    *,
    surface_mode: bool,
    freeze_adsorbate_internal_geometry: bool = False,
    adsorbate_definition: AdsorbateDefinition | None = None,
    adsorbate_fragment_templates: AdsorbateFragmentInput | None = None,
) -> Atoms:
    """Copy candidate and attach slab constraints before TorchSim relaxation."""
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
    relaxer: BatchRelaxer,
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


def _write_relaxed_candidate(
    da: DataConnection,
    original: Atoms,
    relaxed: Atoms,
    energy: float,
    *,
    n_slab: int,
    composition: list[str] | None,
    adsorbate_definition: AdsorbateDefinition | None,
    system_type: SystemType,
    surface_mode: bool,
    surface_config: SurfaceSystemConfig | None,
    connectivity_factor: float | None,
    allow_cluster_fragmentation: bool,
    allow_adsorbate_surface_detachment: bool,
    enforce_adsorbate_subgraph_integrity: bool,
    generation: int | None = None,
    run_id: str | None = None,
) -> str | None:
    """Write a single relaxed candidate to the database.

    Returns the validation error string when the structure is disconnected,
    or ``None`` when it is eligible for GA evolution.
    """
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
    validation_error = validate_structure_for_ga_storage(
        original,
        surface_mode=surface_mode,
        n_slab=n_slab,
        system_type=system_type,
        surface_config=surface_config,
        adsorbate_definition=adsorbate_definition,
        connectivity_factor=connectivity_factor,
        allow_cluster_fragmentation=allow_cluster_fragmentation,
        allow_adsorbate_surface_detachment=allow_adsorbate_surface_detachment,
        enforce_adsorbate_subgraph_integrity=enforce_adsorbate_subgraph_integrity,
    )

    if "forces" in relaxed.arrays:
        original.arrays["forces"] = relaxed.arrays["forces"].copy()

    set_tags(
        original,
        **(get_tags(relaxed) or {"potential_energy": energy, "raw_score": -energy}),
    )
    set_tags(
        original,
        ga_eligible=(validation_error is None),
    )
    if validation_error is not None:
        set_tags(
            original,
            ga_ineligible_reason=validation_error,
        )

    comp_meta = list(composition) if composition is not None else []
    extra = ga_run_metadata_extras(
        surface_config,
        n_slab,
        system_type,
        comp_meta,
        adsorbate_definition=adsorbate_definition,
    )
    if generation is not None:
        set_tags(
            original,
            generation=generation,
            run_id=run_id,
            **extra,
        )
    elif run_id is not None:
        set_tags(original, run_id=run_id, **extra)

    original.calc = SinglePointCalculator(original, energy=energy)
    da.add_relaxed_step(original)
    return validation_error


def _relax_unrelaxed_candidates(
    da: DataConnection,
    relaxer: BatchRelaxer,
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
) -> tuple[int, int]:
    """Relax unrelaxed candidates in batches and commit them to the database.

    ``max_batch`` is a hard cap on the number of structures relaxed in this call
    and (unless ``force``) also the minimum backlog required before relaxing at
    all: accumulating up to a large ``max_batch`` is what keeps MLIP relaxations
    efficient on GPU. ``None`` means "no cap, relax whatever is available".
    ``force`` bypasses the accumulation gate only; the cap still applies, so
    draining a large backlog needs repeated calls.

    Returns:
        Tuple of (GA-eligible count, ineligible count) for this relax call.
    """
    available = database_retry(
        da.get_number_of_unrelaxed_candidates,
        config=RetryConfig(max_retries=5),
        operation_name="get_unrelaxed_candidates_count",
    )

    if available == 0:
        return (0, 0)
    if not force and max_batch is not None and available < max_batch:
        return (0, 0)

    to_take = available if max_batch is None else min(available, max_batch)

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
        return (0, 0)

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
        raise SCGORuntimeError("TorchSim relaxer returned mismatched batch size")

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
                validation_error = _write_relaxed_candidate(
                    da,
                    original,
                    relaxed,
                    energy,
                    n_slab=n_slab,
                    composition=composition,
                    adsorbate_definition=adsorbate_definition,
                    system_type=system_type,
                    surface_mode=surface_mode,
                    surface_config=surface_config,
                    connectivity_factor=connectivity_factor,
                    allow_cluster_fragmentation=allow_cluster_fragmentation,
                    allow_adsorbate_surface_detachment=allow_adsorbate_surface_detachment,
                    enforce_adsorbate_subgraph_integrity=enforce_adsorbate_subgraph_integrity,
                    generation=generation,
                    run_id=run_id,
                )
                if validation_error is not None:
                    ineligible_count += 1
                    label = (
                        "Offspring" if generation is not None else "Initial candidate"
                    )
                    logger.debug(
                        "%s %d/%d disconnected after relaxation; storing but excluding from GA population: %s",
                        label,
                        idx + 1,
                        len(batch),
                        validation_error,
                    )
                else:
                    successful_count += 1

    t0 = perf_counter()
    database_retry(
        _write_batch_under_connection,
        config=RetryConfig(max_retries=5),
        operation_name="write_relaxed_batch",
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

    return (successful_count, ineligible_count)


def ga_go(  # noqa: C901
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
    relaxer: BatchRelaxer | None = None,
    batch_size: int | None = None,
    relax_batch_target: int | str | None = "auto",
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
        batch_size: Hard cap on the number of structures passed to the relaxer in
            one call. ``None`` means no cap (the TorchSim autobatcher still
            splits by GPU memory).
        relax_batch_target: How many unrelaxed candidates to accumulate before
            calling the relaxer. ``"auto"`` (default) uses ``population_size``,
            so offspring from consecutive generations are relaxed together
            instead of one small batch per generation — MLIPs need large batches
            to amortize neighbor-list and kernel-launch overhead. Pass a positive
            integer for an explicit target, or ``None``/``0`` for the legacy
            one-relax-call-per-generation behavior. Capped by ``batch_size``
            when that is set. Any remaining backlog is always drained after the
            final generation.
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
        adsorbate_definition: Adsorbate partition/symbols for adsorbate-aware GA runs.
        adsorbate_fragment_template: Optional fragment geometry for hierarchical layout.
        aggressive_burst_multiplier: Burst-size multiplier applied during stagnation bursts.
        allow_adsorbate_surface_detachment: Allow adsorbate to detach from the surface.
        allow_cluster_fragmentation: Allow the core cluster to fragment during mutation.
        cluster_adsorbate_config: Optional placement/validation config for the fragment.
        connectivity_factor: Optional connectivity radius factor for adsorbate integrity.
        db_enable_expression_indexes: Enable SQLite expression indexes on the results DB.
        energy_tolerance: Energy difference (eV) below which structures are duplicates.
        enforce_adsorbate_subgraph_integrity: Keep adsorbate subgraph connectivity intact.
        fmax: Maximum force criterion for convergence in local relaxations (eV/Å).
        freeze_adsorbate_internal_geometry: Freeze internal adsorbate geometry during relaxation.
        ga_adaptive_retry_enabled: Enable adaptive retry of failed relaxation attempts.
        ga_fast_prefilter_enabled: Enable fast prefiltering of candidate structures.
        ga_retry_ceiling_multiplier: Upper bound multiplier for adaptive retry ceiling.
        ga_retry_floor_multiplier: Lower bound multiplier for adaptive retry floor.
        max_mutation_probability: Upper cap on the adaptive mutation probability.
        mutation_probability: Base probability of applying a mutation operator.
        n_jobs_offspring: Number of parallel workers for offspring relaxation.
        n_jobs_population_init: Number of parallel workers for population initialization.
        niter: Total number of GA generations to run.
        niter_local_relaxation: Maximum steps allowed for each local relaxation.
        offspring_fraction: Fraction of the population generated as offspring each generation.
        optimizer: ASE optimizer class for local relaxations.
        output_dir: Directory where the ASE results database is stored.
        population_size: Number of candidate structures in the population.
        recovery_window: Generations over which to attempt stalled-run recovery.
        relaxer: Optional :class:`TorchSimBatchRelaxer` (or
            :class:`~scgo.calculators.ase_batch_relaxer.AseBatchRelaxer`)
            controlling batched relaxations.
        rng: Optional numpy random number generator for reproducibility.
        stagnation_full_trigger: Generations of stagnation before a full burst is triggered.
        stagnation_trigger: Generations of stagnation before a burst is triggered.
        system_type: ``gas_cluster`` or ``gas_cluster_adsorbate``; selects the run policy.
        use_adaptive_mutations: Enable adaptive mutation probability scheduling.
        vacuum: Amount of vacuum to add around the cluster.
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

    run_ctx = validate_and_resolve_run_context(
        system_type=system_type,
        surface_config=surface_config,
        connectivity_factor=connectivity_factor,
        cluster_adsorbate_config=cluster_adsorbate_config,
        fitness_strategy=fitness_strategy,
    )
    connectivity_factor = run_ctx.connectivity_factor
    policy = run_ctx.policy
    fitness_strategy = run_ctx.fitness_strategy
    # Bare ``surface`` uses an empty cluster composition; search-mobile symbols
    # come from the top slab layers via ``resolve_search_mobile_composition``.
    validate_composition(
        composition,
        allow_empty=policy.slab_is_search_target and not policy.has_adsorbate,
        allow_tuple=False,
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
        raise SCGOValidationError(
            f"n_jobs_offspring must be -1, -2, or >= 1, got {n_jobs_offspring}"
        )

    if batch_size is not None and batch_size <= 0:
        batch_size = None

    # Resolved once: the accumulation target is constant for the whole run.
    relax_batch_target_resolved = _resolve_relax_batch_target(
        relax_batch_target,
        population_size=population_size,
        n_offspring=max(1, math.ceil(population_size * offspring_fraction)),
        batch_size=batch_size,
    )
    relax_flush_batch = relax_batch_target_resolved

    # Normalize RNG early and enforce Generator-only policy
    rng = ensure_rng_or_create(rng)
    surface_mode = uses_surface(system_type)
    n_fixed = 0
    search_composition = list(composition)
    deposit_composition = list(composition)
    n_mobile_slab = 0

    if surface_mode:
        if not isinstance(surface_config, SurfaceSystemConfig):
            raise SCGOValidationError(
                "surface_config must be a SurfaceSystemConfig instance or None"
            )
        if policy.slab_is_search_target:
            from scgo.surface.partition import prepare_slab_search_surface_config

            surface_config, partition = prepare_slab_search_surface_config(
                surface_config
            )
            n_fixed = partition.n_fixed
            n_mobile_slab = partition.n_mobile_slab
            search_composition = resolve_search_mobile_composition(
                system_type=system_type,
                composition=list(composition),
                surface_config=surface_config,
                adsorbate_definition=adsorbate_definition,
            )
            if policy.has_adsorbate:
                ads = (
                    adsorbate_definition.get("adsorbate_symbols", [])
                    if adsorbate_definition
                    else []
                )
                deposit_composition = (
                    [str(s) for s in ads] if isinstance(ads, list) else []
                )
            else:
                deposit_composition = []
        slab_ref = surface_config.slab.copy()
        n_slab = len(slab_ref)
        if not policy.slab_is_search_target:
            n_fixed = n_slab
            search_composition = list(composition)
            deposit_composition = list(composition)
        n_to_optimize = len(search_composition)
        if n_to_optimize < 1:
            raise SCGOValidationError(
                f"system_type={system_type!r} has no search-mobile atoms."
            )
        if policy.slab_is_search_target and not policy.has_adsorbate:
            atoms_template = slab_ref.copy()
        elif policy.slab_is_search_target:
            ads_syms = list(search_composition[n_mobile_slab:])
            if ads_syms:
                dummy_top = [[0.0, 0.0, 0.0] for _ in range(len(ads_syms))]
                atoms_template = Atoms(
                    symbols=list(slab_ref.get_chemical_symbols()) + ads_syms,
                    positions=np.vstack(
                        [slab_ref.get_positions(), np.asarray(dummy_top)]
                    ),
                    cell=slab_ref.get_cell(),
                    pbc=slab_ref.get_pbc(),
                )
            else:
                atoms_template = slab_ref.copy()
        else:
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
        n_to_optimize = len(composition)
        search_composition = list(composition)
        cell_side = compute_cell_side(composition, vacuum=vacuum)
        atoms_template = Atoms(
            symbols=composition,
            positions=[[0, 0, 0] for _ in range(n_to_optimize)],  # Dummy positions
            cell=[cell_side] * 3,
            pbc=False,
        )

    pop_for_probe = population_size if population_size is not None else 32
    expected_max_atoms = (n_to_optimize + n_fixed) * pop_for_probe

    if relaxer is None:
        if is_ml_calculator(calculator):
            relaxer = build_torchsim_relaxer(
                calculator,
                fmax=fmax,
                max_steps=niter_local_relaxation,
                expected_max_atoms=expected_max_atoms,
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
    assert relaxer is not None  # built above when not supplied by the caller

    if isinstance(relaxer, TorchSimBatchRelaxer):
        atoms_template.calc = None
    else:
        atoms_template.calc = calculator

    # Diversity scorer needs surface-aware mic; set up after operators / comp_mic.
    if surface_mode and slab_ref is not None:
        slab_for_pairing = slab_ref[:n_fixed].copy() if n_fixed > 0 else slab_ref.copy()
    else:
        slab_for_pairing = None

    adaptive_config = get_adaptive_mutation_config(
        composition=search_composition,
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
        range(n_fixed, n_fixed + n_to_optimize)
        if surface_mode
        else range(n_to_optimize)
    )
    top_z = list({int(atoms_template[i].number) for i in idx_top})
    all_atom_types = get_all_atom_types(atoms_template, top_z)
    blmin = build_blmin_from_zs(all_atom_types, ratio=BLMIN_RATIO_DEFAULT)

    operators_list, name_map = create_mutation_operators(
        composition=search_composition,
        n_to_optimize=n_to_optimize,
        blmin=blmin,
        rng=rng,
        use_adaptive=use_adaptive_mutations,
        system_type=system_type,
        n_slab=n_fixed if policy.slab_is_search_target else n_slab,
        surface_normal_axis=(
            cast("SurfaceSystemConfig", surface_config).surface_normal_axis
            if surface_mode
            else 2
        ),
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

    comp_mic = resolve_structure_mic(system_type, surface_config)
    diversity_scorer = setup_diversity_scorer(
        fitness_strategy=fitness_strategy,
        diversity_reference_db=diversity_reference_db,
        composition=search_composition,
        n_to_optimize=n_to_optimize,
        diversity_max_references=diversity_max_references,
        logger=logger,
        base_dir=output_dir,
        mic=comp_mic,
    )
    comp = create_structure_comparator(n_to_optimize, energy_tolerance, mic=comp_mic)

    t0_batch_build = perf_counter()
    if surface_mode:
        assert slab_ref is not None
        assert surface_config is not None  # validated in the surface_mode block above
        if policy.slab_is_search_target and not policy.has_adsorbate:
            start_generator = SurfaceSlabStartGenerator(
                slab_ref,
                n_fixed=n_fixed,
                rng=rng,
                calculator=None,
                population_size=population_size,
                verbosity=verbosity,
            )
        else:
            start_generator = SurfaceClusterStartGenerator(
                deposit_composition,
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
                verbosity=verbosity,
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
            verbosity=verbosity,
        )
    profile_timings["initial_population_batch_build_s"] = (
        perf_counter() - t0_batch_build
    )
    t0 = perf_counter()
    initial_population = [
        start_generator.get_new_candidate() for _ in range(population_size)
    ]
    profile_timings["initial_population_generation_s"] = perf_counter() - t0

    n_workers_desc = (
        "all CPUs"
        if n_jobs_population_init == -1
        else "all but one CPU"
        if n_jobs_population_init == -2
        else f"{n_jobs_population_init} workers"
    )
    log_info_v(
        logger,
        "Generated initial population of %d candidates (batched, parallel n_jobs=%s)",
        population_size,
        n_workers_desc,
        verbosity=verbosity,
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
        log_info_v(
            logger,
            "Relaxing initial population of %d candidates...",
            population_size,
            verbosity=verbosity,
        )

        logger.debug(
            "Using GA database at %s",
            os.path.join(output_dir, "ga_go.db"),
        )

        initial_pop_count = 0
        initial_discarded_count = 0
        initial_ineligible_relaxed_count = 0
        inserted_initial_population: list[Atoms] = []

        def _insert_unrelaxed(cand):
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
                validation_error = validate_structure_for_ga_storage(
                    cand,
                    surface_mode=surface_mode,
                    n_slab=n_slab,
                    system_type=system_type,
                    surface_config=surface_config,
                    adsorbate_definition=adsorbate_definition,
                    connectivity_factor=connectivity_factor,
                    allow_cluster_fragmentation=allow_cluster_fragmentation,
                    allow_adsorbate_surface_detachment=allow_adsorbate_surface_detachment,
                    enforce_adsorbate_subgraph_integrity=enforce_adsorbate_subgraph_integrity,
                )
                if validation_error is not None:
                    initial_discarded_count += 1
                    logger.debug(
                        "Discarding disconnected initial candidate before DB insert: %s",
                        validation_error,
                    )
                    continue
                database_retry(
                    functools.partial(_insert_unrelaxed, cand),
                    config=RetryConfig(max_retries=5),
                    operation_name="insert_unrelaxed_candidate",
                )
                inserted_initial_population.append(cand)
        profile_timings["initial_unrelaxed_insert_s"] = perf_counter() - t0

        if not inserted_initial_population:
            logger.error(
                "No valid initial GA population after validation (%d discarded)",
                initial_discarded_count,
            )
            return []

        # Helper to write a relaxed batch into the database under a single connection
        def _write_relaxed_batch(batch, relaxed_results):
            nonlocal initial_ineligible_relaxed_count
            with da.c:
                for original, (energy, relaxed) in zip(
                    batch, relaxed_results, strict=True
                ):
                    validation_error = _write_relaxed_candidate(
                        da,
                        original,
                        relaxed,
                        energy,
                        n_slab=n_slab,
                        composition=composition,
                        adsorbate_definition=adsorbate_definition,
                        system_type=system_type,
                        surface_mode=surface_mode,
                        surface_config=surface_config,
                        connectivity_factor=connectivity_factor,
                        allow_cluster_fragmentation=allow_cluster_fragmentation,
                        allow_adsorbate_surface_detachment=allow_adsorbate_surface_detachment,
                        enforce_adsorbate_subgraph_integrity=enforce_adsorbate_subgraph_integrity,
                        generation=0,
                        run_id=run_id,
                    )
                    if validation_error is not None:
                        initial_ineligible_relaxed_count += 1
                        logger.debug(
                            "Initial candidate disconnected after relaxation; storing but excluding from GA population: %s",
                            validation_error,
                        )

        # Process starting population in batches (only candidates inserted above).
        batch_size_internal = batch_size or len(inserted_initial_population)
        t0_relax = 0.0
        t0_write = 0.0
        for i in range(0, len(inserted_initial_population), batch_size_internal):
            batch = inserted_initial_population[i : i + batch_size_internal]
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
                        adsorbate_fragment_templates=adsorbate_fragment_template,
                    )
                    for c in batch
                ]
            )
            t0_relax += perf_counter() - t_start
            _record_relax_batch_steps(
                relaxer, profile_timings, profile_counters, len(batch)
            )
            if len(relaxed_results) != len(batch):
                raise SCGORuntimeError(
                    "TorchSim relaxer returned mismatched batch size"
                )

            t_start = perf_counter()
            database_retry(
                functools.partial(_write_relaxed_batch, batch, relaxed_results),
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
            rng=rng,
            elite_fraction=elite_fraction,
            run_id=run_id,
            **population_kwargs,
        )
        population._write_log()
        eligible_initial = initial_pop_count - initial_ineligible_relaxed_count
        log_info_v(
            logger,
            "Initial population: size=%d, %d GA-eligible, %d discarded pre-relax, %d ineligible post-relax",
            len(population.pop),
            eligible_initial,
            initial_discarded_count,
            initial_ineligible_relaxed_count,
            verbosity=verbosity,
        )
        log_debug_v(
            logger,
            "Initial Population confids=%s",
            [a.info.get("confid") for a in population.pop],
            verbosity=verbosity,
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
        generations_since_relax = 0

        for generation in tqdm(
            range(niter),
            desc=f"  GA generations for {n_to_optimize} mobile atoms",
            disable=not should_show_progress(verbosity),
        ):
            if use_adaptive_mutations:
                adaptive_config = get_adaptive_mutation_config(
                    composition=search_composition,
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
            generation_all_job_results: list[Mapping[str, Any]] = []
            total_crossover_jobs_gen = 0

            log_phase_subheader(
                logger,
                f"Generation {generation}",
                verbosity=verbosity,
            )

            offspring_ctx = OffspringBuildContext(
                atoms_template=cast("Atoms", _picklable_atoms_copy(atoms_template)),
                n_to_optimize=n_to_optimize,
                composition=composition,
                blmin=blmin if ga_fast_prefilter_enabled else {},
                system_type=system_type,
                n_slab=n_slab,
                slab_for_pairing=_picklable_atoms_copy(slab_for_pairing),
                surface_normal_axis=(
                    cast("SurfaceSystemConfig", surface_config).surface_normal_axis
                    if surface_mode
                    else 2
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
                operators_list=operators_list,
                name_map=name_map,
                operators_epoch=generation,
            )
            n_workers_offspring = _resolve_parallel_worker_count(
                n_jobs_offspring, max(1, n_offspring)
            )
            offspring_executor: ProcessPoolExecutor | None = None
            if n_workers_offspring > 1:
                offspring_executor = ProcessPoolExecutor(
                    max_workers=n_workers_offspring,
                    initializer=_offspring_worker_bootstrap_init,
                    initargs=(offspring_ctx,),
                )
            else:
                _ensure_offspring_worker_state(offspring_ctx)

            try:
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

                    n_workers = _resolve_parallel_worker_count(
                        n_jobs_offspring, len(jobs)
                    )

                    t_parallel = perf_counter()
                    job_results: dict[int, dict[str, Any]] = {}
                    worker_exceptions: list[BaseException] = []
                    if n_workers == 1:
                        for job in jobs:
                            try:
                                result = _build_offspring_worker(
                                    job,
                                    offspring_ctx,
                                )
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
                                log_exception_v(
                                    logger,
                                    "Offspring crossover/mutation worker failed (%s)",
                                    err_name,
                                    verbosity=verbosity,
                                )
                                continue
                            job_results[result["index"]] = result
                    else:
                        assert offspring_executor is not None
                        futures = [
                            offspring_executor.submit(
                                _build_offspring_worker, job, offspring_ctx
                            )
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
                                log_exception_v(
                                    logger,
                                    "Offspring crossover/mutation worker failed (%s)",
                                    err_name,
                                    verbosity=verbosity,
                                )
                                continue
                            job_results[result["index"]] = result
                    total_crossover_jobs_gen += len(job_results)
                    generation_all_job_results.extend(job_results.values())
                    if len(jobs) > 0 and len(job_results) == 0 and worker_exceptions:
                        first = worker_exceptions[0]
                        if not all(
                            isinstance(e, ValueError) for e in worker_exceptions
                        ):
                            raise SCGORuntimeError(
                                f"All {len(jobs)} parallel offspring workers failed"
                            ) from first
                    t_offspring_parallel_wall_gen += perf_counter() - t_parallel
                    if worker_failures_gen:
                        profile_counters["offspring_worker_failures"] += (
                            worker_failures_gen
                        )
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
                        job_result = job_results.get(idx)
                        if job_result is None:
                            continue
                        t_operator_setup_gen += float(job_result["operator_setup_s"])
                        t_crossover_gen += float(job_result["crossover_s"])
                        t_mutation_gen += float(job_result["mutation_s"])
                        child = job_result["child"]
                        if child is None:
                            reason = job_result.get("failure_reason") or "unknown"
                            retry_failure_reasons_gen[reason] = (
                                retry_failure_reasons_gen.get(reason, 0) + 1
                            )
                            continue
                        pending_inserts.append((child, job_result["desc"]))
                    if pending_inserts:
                        t0 = perf_counter()
                        with da.c:
                            for child, desc in pending_inserts:
                                database_retry(
                                    functools.partial(
                                        da.add_unrelaxed_candidate,
                                        child,
                                        description=desc,
                                    ),
                                    config=RetryConfig(max_retries=5),
                                    operation_name="add_unrelaxed_offspring",
                                )
                                created += 1
                        t_db_unrelaxed_gen += perf_counter() - t0
            finally:
                if offspring_executor is not None:
                    offspring_executor.shutdown(wait=True)
                _OFFSPRING_WORKER_STATE.clear()

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

            log_generation_offspring_summaries(
                logger,
                verbosity=verbosity,
                job_results=generation_all_job_results,
                total_jobs=total_crossover_jobs_gen,
                created=created,
                n_offspring=n_offspring,
                attempts=attempts,
            )

            # Ask the relaxer to process available unrelaxed candidates now.
            # ``relax_batch_target`` accumulates offspring across generations so
            # each relax call is large enough to keep an MLIP/GPU busy; the
            # stale-flush guard prevents the population from starving when
            # offspring creation is slow.
            per_gen_max = relax_batch_target_resolved
            generations_since_relax += 1
            stale_flush = generations_since_relax >= max(
                2, math.ceil(per_gen_max / n_offspring) + 1
            )
            pre_db_read = float(profile_timings.get("db_read_s", 0.0))
            pre_relax = float(profile_timings.get("relax_batch_s", 0.0))
            pre_db_write = float(profile_timings.get("db_write_s", 0.0))
            pre_pop_update = float(profile_timings.get("population_update_s", 0.0))
            t0_relax_call = perf_counter()
            eligible_count, ineligible_count = _relax_unrelaxed_candidates(
                da,
                relaxer,
                population=population,
                max_batch=per_gen_max,
                force=stale_flush,
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
            if (eligible_count + ineligible_count) > 0:
                generations_since_relax = 0
            offspring_count = eligible_count
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
            if (eligible_count + ineligible_count) > 0:
                log_info_v(
                    logger,
                    "Relaxation: %d/%d GA-eligible, %d ineligible",
                    eligible_count,
                    eligible_count + ineligible_count,
                    ineligible_count,
                    verbosity=verbosity,
                )
            if offspring_count > 0:
                profile_counters["offspring_relaxed"] += int(offspring_count)

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
                    stopping_metric = (
                        "fitness"
                        if fitness_strategy != FitnessStrategy.LOW_ENERGY
                        else "energy"
                    )
                    log_info_v(
                        logger,
                        "Early stopping triggered: no %s improvement for %d generations (best %s: %.6f)",
                        stopping_metric,
                        generations_without_improvement,
                        stopping_metric,
                        best_value,
                        verbosity=verbosity,
                    )
                    break

        # Final flush: drain every remaining unrelaxed candidate. ``max_batch`` is
        # a hard cap, so repeat until the backlog is empty. Every taken candidate
        # is committed as relaxed, hence the loop always terminates; the guard
        # only protects against a pathological database state.
        drain_guard = 0
        while True:
            drained_eligible, drained_ineligible = _relax_unrelaxed_candidates(
                da,
                relaxer,
                population=population,
                max_batch=relax_flush_batch,
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
            if (drained_eligible + drained_ineligible) == 0:
                break
            log_info_v(
                logger,
                "Relaxation: %d/%d GA-eligible, %d ineligible",
                drained_eligible,
                drained_eligible + drained_ineligible,
                drained_ineligible,
                verbosity=verbosity,
            )
            drain_guard += 1
            if drain_guard > _MAX_FINAL_FLUSH_BATCHES:
                logger.warning(
                    "Final relaxation flush exceeded %d batches; "
                    "stopping to avoid an unbounded loop.",
                    _MAX_FINAL_FLUSH_BATCHES,
                )
                break

        all_candidates = database_retry(
            da.get_all_relaxed_candidates,
            config=RetryConfig(max_retries=5),
            operation_name="get_final_all_relaxed_candidates",
        )
        if run_id is not None:
            all_candidates = filter_by_tags(all_candidates, run_id=run_id)
        all_candidates = [
            cand
            for cand in all_candidates
            if bool(get_tag(cand, "ga_eligible", default=True))
        ]
        all_minima = extract_minima_from_database(all_candidates)

        log_info_v(
            logger,
            "GA evolution complete. Found %d unique minima.",
            len(all_minima),
            verbosity=verbosity,
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
        extra_payload: dict[str, Any] = {
            "counters": profile_counters,
            "retry_failures": profile_retry_failures,
        }
        if per_generation is not None:
            extra_payload["per_generation"] = per_generation
        timing_dir = timing_output_dir if timing_output_dir is not None else output_dir
        run_id_for_timing = os.path.basename(str(timing_dir).rstrip(os.sep))
        out_payload = build_timing_payload(
            backend="torchsim_ga",
            timings_s=profile_timings,
            run_id=run_id_for_timing,
            extra=extra_payload,
        )
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
