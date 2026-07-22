"""Global-optimization (GO) trial and campaign runners.

Implements algorithm selection and the low-level GO execution used by the
public ``run_go`` / ``run_go_campaign`` API in :mod:`scgo.runner_api`.

Note on the local ``scgo.runner_api`` imports inside :func:`_run_go_trials`
and :func:`_run_go_campaign_compositions`: ``scgo.runner_api`` re-exports
``run_trials`` and ``get_calculator_class`` (and, transitively, this module's
own ``_run_go_trials``) as its own module attributes specifically so tests can
``monkeypatch.setattr("scgo.runner_api.run_trials", ...)`` etc. Since
``scgo.runner_api`` imports from this module at top level, importing it back
here at module load time would be circular; the calls are therefore routed
through a function-local import so the patched attribute on
``scgo.runner_api`` is honored regardless of where the call originates.
"""

from __future__ import annotations

import copy
import os
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from ase import Atoms
from ase.calculators.calculator import Calculator

from scgo.exceptions import SCGOValidationError
from scgo.system_types import SystemType, get_system_policy
from scgo.utils.helpers import get_cluster_formula
from scgo.utils.logging import configure_logging, get_logger
from scgo.utils.output_paths import (
    resolve_go_campaign_searches_dir,
    resolve_go_searches_dir,
)
from scgo.utils.rng_helpers import ensure_rng
from scgo.utils.run_helpers import (
    cleanup_torch_cuda,
    initialize_params,
    log_configuration,
    prepare_algorithm_kwargs,
    validate_algorithm_params,
)
from scgo.utils.run_tracking import ensure_run_id
from scgo.utils.validation import validate_composition

ScgoMinimaAlgorithm = Literal["simple", "bh", "ga"]


def select_scgo_minima_algorithm(
    n_atoms: int, system_type: SystemType
) -> ScgoMinimaAlgorithm:
    """Select global optimizer for composition size and system type.

    Uses the mobile-atom count (core + adsorbate symbols for adsorbate modes).
    Plain ``gas_cluster`` alone may use ``simple`` for 1-2 atoms; adsorbate and
    surface modes never select ``simple``.
    """
    policy = get_system_policy(system_type)
    simple_allowed = not policy.uses_surface and not policy.has_adsorbate
    if n_atoms <= 2 and simple_allowed:
        return "simple"
    if n_atoms == 3:
        if policy.has_adsorbate:
            return "ga"
        return "bh"
    return "ga"


def _run_go_trials(
    composition: list[str],
    system_type: SystemType,
    params: dict | None = None,
    seed: int | None = None,
    verbosity: int = 1,
    run_id: str | None = None,
    clean: bool = False,
    output_dir: str | Path | None = None,
    calculator_for_global_optimization: Calculator | None = None,
    *,
    params_already_merged: bool = False,
) -> list[tuple[float, Atoms]]:
    """Run global optimization for a composition; return unique minima sorted by energy."""
    from scgo import runner_api as _runner_api

    configure_logging(verbosity)
    logger = get_logger(__name__)

    validate_composition(composition, allow_empty=False, allow_tuple=False)

    # Initialize and merge params with defaults
    if not params_already_merged:
        params = initialize_params(params)
    else:
        params = copy.deepcopy(params or {})

    # Validate calculator availability
    calculator_name = params.get("calculator", "MACE")
    _ = _runner_api.get_calculator_class(calculator_name)

    # Validate params structure - rng should not be in optimizer_params
    for algo in ["bh", "ga"]:
        algo_params = params["optimizer_params"].get(algo, {})
        if "rng" in algo_params:
            raise SCGOValidationError(
                f'"rng" should not be in params["optimizer_params"]["{algo}"]. '
                f'Use the "seed" parameter instead.'
            )

    # Prefer explicit function seed arg; fall back to params['seed'] if provided
    if seed is None:
        seed = params.get("seed", None)

    # Convert seed to generator at API boundary
    rng = ensure_rng(seed)

    n_atoms = len(composition)
    cluster_formula = get_cluster_formula(composition)
    main_output_dir = str(resolve_go_searches_dir(output_dir, cluster_formula))

    # Algorithm selection: Use simple optimization for 1-2 atoms, BH for 3, GA for larger
    chosen_go = select_scgo_minima_algorithm(n_atoms, system_type)
    if chosen_go == "simple":
        logger.info(
            f"Selected simple optimization for {n_atoms}-atom cluster (trivial structure)"
        )
    elif chosen_go == "bh":
        logger.info(
            "Selected Basin Hopping for %d-atom cluster (small cluster)", n_atoms
        )
    else:
        logger.info("Selected Genetic Algorithm for %d-atom cluster", n_atoms)

    # Extract algorithm-specific parameters without mutation
    algo_params = params["optimizer_params"].get(chosen_go, {})

    user_params = None if params_already_merged else params
    params_base = None if params_already_merged else _runner_api.get_default_params()

    # Validate algorithm-specific parameters
    validate_algorithm_params(algo_params, chosen_go, verbosity)

    # Get calculator kwargs if provided
    calculator_kwargs = params.get("calculator_kwargs", {})

    # Unified parameter preparation (resolves auto params, fitness strategy, diversity, etc.)
    global_optimizer_kwargs = prepare_algorithm_kwargs(
        algo_params=algo_params,
        params=params,
        composition=composition,
        chosen_go=chosen_go,
        system_type=system_type,
    )

    # Validate that no unexpected top-level keys were provided
    expected_top_level_keys = {
        "validate_with_hessian",
        "calculator",
        "calculator_kwargs",
        "surface_config",
        "fmax_threshold",
        "check_hessian",
        "imag_freq_threshold",
        "optimizer_params",
        "fitness_strategy",
        "diversity_reference_db",
        "diversity_max_references",
        "diversity_update_interval",
        "tag_final_minima",
        "connectivity_factor",
        "allow_cluster_fragmentation",
        "allow_adsorbate_surface_detachment",
        "enforce_adsorbate_subgraph_integrity",
        "freeze_adsorbate_internal_geometry",
        "adsorbate_definition",
        "adsorbate_fragment_template",
        "cluster_adsorbate_config",
        "validation_n_jobs",
        "seed",  # seed is handled separately at API boundary, not passed to algorithms
    }
    unexpected_keys = set(params.keys()) - expected_top_level_keys
    if unexpected_keys:
        raise SCGOValidationError(
            f"Unexpected parameter keys: {sorted(unexpected_keys)}. "
            f"Expected keys: {sorted(expected_top_level_keys)}"
        )

    # Log the final configuration being used
    log_configuration(
        params=params,
        chosen_go=chosen_go,
        cluster_formula=cluster_formula,
        n_atoms=n_atoms,
        global_optimizer_kwargs=global_optimizer_kwargs,
        verbosity=verbosity,
        user_params=user_params,
        params_base=params_base,
    )

    final_unique_minima = _runner_api.run_trials(
        composition=composition,
        global_optimizer=chosen_go,
        global_optimizer_kwargs=global_optimizer_kwargs,
        output_dir=main_output_dir,
        calculator_for_global_optimization=(
            calculator_for_global_optimization
            if calculator_for_global_optimization is not None
            else _runner_api.get_calculator_class(params["calculator"])(
                **calculator_kwargs
            )
        ),
        validate_with_hessian=params.get("validate_with_hessian", False),
        fmax_threshold=params.get("fmax_threshold", 0.05),
        check_hessian=params.get("check_hessian", True),
        imag_freq_threshold=params.get("imag_freq_threshold", 50.0),
        validation_n_jobs=params.get("validation_n_jobs", 1),
        tag_final_minima=params.get("tag_final_minima", True),
        rng=rng,
        run_id=run_id,
        clean=clean,
    )

    cleanup_torch_cuda(logger=logger)

    return final_unique_minima


def _run_go_campaign_compositions(
    compositions: Iterable[list[str]],
    system_type: SystemType,
    params: dict | None = None,
    seed: int | None = None,
    verbosity: int = 1,
    run_id: str | None = None,
    clean: bool = False,
    output_dir: str | Path | None = None,
    *,
    params_already_merged: bool = False,
) -> dict[str, list[tuple[float, Atoms]]]:
    """Run optimizations for an iterable of compositions; return mapping formula->minima."""
    from scgo import runner_api as _runner_api

    if params_already_merged:
        params = copy.deepcopy(params or {})
    else:
        params = initialize_params(params)
    configure_logging(verbosity)

    # Validate params structure early: 'rng' must not be present inside
    # optimizer-specific params. Raise ValueError so callers get immediate
    # feedback instead of having the error swallowed during campaign
    # iteration.
    for algo in ["bh", "ga"]:
        algo_params = params["optimizer_params"].get(algo, {})
        if "rng" in algo_params:
            raise SCGOValidationError(
                f'"rng" should not be in params["optimizer_params"]["{algo}"]. '
                f'Use the "seed" parameter instead.'
            )
    logger = get_logger(__name__)

    # Generate run_id once at campaign start if not provided
    run_id = ensure_run_id(run_id, verbosity=verbosity, logger=logger)

    # Prefer explicit function seed arg; fall back to params['seed'] if provided
    if seed is None:
        seed = params.get("seed", None)

    # Convert seed to generator at API boundary
    rng = ensure_rng(seed)

    all_results = {}
    compositions_list = list(compositions)
    if not compositions_list:
        raise SCGOValidationError("compositions iterable must not be empty")
    num_compositions = len(compositions_list)
    logger.info("Starting campaign for %d compositions.", num_compositions)

    # Create calculator once and reuse it for all compositions to avoid file handle leaks
    calculator_kwargs = params.get("calculator_kwargs", {})
    calculator_for_global_optimization = _runner_api.get_calculator_class(
        params["calculator"]
    )(
        **calculator_kwargs,
    )

    for i, composition in enumerate(compositions_list):
        formula_str = get_cluster_formula(composition)
        if verbosity >= 1:
            logger.info("\n%s", "=" * 60)
            logger.info(
                "Running minima search for %s (%d/%d)",
                formula_str,
                i + 1,
                num_compositions,
            )
            logger.info("%s", "=" * 60)

        comp_seed = int(rng.integers(0, 2**63 - 1))
        trial_output_dir = resolve_go_campaign_searches_dir(output_dir, formula_str)
        trial_output_dir_str = (
            str(trial_output_dir) if trial_output_dir is not None else None
        )

        try:
            results = _runner_api._run_go_trials(
                composition,
                system_type,
                params,
                seed=comp_seed,
                verbosity=verbosity,
                run_id=run_id,
                clean=clean,
                output_dir=trial_output_dir_str,
                calculator_for_global_optimization=calculator_for_global_optimization,
                params_already_merged=True,
            )
            # Always add results (possibly empty) so the API returns a key for each
            # requested composition; this makes the function predictable for
            # downstream consumers and tests.
            all_results[formula_str] = results
            if not results and verbosity >= 1:
                logger.warning("No minima found for %s (results empty)", formula_str)
            if verbosity >= 1:
                logger.info("Finished processing %s.", formula_str)
                logger.info(
                    "  Returned %d final minima for %s", len(results), formula_str
                )
        except (
            RuntimeError,
            ValueError,
            OSError,
            sqlite3.DatabaseError,
            SCGOValidationError,
        ) as e:
            # Enhanced error logging for HPC debugging
            error_details = [
                f"Failed to process {formula_str}: {e}",
                f"Working directory: {os.getcwd()}",
            ]
            if trial_output_dir:
                error_details.append(f"Output directory: {trial_output_dir}")
                if os.path.exists(trial_output_dir):
                    try:
                        files = os.listdir(trial_output_dir)
                        error_details.append(f"Output dir contents: {files}")
                    except OSError:
                        error_details.append(
                            "Output dir exists but cannot list contents"
                        )
                else:
                    error_details.append("Output directory does not exist")

            logger.error(" | ".join(error_details), exc_info=(verbosity >= 2))
            all_results[formula_str] = []
            if verbosity >= 1:
                logger.warning(
                    f"Skipping {formula_str} and continuing campaign "
                    f"({i + 1}/{num_compositions})"
                )
            continue

    # Best-effort: drop shared calculator reference and free CUDA memory to avoid
    # fragmentation when campaigns are run sequentially in the same process.
    del calculator_for_global_optimization
    cleanup_torch_cuda(logger=logger)

    return all_results
