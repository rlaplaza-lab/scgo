"""Core workflow functions for global optimization.

Coordinates datetime-tagged runs, manages output, filters results, validates minima.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.emt import EMT
from ase.io import write
from ase_ga.utilities import get_all_atom_types

from scgo.algorithms import bh_go, ga_go, simple_go
from scgo.database import SCGODatabaseManager
from scgo.database.metadata import (
    add_metadata,
    get_metadata,
    mark_final_minima_in_db,
)
from scgo.initialization import create_initial_cluster
from scgo.initialization.atomic_radii import build_blmin_from_zs
from scgo.initialization.initialization_config import BLMIN_RATIO_DEFAULT
from scgo.surface.config import SurfaceSystemConfig
from scgo.surface.deposition import create_deposited_cluster
from scgo.surface.validation import (
    validate_stored_mobile_partition_metadata,
    validate_stored_slab_adsorbate_metadata,
)
from scgo.system_types import (
    get_system_policy,
    validate_adsorbate_definition,
    validate_system_type_settings,
)
from scgo.utils.fitness_strategies import ensure_fitness_strategy_resolved
from scgo.utils.helpers import (
    adsorbate_primary_cell_shift,
    apply_primary_cell_shift,
    canonicalize_storage_frame,
    ensure_directory_exists,
    ensure_final_id,
    filter_dict_keys,
    filter_unique_minima,
    get_cluster_formula,
    get_provenance,
    is_true_minimum,
)
from scgo.utils.logging import get_logger
from scgo.utils.parallel_workers import resolve_n_jobs_to_workers
from scgo.utils.rng_helpers import create_child_rng
from scgo.utils.run_tracking import (
    RunMetadataJSONEncoder,
    ensure_run_id,
    save_run_metadata,
)
from scgo.utils.ts_provenance import ts_output_provenance
from scgo.utils.validation import validate_composition

# Default required calculator methods
_DEFAULT_REQUIRED_METHODS = ["get_potential_energy", "get_forces"]

_SURFACE_SYSTEM_TYPES = frozenset({"surface_cluster", "surface_cluster_adsorbate"})

_VALIDATION_CALCULATOR: Calculator | None = None


def _init_validation_worker(calculator: Calculator) -> None:
    global _VALIDATION_CALCULATOR
    _VALIDATION_CALCULATOR = calculator


def _validate_minimum_worker(
    payload: tuple[float, Atoms, float, bool, float],
) -> tuple[float, Atoms] | None:
    energy, atoms, fmax_threshold, check_hessian, imag_freq_threshold = payload
    if _VALIDATION_CALCULATOR is None:
        raise RuntimeError("Validation worker calculator not initialized")
    if is_true_minimum(
        atoms=atoms,
        calculator=_VALIDATION_CALCULATOR,
        fmax_threshold=fmax_threshold,
        check_hessian=check_hessian,
        imag_freq_threshold=imag_freq_threshold,
    ):
        return (energy, atoms)
    return None


def _create_surface_initialized_atoms(
    *,
    composition: list[str],
    surface_config: SurfaceSystemConfig,
    rng: np.random.Generator,
    adsorbate_definition: Any = None,
    adsorbate_fragment_template: Atoms | None = None,
    cluster_adsorbate_config: Any = None,
) -> Atoms:
    slab = surface_config.slab
    n_slab = len(slab)
    n_top = len(composition)
    template = Atoms(
        symbols=list(slab.get_chemical_symbols()) + composition,
        positions=np.vstack([slab.get_positions(), np.zeros((n_top, 3))]),
        cell=slab.cell,
        pbc=slab.pbc,
    )
    idx_top = range(n_slab, n_slab + n_top)
    blmin = build_blmin_from_zs(
        get_all_atom_types(template, list(idx_top)),
        ratio=BLMIN_RATIO_DEFAULT,
    )
    deposited = create_deposited_cluster(
        composition=composition,
        slab=slab,
        blmin=blmin,
        rng=rng,
        config=surface_config,
        adsorbate_definition=adsorbate_definition,
        adsorbate_fragment_template=adsorbate_fragment_template,
        cluster_adsorbate_config=cluster_adsorbate_config,
    )
    if deposited is None:
        raise RuntimeError("Failed to create initial surface-supported structure.")
    return deposited


def _create_gas_cluster_adsorbate_initial_atoms(
    *,
    composition: list[str],
    rng: np.random.Generator,
    adsorbate_definition: Any,
    adsorbate_fragment_template: Atoms | None = None,
    cluster_adsorbate_config: Any = None,
    vacuum: float = 10.0,
    init_mode: str = "smart",
    max_hierarchical_attempts: int = 200,
    previous_search_glob: str = "**/*.db",
) -> Atoms:
    """Build hierarchical gas-phase core+fragment seed for adsorbate runs."""
    from scgo.cluster_adsorbate.hierarchical import (
        build_hierarchical_core_fragment_cluster,
    )

    atoms = build_hierarchical_core_fragment_cluster(
        composition,
        adsorbate_definition,
        rng,
        previous_search_glob,
        adsorbate_fragment_template,
        cluster_adsorbate_config,
        cluster_init_vacuum=vacuum,
        init_mode=init_mode,
        max_placement_attempts=max_hierarchical_attempts,
    )
    if atoms is None:
        raise RuntimeError(
            "Failed to build hierarchical gas-phase core+fragment seed; "
            "increase max_hierarchical_attempts or relax fragment placement."
        )
    return atoms


def _is_slab_surface_minimum(atoms: Atoms) -> tuple[bool, int]:
    """Return whether ``atoms`` is a slab surface minimum and its ``n_slab_atoms``."""
    system_type = get_metadata(atoms, "system_type")
    n_slab = int(get_metadata(atoms, "n_slab_atoms", 0) or 0)
    if system_type in _SURFACE_SYSTEM_TYPES and n_slab > 0:
        return True, n_slab
    return False, n_slab


def _resolve_surface_alignment_kwargs(
    global_optimizer_kwargs: dict[str, Any],
) -> dict[str, Any] | None:
    """Resolve slab final-write alignment knobs from GO kwargs and system policy."""
    system_type = global_optimizer_kwargs.get("system_type")
    if not isinstance(system_type, str):
        raise ValueError(
            "system_type must be set in global_optimizer_kwargs for surface alignment."
        )
    policy = get_system_policy(system_type)  # type: ignore[arg-type]
    if not policy.uses_surface:
        return None

    cell_remap = bool(global_optimizer_kwargs.get("neb_surface_cell_remap", True))
    lattice_rotation = bool(
        global_optimizer_kwargs.get("neb_surface_lattice_rotation", True)
    )
    max_shift = int(global_optimizer_kwargs.get("neb_surface_max_lattice_shift", 1))
    cell_remap = policy.neb_surface_cell_remap and cell_remap
    lattice_rotation = policy.neb_surface_lattice_rotation and lattice_rotation
    return {
        "enable_cell_remap": cell_remap,
        "enable_lattice_rotation": lattice_rotation,
        "max_lattice_shift": max_shift,
    }


def _align_slab_minimum_to_reference(
    reference: Atoms,
    candidate: Atoms,
    *,
    n_slab: int,
    enable_cell_remap: bool,
    enable_lattice_rotation: bool,
    max_lattice_shift: int,
) -> None:
    """Align ``candidate`` to ``reference`` using the TS slab PBC protocol (in-place)."""
    from scgo.ts_search.transition_state import _align_product_surface_pbc

    aligned = _align_product_surface_pbc(
        reference,
        candidate.get_positions(),
        n_slab=n_slab,
        enable_cell_remap=enable_cell_remap,
        enable_lattice_rotation=enable_lattice_rotation,
        max_lattice_shift=max_lattice_shift,
    )
    candidate.set_positions(aligned)
    candidate.set_cell(reference.cell)
    candidate.pbc = reference.pbc


# Consumed by ``scgo`` for hierarchical/surface init only; not passed to simple/BH.
_INIT_ONLY_OPTIMIZER_KWARGS = frozenset(
    {
        "adsorbate_fragment_template",
        "vacuum",
        "init_mode",
        "max_hierarchical_attempts",
        "previous_search_glob",
    }
)


def _write_timing_json_enabled(global_optimizer_kwargs: dict[str, Any]) -> bool:
    return bool(global_optimizer_kwargs.get("write_timing_json", False))


def _optimizer_kwargs_for_algorithm_call(
    optimizer_kwargs: dict[str, Any],
    *,
    global_optimizer: str,
) -> dict[str, Any]:
    """Return kwargs safe to pass to simple/BH after initial structure construction."""
    if global_optimizer == "ga":
        return optimizer_kwargs
    return {
        key: value
        for key, value in optimizer_kwargs.items()
        if key not in _INIT_ONLY_OPTIMIZER_KWARGS
    }


def _sanitize_global_optimizer_kwargs_for_metadata(
    global_optimizer_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Copy kwargs for JSON metadata: drop non-serializable objects (relaxer, slab)."""
    gok = global_optimizer_kwargs.copy()
    gok.pop("relaxer", None)
    gok.pop("adsorbate_fragment_template", None)
    gok.pop("cluster_adsorbate_config", None)
    surface_config = gok.pop("surface_config", None)
    if surface_config is not None:
        if not isinstance(surface_config, SurfaceSystemConfig):
            raise TypeError(
                "surface_config must be a SurfaceSystemConfig instance or None"
            )
        slab = surface_config.slab
        n_slab = len(slab)
        gok["surface_config"] = {
            "present": True,
            "n_slab_atoms": n_slab,
            "slab_chemical_symbols": list(slab.get_chemical_symbols()),
            "surface_normal_axis": surface_config.surface_normal_axis,
            "fix_all_slab_atoms": surface_config.fix_all_slab_atoms,
            "n_fix_bottom_slab_layers": surface_config.n_fix_bottom_slab_layers,
            "n_relax_top_slab_layers": surface_config.n_relax_top_slab_layers,
            "adsorption_height_min": surface_config.adsorption_height_min,
            "adsorption_height_max": surface_config.adsorption_height_max,
            "comparator_use_mic": surface_config.comparator_use_mic,
            "cluster_init_vacuum": surface_config.cluster_init_vacuum,
            "init_mode": surface_config.init_mode,
            "max_placement_attempts": surface_config.max_placement_attempts,
        }
    return gok


# Algorithm registry
_ALGORITHM_REGISTRY: dict[str, dict[str, Any]] = {
    "simple": {
        "function": simple_go,
    },
    "bh": {
        "function": bh_go,
    },
    "ga": {
        "function": ga_go,
    },
}


def _ensure_calculator(calculator: Calculator | None) -> Calculator:
    """Return *calculator* or a default EMT instance when None."""
    return calculator or EMT()


def _validate_calculator_compatibility(
    calculator: Calculator,
    required_methods: list[str] | None = None,
) -> tuple[bool, str]:
    """Validate calculator has required methods and returns expected types.

    Args:
        calculator: ASE calculator instance
        required_methods: List of method names to check (default: ["get_potential_energy", "get_forces"])

    Returns:
        tuple: (is_valid, error_message)
    """
    required_methods = required_methods or _DEFAULT_REQUIRED_METHODS

    missing_methods = [
        method_name
        for method_name in required_methods
        if not hasattr(calculator, method_name)
        or not callable(getattr(calculator, method_name))
    ]

    if missing_methods:
        return False, f"Calculator missing required methods: {missing_methods}"

    return True, "Calculator is compatible"


def scgo(
    composition: list[str],
    global_optimizer: str,
    global_optimizer_kwargs: dict[str, Any],
    output_dir: str,
    rng: np.random.Generator,
    calculator_for_global_optimization: Calculator | None = None,
    verbosity: int = 1,
    run_id: str | None = None,
    clean: bool = False,
    timing_output_dir: str | None = None,
    timing_collector: list[dict[str, Any]] | None = None,
) -> list[tuple[float, Atoms]]:
    """Run global optimization for a fixed composition into one run directory.

    Args:
        composition: List of atomic symbols.
        global_optimizer: Optimizer name ("simple", "bh", or "ga").
        global_optimizer_kwargs: Optimizer parameters.
        output_dir: Run output directory (typically ``run_*/``).
        rng: Random number generator.
        calculator_for_global_optimization: ASE calculator.
        verbosity: Verbosity level (0=quiet, 1=normal, 2=debug, 3=trace).
        run_id: Optional run ID.
        clean: Start fresh if True.

    Returns:
        List of (energy, Atoms) for minima.

    Raises:
        ValueError: For invalid parameters.
    """
    logger = get_logger(__name__)

    validate_composition(composition, allow_empty=False, allow_tuple=False)

    if not isinstance(global_optimizer, str):
        raise ValueError("global_optimizer must be a string")

    if not isinstance(global_optimizer_kwargs, dict):
        raise ValueError("global_optimizer_kwargs must be a dictionary")

    if not isinstance(output_dir, str) or not output_dir:
        raise ValueError("output_dir must be a non-empty string")

    # RNG must be a numpy Generator (required for deterministic behavior)
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy.random.Generator")

    if not isinstance(verbosity, int) or verbosity not in (0, 1, 2, 3):
        raise ValueError("verbosity must be one of 0, 1, 2, or 3")

    calculator_for_global_optimization = _ensure_calculator(
        calculator_for_global_optimization
    )

    # Ensure file-based calculators run in the trial directory to avoid collisions
    if hasattr(calculator_for_global_optimization, "directory"):
        calculator_for_global_optimization.directory = output_dir

    is_valid, error_msg = _validate_calculator_compatibility(
        calculator_for_global_optimization
    )
    if not is_valid:
        calc_type = type(calculator_for_global_optimization).__name__
        calc_module = type(calculator_for_global_optimization).__module__
        raise ValueError(
            f"Calculator validation failed: {error_msg}. "
            f"Calculator type: {calc_type} (from {calc_module}). "
            f"Ensure the calculator implements get_potential_energy() and get_forces() methods."
        )

    # Filter keys handled at scgo/run_trials level so **optimizer_kwargs cannot
    # override explicit run_id/clean.
    optimizer_name_lower = global_optimizer.lower()
    if optimizer_name_lower not in _ALGORITHM_REGISTRY:
        raise ValueError(
            f"Unknown global_optimizer: {global_optimizer}. "
            f"Must be one of {list(_ALGORITHM_REGISTRY.keys())}"
        )
    optimizer_kwargs = filter_dict_keys(
        global_optimizer_kwargs,
        {"run_id", "clean", "timing_output_dir", "timing_collector"},
    )
    timing_kwargs: dict[str, Any] = {}
    if timing_output_dir is not None:
        timing_kwargs["timing_output_dir"] = timing_output_dir
    if timing_collector is not None:
        timing_kwargs["timing_collector"] = timing_collector
    if "fitness_strategy" in optimizer_kwargs:
        optimizer_kwargs["fitness_strategy"] = ensure_fitness_strategy_resolved(
            optimizer_kwargs["fitness_strategy"]
        )
    system_type = optimizer_kwargs.get("system_type")
    if not isinstance(system_type, str):
        raise ValueError(
            "system_type must be set in global_optimizer_kwargs "
            "(e.g. 'gas_cluster', 'surface_cluster')."
        )
    policy = get_system_policy(system_type)
    surface_cfg = optimizer_kwargs.get("surface_config")
    validate_system_type_settings(
        system_type=system_type,
        surface_config=surface_cfg
        if isinstance(surface_cfg, SurfaceSystemConfig)
        else None,
    )
    validate_adsorbate_definition(
        system_type=system_type,
        composition=composition,
        adsorbate_definition=optimizer_kwargs.get("adsorbate_definition"),
        context="scgo",
    )
    if policy.has_adsorbate and not policy.uses_surface:
        ads_def = optimizer_kwargs.get("adsorbate_definition")
        if isinstance(ads_def, dict):
            core_symbols = [str(s) for s in ads_def.get("core_symbols", [])]
            if len(core_symbols) == 0:
                logger.info(
                    "Gas adsorbate run with empty core_symbols: skipping global optimization."
                )
                return []

    ensure_directory_exists(output_dir)

    algo_config = _ALGORITHM_REGISTRY[optimizer_name_lower]
    algo_function = algo_config["function"]

    if optimizer_name_lower == "ga":
        all_minima = ga_go(
            composition=composition,
            output_dir=output_dir,
            calculator=calculator_for_global_optimization,
            rng=rng,
            verbosity=verbosity,
            run_id=run_id,
            clean=clean,
            **{**optimizer_kwargs, **timing_kwargs},
        )
    else:
        # Non-GA algorithms need explicit starting atoms.
        if policy.uses_surface:
            surface_config = optimizer_kwargs.get("surface_config")
            if not isinstance(surface_config, SurfaceSystemConfig):
                raise ValueError(
                    f"system_type={system_type!r} requires surface_config for "
                    f"{optimizer_name_lower.upper()} initialization."
                )
            atoms = _create_surface_initialized_atoms(
                composition=composition,
                surface_config=surface_config,
                rng=rng,
                adsorbate_definition=optimizer_kwargs.get("adsorbate_definition"),
                adsorbate_fragment_template=optimizer_kwargs.get(
                    "adsorbate_fragment_template"
                ),
                cluster_adsorbate_config=optimizer_kwargs.get(
                    "cluster_adsorbate_config"
                ),
            )
            optimizer_kwargs.setdefault("n_slab", len(surface_config.slab))
        elif policy.has_adsorbate:
            ads_def = optimizer_kwargs.get("adsorbate_definition")
            if not isinstance(ads_def, dict):
                raise ValueError(
                    f"system_type={system_type!r} requires adsorbate_definition in "
                    f"global_optimizer_kwargs for {optimizer_name_lower.upper()}."
                )
            if optimizer_kwargs.get("adsorbate_fragment_template") is None:
                raise ValueError(
                    f"system_type={system_type!r} requires adsorbate_fragment_template "
                    "for hierarchical adsorbate initialization."
                )
            vac = float(optimizer_kwargs.get("vacuum", 10.0))
            mode = str(optimizer_kwargs.get("init_mode", "smart"))
            max_h = int(optimizer_kwargs.get("max_hierarchical_attempts", 200))
            glb = str(optimizer_kwargs.get("previous_search_glob", "**/*.db"))
            atoms = _create_gas_cluster_adsorbate_initial_atoms(
                composition=composition,
                rng=rng,
                adsorbate_definition=ads_def,
                adsorbate_fragment_template=optimizer_kwargs.get(
                    "adsorbate_fragment_template"
                ),
                cluster_adsorbate_config=optimizer_kwargs.get(
                    "cluster_adsorbate_config"
                ),
                vacuum=vac,
                init_mode=mode,
                max_hierarchical_attempts=max_h,
                previous_search_glob=glb,
            )
        else:
            atoms = create_initial_cluster(composition, rng=rng)
        atoms.calc = calculator_for_global_optimization
        algo_kwargs = _optimizer_kwargs_for_algorithm_call(
            optimizer_kwargs,
            global_optimizer=optimizer_name_lower,
        )
        all_minima = algo_function(
            atoms=atoms,
            output_dir=output_dir,
            rng=rng,
            verbosity=verbosity,
            run_id=run_id,
            clean=clean,
            **algo_kwargs,
            **timing_kwargs,
        )

    if not all_minima:
        logger.info("Global optimization finished but found no valid minima.")
        return []

    for _, atoms_obj in all_minima:
        add_metadata(atoms_obj, run_id=run_id)

    return all_minima


def run_trials(
    composition: list[str],
    global_optimizer: str,
    global_optimizer_kwargs: dict[str, Any],
    output_dir: str,
    rng: np.random.Generator,
    calculator_for_global_optimization: Calculator | None = None,
    validate_with_hessian: bool = True,
    fmax_threshold: float = 0.05,
    check_hessian: bool = True,
    imag_freq_threshold: float = 50.0,
    validation_n_jobs: int = 1,
    tag_final_minima: bool = True,
    verbosity: int = 1,
    run_id: str | None = None,
    clean: bool = False,
) -> list[tuple[float, Atoms]]:
    """Run global optimization once, filter and validate results across runs.

    Args:
        composition: List of atomic symbols.
        global_optimizer: Optimizer name (e.g., "bh", "ga").
        global_optimizer_kwargs: Optimizer parameters.
        output_dir: Searches directory (parent of ``run_*/`` dirs).
        rng: Random number generator.
        calculator_for_global_optimization: ASE calculator.
        validate_with_hessian: Whether to validate with Hessian.
        verbosity: Verbosity level.
        run_id: Optional run ID.
        clean: Start fresh if True.

    Returns:
        List of (energy, Atoms) for unique minima.
    """
    logger = get_logger(__name__)

    # Validate inputs early
    validate_composition(composition, allow_empty=False, allow_tuple=False)

    if not isinstance(global_optimizer, str):
        raise ValueError("global_optimizer must be a string")

    if not isinstance(global_optimizer_kwargs, dict):
        raise ValueError("global_optimizer_kwargs must be a dictionary")

    if not isinstance(global_optimizer_kwargs.get("system_type"), str):
        raise ValueError(
            "system_type must be set in global_optimizer_kwargs "
            "(e.g. 'gas_cluster', 'surface_cluster')."
        )

    if not isinstance(output_dir, str) or not output_dir:
        raise ValueError("output_dir must be a non-empty string")

    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy.random.Generator")

    if not isinstance(validate_with_hessian, bool):
        raise ValueError("validate_with_hessian must be a boolean")

    if not isinstance(verbosity, int) or verbosity not in (0, 1, 2, 3):
        raise ValueError("verbosity must be one of 0, 1, 2, or 3")

    calculator_for_global_optimization = _ensure_calculator(
        calculator_for_global_optimization
    )

    # Generate run_id if not provided
    run_id = ensure_run_id(run_id, verbosity=verbosity, logger=logger)

    # Create run-specific output directory
    run_output_dir = os.path.join(output_dir, run_id)
    ensure_directory_exists(run_output_dir)

    # Ensure final unique minima directory exists even if no minima are found
    final_xyz_dir = os.path.join(output_dir, "final_unique_minima")
    ensure_directory_exists(final_xyz_dir)

    # Cache cluster formula (used multiple times)
    composition_str = get_cluster_formula(composition)

    # Save run metadata (include formula and run parameters for traceability)
    gok_for_metadata = _sanitize_global_optimizer_kwargs_for_metadata(
        global_optimizer_kwargs
    )
    params = {
        "global_optimizer": global_optimizer,
        "global_optimizer_kwargs": gok_for_metadata,
        "validate_with_hessian": validate_with_hessian,
        "verbosity": verbosity,
        "clean": clean,
        "calculator": calculator_for_global_optimization.__class__.__name__
        if calculator_for_global_optimization
        else None,
    }
    save_run_metadata(
        run_output_dir,
        run_id,
        metadata={
            "composition": composition,
            "formula": composition_str,
            "params": params,
        },
    )

    # Load previous run results BEFORE running trials (better UX)
    previous_minima = []
    if not clean:
        # Use database manager for efficient loading with caching
        with SCGODatabaseManager(
            base_dir=output_dir, enable_caching=True
        ) as db_manager:
            previous_minima = db_manager.load_previous_results(
                composition=composition,
                current_run_id=run_id,
                prefer_final_unique=True,
            )
            if previous_minima:
                logger.info(
                    f"Loaded {len(previous_minima)} minima from previous runs "
                    f"(excluding current run {run_id})"
                )

    all_raw_minima = []
    write_timing = _write_timing_json_enabled(global_optimizer_kwargs)

    run_rng = create_child_rng(rng)
    logger.info("Running global optimization for run %s", run_id)

    all_raw_minima = scgo(
        composition=composition,
        global_optimizer=global_optimizer,
        global_optimizer_kwargs=global_optimizer_kwargs,
        output_dir=run_output_dir,
        rng=run_rng,
        calculator_for_global_optimization=calculator_for_global_optimization,
        verbosity=verbosity,
        run_id=run_id,
        clean=clean,
        timing_output_dir=run_output_dir if write_timing else None,
    )

    # Combine all results (previous + current) before deduplication
    if previous_minima:
        all_minima_for_filtering = previous_minima + all_raw_minima
        logger.info(
            f"Combined {len(previous_minima)} previous + {len(all_raw_minima)} current minima"
        )
    else:
        all_minima_for_filtering = all_raw_minima

    if not all_minima_for_filtering:
        logger.info("No minima found.")
        _write_results_summary(
            output_dir=output_dir,
            final_minima=[],
            composition_str=composition_str,
            run_id=run_id,
            params=params,
        )
        return []

    logger.info(
        f"Run complete. Found {len(all_raw_minima)} raw minima from current run."
    )
    logger.info("Filtering for unique structures across all runs...")
    surface_cfg = global_optimizer_kwargs.get("surface_config")
    dedupe_mic = (
        bool(surface_cfg.comparator_use_mic) if surface_cfg is not None else False
    )
    unique_candidates = filter_unique_minima(
        all_minima_for_filtering,
        n_top=len(composition),
        mic=dedupe_mic,
    )
    logger.info(f"Found {len(unique_candidates)} unique candidates.")

    if not unique_candidates:
        _write_results_summary(
            output_dir=output_dir,
            final_minima=[],
            composition_str=composition_str,
            run_id=run_id,
            params=params,
        )
        return []

    if validate_with_hessian:
        logger.info(
            f"Validating {len(unique_candidates)} unique candidates to confirm they are true minima...",
        )

        # Ensure validation runs in a separate directory to avoid overwriting run files
        if hasattr(calculator_for_global_optimization, "directory"):
            val_dir = os.path.join(output_dir, "validation")
            ensure_directory_exists(val_dir)
            calculator_for_global_optimization.directory = val_dir

        validated_minima = []
        n_validate_workers = (
            resolve_n_jobs_to_workers(validation_n_jobs)
            if check_hessian and validation_n_jobs != 1
            else 1
        )
        payloads = [
            (energy, atoms, fmax_threshold, check_hessian, imag_freq_threshold)
            for energy, atoms in unique_candidates
        ]

        if n_validate_workers > 1 and len(payloads) > 1:
            logger.info(
                "Validating %d unique candidates with %d parallel workers...",
                len(payloads),
                n_validate_workers,
            )
            with ProcessPoolExecutor(
                max_workers=min(n_validate_workers, len(payloads)),
                initializer=_init_validation_worker,
                initargs=(calculator_for_global_optimization,),
            ) as executor:
                futures = [
                    executor.submit(_validate_minimum_worker, payload)
                    for payload in payloads
                ]
                for i, future in enumerate(as_completed(futures), 1):
                    try:
                        validated = future.result()
                    except (OSError, RuntimeError, ValueError) as e:
                        logger.warning("Validation failed for candidate %d: %s", i, e)
                        continue
                    if validated is not None:
                        validated_minima.append(validated)
        else:
            for i, (energy, atoms) in enumerate(unique_candidates):
                logger.info(
                    f"Validating candidate {i + 1}/{len(unique_candidates)} (E={energy:.4f} eV)...",
                )
                try:
                    is_valid = is_true_minimum(
                        atoms=atoms,
                        calculator=calculator_for_global_optimization,
                        fmax_threshold=fmax_threshold,
                        check_hessian=check_hessian,
                        imag_freq_threshold=imag_freq_threshold,
                    )
                    if is_valid:
                        validated_minima.append((energy, atoms))
                    else:
                        logger.info(f"Candidate {i + 1} rejected")
                except (OSError, RuntimeError, ValueError) as e:
                    logger.warning(
                        f"Validation failed for candidate {i + 1} (E={energy:.4f} eV): {e}"
                    )

        if not validated_minima:
            logger.info(
                "Validation finished. No candidates were confirmed as true minima."
            )
            _write_results_summary(
                output_dir=output_dir,
                final_minima=[],
                composition_str=composition_str,
                run_id=run_id,
                params=params,
            )
            return []

        final_minima = validated_minima
    else:
        final_minima = unique_candidates

    best_energy, _ = final_minima[0]
    logger.info(f"Process complete. Found {len(final_minima)} final unique minima.")
    logger.info(f"Best potential energy: {best_energy:.4f} eV")

    final_xyz_dir = os.path.join(output_dir, "final_unique_minima")
    logger.info(
        f'Writing {len(final_minima)} final structures to "{os.path.basename(final_xyz_dir)}"'
    )

    # Write results summary file (composition_str already cached above)
    _write_results_summary(
        output_dir=output_dir,
        final_minima=final_minima,
        composition_str=composition_str,
        run_id=run_id,
        params=params,
    )

    align_kwargs_source = dict(global_optimizer_kwargs)
    if not isinstance(align_kwargs_source.get("system_type"), str):
        raise ValueError(
            "system_type must be set in global_optimizer_kwargs for result alignment."
        )
    surface_align_kwargs = _resolve_surface_alignment_kwargs(align_kwargs_source)
    reference_atoms: Atoms | None = None
    reference_n_slab = 0
    reference_primary_cell_shift: np.ndarray | None = None
    if surface_align_kwargs and final_minima:
        _best_energy, best_atoms = final_minima[0]
        is_slab_ref, reference_n_slab = _is_slab_surface_minimum(best_atoms)
        if is_slab_ref:
            reference_atoms = best_atoms.copy()
            reference_atoms.calc = None
            reference_primary_cell_shift = adsorbate_primary_cell_shift(
                reference_atoms, n_slab=reference_n_slab
            )

    final_minima_info: list[dict] = []
    written_xyz: set[Path] = set()
    for i, (_energy, atoms) in enumerate(final_minima):
        provenance = get_provenance(atoms)
        atoms_run_id = provenance.get("run_id", run_id)

        filename = f"{composition_str}_minimum_{i + 1:02d}_{atoms_run_id}.xyz"
        filepath = os.path.join(final_xyz_dir, filename)

        # Match DB rows by pre-alignment geometry (same frame as relaxed candidates).
        final_id = ensure_final_id(atoms, _energy)

        atoms_clean = atoms.copy()
        atoms_clean.calc = None
        n_slab_meta = get_metadata(atoms_clean, "n_slab_atoms", 0) or 0
        system_type = get_metadata(atoms_clean, "system_type")
        try:
            validate_stored_slab_adsorbate_metadata(atoms_clean)
            validate_stored_mobile_partition_metadata(atoms_clean)
        except ValueError as e:
            logger.warning("Structure metadata check before write: %s", e)
        aligned_to_surface_reference = False
        if reference_atoms is not None and surface_align_kwargs is not None:
            is_slab_candidate, _ = _is_slab_surface_minimum(atoms_clean)
            if is_slab_candidate:
                _align_slab_minimum_to_reference(
                    reference_atoms,
                    atoms_clean,
                    n_slab=reference_n_slab,
                    enable_cell_remap=surface_align_kwargs["enable_cell_remap"],
                    enable_lattice_rotation=surface_align_kwargs[
                        "enable_lattice_rotation"
                    ],
                    max_lattice_shift=surface_align_kwargs["max_lattice_shift"],
                )
                aligned_to_surface_reference = True
                if reference_primary_cell_shift is not None and np.any(
                    reference_primary_cell_shift != 0
                ):
                    apply_primary_cell_shift(atoms_clean, reference_primary_cell_shift)
        if not aligned_to_surface_reference:
            if (
                system_type in {"surface_cluster", "surface_cluster_adsorbate"}
                and n_slab_meta
            ):
                canonicalize_storage_frame(
                    atoms_clean,
                    pbc_aware=True,
                    center=False,
                    n_slab=int(n_slab_meta),
                )
            else:
                canonicalize_storage_frame(atoms_clean)
        if "tags" in atoms_clean.arrays:
            del atoms_clean.arrays["tags"]

        write(filepath, atoms_clean)
        written_xyz.add(Path(filepath))

        final_minima_info.append(
            {
                "atoms": atoms,
                "energy": _energy,
                "rank": i + 1,
                "final_written": filepath,
                "final_id": final_id,
            }
        )

    # Drop superseded XYZ files so the folder mirrors the deduplicated final set.
    for stale in Path(final_xyz_dir).glob(f"{composition_str}_minimum_*.xyz"):
        if stale not in written_xyz:
            stale.unlink(missing_ok=True)

    # Mark final minima in DB (if enabled) to avoid re-scanning later
    if tag_final_minima:
        try:
            mark_final_minima_in_db(final_minima_info, base_dir=output_dir)
        except (sqlite3.DatabaseError, sqlite3.OperationalError, OSError) as e:
            # Consider DB tagging a systemic failure -- surface it after logging
            logger.warning(f"Failed to tag final minima in DB: {e}")
            raise

    return final_minima


def _write_results_summary(
    output_dir: str,
    final_minima: list[tuple[float, Atoms]],
    composition_str: str,
    run_id: str,
    params: dict[str, Any] | None = None,
) -> None:
    """Write a summary file of results by run.

    Args:
        output_dir: Base output directory.
        final_minima: List of final unique minima.
        composition_str: Chemical formula string.
        run_id: Current run ID.
        params: Same snapshot as ``run_*/metadata.json`` (optimizer, trials, etc.).
    """
    logger = get_logger(__name__)

    # Count structures by run_id
    run_counts = Counter()
    for _, atoms in final_minima:
        provenance = get_provenance(atoms)
        run_id_from_atoms = provenance.get("run_id", run_id)
        run_counts[run_id_from_atoms] += 1

    summary = ts_output_provenance()
    timing_relpath = f"{run_id}/timing.json" if run_id else None
    if run_id and not os.path.isfile(os.path.join(output_dir, run_id, "timing.json")):
        timing_relpath = None
    summary.update(
        {
            "composition": composition_str,
            "total_unique_minima": len(final_minima),
            "minima_by_run": dict(run_counts),
            "current_run_id": run_id,
            "params": params,
            "run_metadata_relpath": (f"{run_id}/metadata.json" if run_id else None),
            "run_timing_relpath": timing_relpath,
        }
    )

    summary_file = os.path.join(output_dir, "results_summary.json")
    try:
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, cls=RunMetadataJSONEncoder)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Wrote results summary to {summary_file}")
    except (OSError, TypeError) as e:
        logger.warning(f"Failed to write results summary: {e}")
        raise
