"""Simple optimization for 1-2 atom clusters.

This module provides a minimal optimization approach for very small clusters
(1-2 atoms) where Basin Hopping is unnecessary. For these cases, there's only
one meaningful structure to optimize, so we just perform a single local
optimization without iterations.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from ase import Atoms
from ase.optimize import LBFGS
from ase.optimize.optimize import Optimizer

from scgo.database import HPC_DATABASE_EXCEPTIONS, close_data_connection, setup_database
from scgo.database.sync import PRESET_HPC, database_retry
from scgo.exceptions import SCGOValidationError
from scgo.metadata.atoms import set_tags
from scgo.utils.helpers import (
    extract_minima_from_database,
    perform_local_relaxation,
)
from scgo.utils.logging import get_logger, log_debug_v, log_info_v
from scgo.utils.rng_helpers import create_child_rng, ensure_rng_or_create
from scgo.utils.validation import (
    validate_atoms,
    validate_calculator_attached,
    validate_positive,
)

# Atoms closer than this (Å) are effectively coincident: the local relaxation
# would see non-finite forces, so a small rng-drawn displacement is applied.
_COINCIDENT_ATOM_TOLERANCE = 1e-3
# Displacement (Å) used when breaking a coincident-atom starting geometry.
_SYMMETRY_BREAK_DISPLACEMENT = 0.5
# ``perform_local_relaxation`` keys callers may forward through **kwargs.
_RELAXATION_KWARG_KEYS = ("logfile", "trajectory")


def _break_coincident_atoms(
    atoms: Atoms | None,
    rng: np.random.Generator,
) -> bool:
    """Displace coincident atoms so the local relaxation stays well defined.

    Args:
        atoms: Structure to fix in place (no-op for None or single atoms).
        rng: Generator used to draw the displacement direction.

    Returns:
        True if a displacement was applied.
    """
    if atoms is None or len(atoms) < 2:
        return False
    positions = atoms.get_positions()
    distance = float(np.linalg.norm(positions[1] - positions[0]))
    if distance >= _COINCIDENT_ATOM_TOLERANCE:
        return False

    direction = np.asarray(rng.normal(size=3), dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm <= 0.0:
        direction = np.array([1.0, 0.0, 0.0])
        norm = 1.0
    positions[1] = positions[1] + direction / norm * _SYMMETRY_BREAK_DISPLACEMENT
    atoms.set_positions(positions)
    return True


def simple_go(
    atoms: Atoms,
    output_dir: str,
    rng: np.random.Generator,
    niter: int = 1,
    fmax: float = 0.05,
    niter_local_relaxation: int = 250,
    optimizer: type[Optimizer] = LBFGS,
    verbosity: int = 1,
    run_id: str | None = None,
    clean: bool = False,
    **kwargs: Any,
) -> list[tuple[float, Atoms]]:
    """Simple local optimization for 1-2 atom clusters.

    Performs a single local optimization without Basin Hopping iterations.

    Args:
        atoms: Initial Atoms object representing the cluster. Calculator must be attached.
        output_dir: Directory where ASE database will be stored.
        rng: Random number generator (numpy.random.Generator). Required. Used to
            break a degenerate (coincident-atom) starting geometry reproducibly.
        niter: Accepted for signature parity with the other optimizers and
            ignored; exactly one local relaxation is performed. Default 1.
        fmax: Maximum force criterion for convergence (eV/Å). Default 0.05.
        niter_local_relaxation: Maximum steps for each local relaxation. Default 250.
        optimizer: ASE optimizer class (e.g., BFGS) for local relaxations. Default LBFGS.
        verbosity: Verbosity level (0=quiet, 1=normal, 2=debug, 3=trace). Default 1.
            Gates the progress/diagnostic logging emitted by this function.
        run_id: Optional run id for database provenance (same as other optimizers).
        clean: If True, remove an existing database in the trial directory.
        **kwargs: Extra keys from shared ``global_optimizer_kwargs``. ``logfile``
            and ``trajectory`` are forwarded to the local relaxation; any other
            key is ignored and reported at debug level.

    Returns:
        List of (energy, Atoms) tuples for local minima found. Typically a single structure.

    Raises:
        SCGOValidationError: If atoms is not an ASE Atoms object, no calculator
            is attached, fmax is not positive, or the cluster is not 1-2 atoms.
    """
    logger = get_logger(__name__)

    validate_atoms(atoms)
    calculator = validate_calculator_attached(atoms, "simple optimization")
    validate_positive("fmax", fmax, strict=True)
    rng = ensure_rng_or_create(rng)

    n_atoms = len(atoms)
    if n_atoms < 1 or n_atoms > 2:
        raise SCGOValidationError(
            f"simple_go only supports 1-2 atoms, got {n_atoms} atoms"
        )

    relaxation_kwargs = {k: kwargs[k] for k in _RELAXATION_KWARG_KEYS if k in kwargs}
    ignored_kwargs = sorted(set(kwargs) - set(relaxation_kwargs))
    if ignored_kwargs:
        log_debug_v(
            logger,
            "simple_go ignoring unsupported kwargs: %s",
            ", ".join(ignored_kwargs),
            verbosity=verbosity,
        )
    # Detach calculator temporarily for DB setup to avoid pickling issues
    calc = atoms.calc
    atoms.calc = None
    da = setup_database(
        output_dir,
        "simple_go.db",
        atoms,
        initial_candidate=atoms,
        remove_existing=clean,
        run_id=run_id,
    )
    atoms.calc = calc

    log_info_v(
        logger,
        "Performing simple optimization for %d-atom cluster",
        n_atoms,
        verbosity=verbosity,
    )

    try:
        a_optimized = database_retry(
            da.get_an_unrelaxed_candidate,
            config=PRESET_HPC,
            exception_types=HPC_DATABASE_EXCEPTIONS,
        )
        if _break_coincident_atoms(a_optimized, create_child_rng(rng)):
            logger.warning(
                "Coincident atoms in starting geometry: applied a random "
                "symmetry-breaking displacement before relaxation"
            )
        energy = perform_local_relaxation(
            a_optimized,
            calculator,
            optimizer,
            fmax,
            niter_local_relaxation,
            **relaxation_kwargs,
        )
        log_debug_v(
            logger,
            "Simple optimization finished (energy: %.4f eV)",
            energy,
            verbosity=verbosity,
        )

        if run_id is not None and a_optimized is not None:
            set_tags(a_optimized, run_id=run_id)

        database_retry(
            lambda: da.add_relaxed_step(a_optimized),
            config=PRESET_HPC,
            exception_types=HPC_DATABASE_EXCEPTIONS,
        )

        all_candidates = database_retry(
            da.get_all_relaxed_candidates,
            config=PRESET_HPC,
            exception_types=HPC_DATABASE_EXCEPTIONS,
        )
        all_minima = extract_minima_from_database(all_candidates)

        if not all_minima:
            return []

        return all_minima

    finally:
        close_data_connection(da, log_errors=False)
