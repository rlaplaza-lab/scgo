"""Place gas-phase cluster seeds onto a slab for GA initialization."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import TYPE_CHECKING

import numpy as np
from ase import Atoms
from ase.spacegroup import Spacegroup
from ase_ga.utilities import atoms_too_close, atoms_too_close_two_sets

from scgo.cluster_adsorbate.hierarchical import (
    build_hierarchical_core_fragment_cluster,
)
from scgo.initialization.geometry_helpers import _generate_rotation_matrix
from scgo.initialization.initialization_config import CONNECTIVITY_FACTOR
from scgo.surface.validation import validate_supported_cluster_deposit
from scgo.utils.logging import get_logger
from scgo.utils.parallel_workers import resolve_n_jobs_to_workers

if TYPE_CHECKING:
    from numpy.random import Generator

    from scgo.cluster_adsorbate.config import ClusterAdsorbateConfig
    from scgo.surface.config import SurfaceSystemConfig
    from scgo.system_types import AdsorbateDefinition

logger = get_logger(__name__)
_ASE_SPACEGROUP_WARMUP_LOCK = Lock()
_ASE_SPACEGROUP_WARMED = False


def _warmup_ase_spacegroup_cache() -> None:
    """Best-effort one-time warmup to avoid concurrent cold-file opens."""
    global _ASE_SPACEGROUP_WARMED
    if _ASE_SPACEGROUP_WARMED:
        return
    with _ASE_SPACEGROUP_WARMUP_LOCK:
        if _ASE_SPACEGROUP_WARMED:
            return
        try:
            # 225 is used by FCC/octahedral template generation.
            Spacegroup(225)
            _ASE_SPACEGROUP_WARMED = True
        except Exception:
            # Keep broad by design: warmup is optional and must never block deposition.
            logger.debug(
                "Failed to pre-warm ASE Spacegroup cache; continuing without warmup.",
                exc_info=True,
            )


def slab_surface_extreme(slab: Atoms, axis: int, *, upper: bool = True) -> float:
    """Return max (or min) Cartesian coordinate of slab atoms along ``axis``."""
    pos = slab.get_positions()
    if len(pos) == 0:
        return 0.0
    return float(np.max(pos[:, axis]) if upper else np.min(pos[:, axis]))


def _in_plane_translation_near_slab_atom(
    slab: Atoms, axis: int, rng: Generator, cluster_radius: float
) -> np.ndarray:
    """Random fractional shift biased towards slab atoms.

    Chooses a random slab atom and places the cluster near it, with some randomness.
    The cluster is placed within a distance that ensures connectivity.

    Args:
        slab: The slab atoms
        axis: Surface normal axis
        rng: Random number generator
        cluster_radius: Approximate radius of the cluster

    Returns:
        Shift vector in the plane
    """
    cell = slab.get_cell()

    # Get slab atom positions in the plane (excluding the surface normal axis)
    slab_positions = slab.get_positions()

    # Choose a random slab atom
    n_slab = len(slab)
    if n_slab == 0:
        # Fallback to random placement
        u, v = rng.random(), rng.random()
        if axis == 0:
            return np.asarray(u * cell[1] + v * cell[2], dtype=float)
        elif axis == 1:
            return np.asarray(u * cell[0] + v * cell[2], dtype=float)
        else:
            return np.asarray(u * cell[0] + v * cell[1], dtype=float)

    # Choose a random slab atom
    atom_idx = rng.integers(0, n_slab)
    atom_pos = slab_positions[atom_idx]

    # Create shift in the plane, biased towards this atom
    # Add some randomness to avoid exact overlap, but keep it small to ensure connectivity
    offset_scale = (
        cluster_radius * 0.1
    )  # Place very near the atom (reduced from 0.3 to 0.1 to improve connectivity)

    # Random direction in the plane
    if axis == 0:
        # Plane is yz
        angle = rng.uniform(0, 2 * np.pi)
        dy = offset_scale * np.cos(angle)
        dz = offset_scale * np.sin(angle)
        return np.asarray([0, atom_pos[1] + dy, atom_pos[2] + dz], dtype=float)
    elif axis == 1:
        # Plane is xz
        angle = rng.uniform(0, 2 * np.pi)
        dx = offset_scale * np.cos(angle)
        dz = offset_scale * np.sin(angle)
        return np.asarray([atom_pos[0] + dx, 0, atom_pos[2] + dz], dtype=float)
    else:
        # Plane is xy (axis == 2)
        angle = rng.uniform(0, 2 * np.pi)
        dx = offset_scale * np.cos(angle)
        dy = offset_scale * np.sin(angle)
        return np.asarray([atom_pos[0] + dx, atom_pos[1] + dy, 0], dtype=float)


def _in_plane_translation(
    slab: Atoms, axis: int, rng: Generator, cluster_radius: float | None = None
) -> np.ndarray:
    """Random fractional shift along the two cell directions not dominated by ``axis``.

    For ``axis == 2``, uses ``cell[0]`` and ``cell[1]``. Uses ``[0, 1)`` fractions.
    If cluster_radius is provided, biases placement towards slab atoms.

    Args:
        slab: The slab atoms
        axis: Surface normal axis
        rng: Random number generator
        cluster_radius: Approximate radius of the cluster (optional)

    Returns:
        Shift vector
    """
    if cluster_radius is not None and cluster_radius > 0:
        return _in_plane_translation_near_slab_atom(slab, axis, rng, cluster_radius)

    cell = slab.get_cell()
    u, v = rng.random(), rng.random()
    if axis == 0:
        shift = u * cell[1] + v * cell[2]
    elif axis == 1:
        shift = u * cell[0] + v * cell[2]
    else:
        shift = u * cell[0] + v * cell[1]
    return np.asarray(shift, dtype=float)


def combine_slab_adsorbate(slab: Atoms, adsorbate: Atoms) -> Atoms:
    """Concatenate slab and adsorbate; adsorbate cell/pbc are replaced by slab's."""
    ads = adsorbate.copy()
    ads.set_cell(slab.get_cell())
    ads.set_pbc(slab.get_pbc())
    return slab.copy() + ads


def _random_rotation_matrix(rng: Generator) -> np.ndarray:
    """Return a uniformly random 3D rotation matrix."""
    rotation_axis = rng.standard_normal(3)
    rotation_axis /= np.linalg.norm(rotation_axis)
    rotation_angle = float(rng.uniform(0.0, 2.0 * np.pi))
    return _generate_rotation_matrix(rotation_axis, rotation_angle)


def _place_cluster_above_slab(
    cluster_positions: np.ndarray,
    slab: Atoms,
    slab_top: float,
    axis: int,
    rng: Generator,
    config: SurfaceSystemConfig,
    cluster_atomic_numbers: np.ndarray | None = None,
) -> np.ndarray:
    """Rotate/translate centered cluster positions into a deposited position.

    Args:
        cluster_positions: Centered cluster positions (will be rotated/translated).
        slab: The slab atoms.
        slab_top: Maximum z-coordinate of slab atoms along surface normal axis.
        axis: Surface normal axis index (0, 1, or 2).
        rng: Random number generator.
        config: Surface system configuration.
        cluster_atomic_numbers: Atomic numbers of cluster atoms. If provided,
            used to calculate covalent radii for connectivity-based placement.
    """
    rotated_positions = cluster_positions @ _random_rotation_matrix(rng).T

    # Estimate cluster radius for biasing placement towards slab atoms
    cluster_radius = float(np.max(np.linalg.norm(rotated_positions, axis=1)))

    translated_positions = rotated_positions + _in_plane_translation(
        slab, axis, rng, cluster_radius
    )

    # Calculate minimum height needed for connectivity.
    # The cluster extends from cluster_min to cluster_max along the axis.
    # To ensure connectivity, we need at least one cluster atom to be within
    # the connectivity threshold of a slab atom. The slab top is at slab_top,
    # and slab atoms can be slightly below this (e.g., in lower layers).
    # We use a conservative estimate: place the cluster so that its bottom
    # is at most (connectivity_threshold - cluster_extent) above the slab top,
    # where cluster_extent is the distance from the cluster center to its bottom.
    # This ensures that even with some randomness in placement, connectivity is likely.
    from scgo.initialization.geometry_helpers import get_covalent_radius

    # Use connectivity factor from config, falling back to default
    cf = config.structure_connectivity_factor

    # Get representative covalent radius for connectivity calculation
    # For the slab, use the first atom's radius (typically all same element)
    slab_symbols = slab.get_chemical_symbols()
    slab_radius = (
        get_covalent_radius(slab_symbols[0]) if slab_symbols else 1.36
    )  # fallback for Pt

    # For the cluster, use actual atomic numbers if provided, otherwise estimate
    if cluster_atomic_numbers is not None and len(cluster_atomic_numbers) > 0:
        # Use the maximum covalent radius in the cluster to ensure connectivity
        # for the largest atoms (which are most likely to be farthest from slab)
        from ase.data import atomic_numbers as ase_atomic_numbers

        # Build reverse mapping from atomic number to symbol
        number_to_symbol = {v: k for k, v in ase_atomic_numbers.items()}

        unique_atomic_numbers = set(cluster_atomic_numbers)
        cluster_radii = [
            get_covalent_radius(number_to_symbol.get(int(z), str(int(z))))
            for z in unique_atomic_numbers
        ]
        cluster_radius_est = max(cluster_radii) if cluster_radii else 1.36
    else:
        # Fallback to conservative estimate
        cluster_radius_est = 1.36  # Use Pt radius as default (more realistic than 1.0)

    # Use the sum of radii (not min) for the connectivity threshold
    # This ensures we're using the actual bond distance for the pair
    connectivity_threshold = cf * (slab_radius + cluster_radius_est)

    # Cluster extent below its center (cluster is centered at 0)
    cluster_min = float(np.min(rotated_positions[:, axis]))

    # To ensure connectivity, we want the closest approach between any cluster atom
    # and any slab atom to be within the connectivity threshold.
    # The slab top is at slab_top, and slab atoms can be up to ~2 Å below this.
    # A conservative approach: place the cluster so that its bottom is within
    # (connectivity_threshold - 2.0) of the slab top, ensuring overlap with slab atom positions.
    # But we also need to respect the user-specified height range.

    # Use a height that ensures connectivity while respecting the config range
    # Place the cluster bottom at a fraction of the connectivity threshold above slab top
    # This ensures atoms are close enough for bonding
    target_height_above_slab = min(
        config.adsorption_height_max,
        max(config.adsorption_height_min, connectivity_threshold * 0.4),
    )

    sampled_height = float(
        rng.uniform(config.adsorption_height_min, config.adsorption_height_max)
    )

    # Use the minimum of sampled height and target height to ensure connectivity
    # But don't go below the minimum
    effective_height = max(
        config.adsorption_height_min, min(sampled_height, target_height_above_slab)
    )

    translated_positions[:, axis] += slab_top + effective_height - cluster_min
    return translated_positions


def create_deposited_cluster(
    composition: Sequence[str],
    slab: Atoms,
    blmin: dict,
    rng: Generator,
    config: SurfaceSystemConfig,
    previous_search_glob: str = "**/*.db",
    adsorbate_definition: AdsorbateDefinition | None = None,
    adsorbate_fragment_template: Atoms | None = None,
    cluster_adsorbate_config: ClusterAdsorbateConfig | None = None,
) -> Atoms | None:
    """One adsorbate+slab structure, or None if placement fails.

    For non-adsorbate runs: build one gas-phase cluster for ``composition``, then
    place above slab. For adsorbate runs: build hierarchical core+fragment first.
    """
    axis = config.surface_normal_axis
    slab_top = slab_surface_extreme(slab, axis, upper=True)
    from scgo.initialization import create_initial_cluster

    for _ in range(config.max_placement_attempts):
        if adsorbate_definition is None:
            cluster_seed = create_initial_cluster(
                list(composition),
                vacuum=config.cluster_init_vacuum,
                rng=rng,
                previous_search_glob=previous_search_glob,
                mode=config.init_mode,
            )
        else:
            if adsorbate_fragment_template is None:
                raise ValueError(
                    "create_deposited_cluster requires adsorbate_fragment_template "
                    "for hierarchical adsorbate initialization."
                )
            core_symbols = [
                str(s) for s in adsorbate_definition.get("core_symbols", [])
            ]
            if len(core_symbols) == 0:
                cluster_seed = adsorbate_fragment_template.copy()
            else:
                cluster_seed = build_hierarchical_core_fragment_cluster(
                    composition,
                    adsorbate_definition,
                    rng,
                    previous_search_glob,
                    adsorbate_fragment_template,
                    cluster_adsorbate_config,
                    cluster_init_vacuum=config.cluster_init_vacuum,
                    init_mode=config.init_mode,
                    max_placement_attempts=config.max_placement_attempts,
                )
                if cluster_seed is None:
                    continue
        atomic_numbers = cluster_seed.get_atomic_numbers()
        cluster_positions = cluster_seed.get_positions().copy()

        cluster_positions -= np.mean(cluster_positions, axis=0)
        deposited_positions = _place_cluster_above_slab(
            cluster_positions=cluster_positions,
            slab=slab,
            slab_top=slab_top,
            axis=axis,
            rng=rng,
            config=config,
            cluster_atomic_numbers=atomic_numbers,
        )

        adsorbate = Atoms(
            numbers=atomic_numbers,
            positions=deposited_positions,
            cell=slab.get_cell(),
            pbc=slab.get_pbc(),
        )

        if atoms_too_close(adsorbate, blmin, use_tags=False):
            continue
        if atoms_too_close_two_sets(adsorbate, slab, blmin):
            continue

        combined = combine_slab_adsorbate(slab, adsorbate)
        n_slab = len(slab)
        skip_supported_cluster_check = bool(
            adsorbate_definition is not None
            and len([str(s) for s in adsorbate_definition.get("core_symbols", [])]) == 0
        )
        if skip_supported_cluster_check:
            return combined

        # Extract connectivity_factor: prefer cluster_adsorbate_config, then surface_config, then default
        connectivity_factor = CONNECTIVITY_FACTOR
        if cluster_adsorbate_config is not None:
            connectivity_factor = cluster_adsorbate_config.structure_connectivity_factor
        elif (
            hasattr(config, "structure_connectivity_factor")
            and config.structure_connectivity_factor is not None
        ):
            connectivity_factor = config.structure_connectivity_factor

        ok, err = validate_supported_cluster_deposit(
            combined,
            n_slab,
            surface_normal_axis=config.surface_normal_axis,
            use_mic=bool(config.comparator_use_mic),
            connectivity_factor=connectivity_factor,
        )
        if not ok:
            logger.debug(
                "create_deposited_cluster: rejected by supported-cluster check: %s", err
            )
            continue
        return combined

    logger.warning(
        "create_deposited_cluster: exceeded max_placement_attempts=%s",
        config.max_placement_attempts,
    )
    return None


def create_deposited_cluster_batch(
    composition: Sequence[str],
    slab: Atoms,
    blmin: dict,
    n_structures: int,
    rng: Generator,
    config: SurfaceSystemConfig,
    *,
    previous_search_glob: str = "**/*.db",
    n_jobs: int = 1,
    adsorbate_definition: AdsorbateDefinition | None = None,
    adsorbate_fragment_template: Atoms | None = None,
    cluster_adsorbate_config: ClusterAdsorbateConfig | None = None,
) -> list[Atoms]:
    """Generate multiple deposited structures (sequential or threaded)."""
    if n_structures <= 0:
        return []

    max_attempts = max(n_structures * 50, config.max_placement_attempts)

    if n_jobs == 1:
        out: list[Atoms] = []
        attempts = 0
        while len(out) < n_structures and attempts < max_attempts:
            attempts += 1
            child_rng = np.random.default_rng(
                rng.integers(0, 2**63 - 1, dtype=np.int64)
            )
            struct = create_deposited_cluster(
                composition,
                slab,
                blmin,
                child_rng,
                config,
                previous_search_glob=previous_search_glob,
                adsorbate_definition=adsorbate_definition,
                adsorbate_fragment_template=adsorbate_fragment_template,
                cluster_adsorbate_config=cluster_adsorbate_config,
            )
            if struct is not None:
                out.append(struct)
        if len(out) < n_structures:
            raise RuntimeError(
                f"Could only generate {len(out)} of {n_structures} deposited structures; "
                "try widening height range or increasing max_placement_attempts."
            )
        return out

    # Parallel: precompute deterministic per-task seeds on the main thread.
    per_worker_limit = max(config.max_placement_attempts, 50)
    task_seeds = [
        int(rng.integers(0, 2**63 - 1, dtype=np.int64)) for _ in range(n_structures)
    ]

    def _build_structure_with_seed(task_seed: int) -> Atoms:
        task_rng = np.random.default_rng(task_seed)
        for _ in range(per_worker_limit):
            child_rng = np.random.default_rng(
                task_rng.integers(0, 2**63 - 1, dtype=np.int64)
            )
            structure = create_deposited_cluster(
                composition,
                slab,
                blmin,
                child_rng,
                config,
                previous_search_glob=previous_search_glob,
                adsorbate_definition=adsorbate_definition,
                adsorbate_fragment_template=adsorbate_fragment_template,
                cluster_adsorbate_config=cluster_adsorbate_config,
            )
            if structure is not None:
                return structure
        raise RuntimeError(
            "Could not generate deposited structure in parallel worker; "
            "try widening height range or increasing max_placement_attempts."
        )

    workers = min(n_structures, resolve_n_jobs_to_workers(n_jobs))
    _warmup_ase_spacegroup_cache()
    ordered_results: list[Atoms | None] = [None] * n_structures
    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="scgo_deposit"
    ) as ex:
        futures = {
            ex.submit(_build_structure_with_seed, seed): idx
            for idx, seed in enumerate(task_seeds)
        }
        for future in as_completed(futures):
            ordered_results[futures[future]] = future.result()
    if any(result is None for result in ordered_results):
        raise RuntimeError("Parallel batch returned too few structures")
    return [result for result in ordered_results if result is not None]
