"""Place molecular fragments near the surface of a gas-phase cluster."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase_ga.utilities import atoms_too_close_two_sets
from numpy.random import Generator

from scgo.cluster_adsorbate.combine import combine_core_adsorbate
from scgo.cluster_adsorbate.config import ClusterAdsorbateConfig
from scgo.cluster_adsorbate.geometry import (
    outermost_point_along_normal,
    random_rotation_matrix,
    random_spin_about_normal,
    random_unit_vector,
    rotation_matrix_a_to_b,
)
from scgo.cluster_adsorbate.sites import (
    SiteType,
    SurfaceSiteCandidate,
    compute_surface_site_candidates,
)
from scgo.cluster_adsorbate.validation import validate_combined_cluster_structure
from scgo.exceptions import SCGOValidationError
from scgo.initialization.atomic_radii import build_blmin_from_zs, get_covalent_radius
from scgo.initialization.initialization_config import (
    CONNECTIVITY_FACTOR,
    PLACEMENT_RELAXATION_FACTOR,
)
from scgo.initialization.steric_scoring import steric_deficit_two_sets
from scgo.utils.logging import get_logger

logger = get_logger(__name__)

_RANKED_CANDIDATES_PER_ATTEMPT = 12


_BLMIN_RATIO_FLOOR = 0.55
_MIN_DISTANCE_FACTOR_FLOOR = 0.3


@dataclass(frozen=True)
class _PlacementTrial:
    site_type: str
    selected_type: SiteType | None
    n_dir: np.ndarray
    anchor_surf: np.ndarray
    height: float


def radii_derived_height_bounds(
    fragment_template: Atoms,
    core: Atoms,
    anchor_index: int,
) -> tuple[float, float]:
    """Heuristic height range from covalent radii of anchor and core atoms."""
    anchor_sym = fragment_template.get_chemical_symbols()[anchor_index]
    r_anchor = get_covalent_radius(anchor_sym)
    r_core = max(get_covalent_radius(s) for s in core.get_chemical_symbols())
    base = (r_anchor + r_core) * CONNECTIVITY_FACTOR
    return base * 0.5, base * 1.2


def _compute_effective_placement_params(
    config: ClusterAdsorbateConfig,
    attempt_ratio: float,
    fragment_template: Atoms,
    core: Atoms,
    anchor_index: int,
) -> tuple[float, float, float, float]:
    """Progressively relax placement thresholds as attempts are exhausted."""
    derived_min, derived_max = radii_derived_height_bounds(
        fragment_template, core, anchor_index
    )
    relax = float(np.clip(attempt_ratio, 0.0, 1.0)) * PLACEMENT_RELAXATION_FACTOR
    height_min = max(
        min(config.height_min, derived_min) * (1.0 - 0.2 * relax),
        derived_min * 0.85,
    )
    height_max = max(
        height_min,
        max(config.height_max, derived_max) * (1.0 + 0.25 * relax),
    )
    blmin_ratio = max(config.blmin_ratio * (1.0 - 0.15 * relax), _BLMIN_RATIO_FLOOR)
    min_dist_factor = max(
        config.structure_min_distance_factor * (1.0 - 0.1 * relax),
        _MIN_DISTANCE_FACTOR_FLOOR,
    )
    return height_min, height_max, blmin_ratio, min_dist_factor


def _select_site_type(
    available_types: list[SiteType],
    rng: Generator,
    within_structure_site_counts: dict[str, int] | None,
    batch_site_counts: dict[str, int] | None,
) -> SiteType:
    """Select site class with anti-repetition weighting."""
    if len(available_types) == 1:
        return available_types[0]
    weights: list[float] = []
    for st in available_types:
        local = (
            0
            if within_structure_site_counts is None
            else int(within_structure_site_counts.get(st, 0))
        )
        batch = 0 if batch_site_counts is None else int(batch_site_counts.get(st, 0))
        weights.append(1.0 / (1.0 + local + batch))
    probs = np.asarray(weights, dtype=float)
    probs /= float(np.sum(probs))
    idx = int(rng.choice(len(available_types), p=probs))
    return available_types[idx]


def blmin_for_core_and_fragment(
    core: Atoms, fragment: Atoms, blmin_ratio: float
) -> dict:
    """Minimum interatomic distances for all element pairs in core ∪ fragment."""
    zs = list(core.numbers) + list(fragment.numbers)
    return build_blmin_from_zs(zs, ratio=blmin_ratio)


def _build_fragment_positions(
    base_frag_pos: np.ndarray,
    n_frag: int,
    n_dir: np.ndarray,
    target: np.ndarray,
    bond_axis: tuple[int, int] | None,
    rng: Generator,
    random_spin: bool,
) -> np.ndarray | None:
    pos = base_frag_pos.copy()
    if n_frag > 1:
        if bond_axis is not None:
            ia, ja = bond_axis
            v = pos[ja] - pos[ia]
            vn = float(np.linalg.norm(v))
            if vn < 1e-10:
                return None
            v = v / vn
            r_align = rotation_matrix_a_to_b(v, n_dir)
            pos = (r_align @ pos.T).T
            if random_spin:
                r_spin = random_spin_about_normal(rng, n_dir)
                pos = (r_spin @ pos.T).T
        else:
            r_rand = random_rotation_matrix(rng)
            pos = (r_rand @ pos.T).T
    pos += target
    return pos


def _generate_placement_trials(
    rng: Generator,
    site_candidates: dict[SiteType, list[SurfaceSiteCandidate]],
    flat_candidates: list[SurfaceSiteCandidate],
    site_pos: np.ndarray,
    relative_site_pos: np.ndarray,
    height_min: float,
    height_max: float,
    within_structure_site_counts: dict[str, int] | None,
    batch_site_counts: dict[str, int] | None,
    n_trials: int,
) -> list[_PlacementTrial]:
    trials: list[_PlacementTrial] = []
    for _ in range(n_trials):
        if flat_candidates:
            available_types: list[SiteType] = [
                st for st in ("vertex", "edge", "facet") if site_candidates[st]
            ]
            selected_type = _select_site_type(
                available_types=available_types,
                rng=rng,
                within_structure_site_counts=within_structure_site_counts,
                batch_site_counts=batch_site_counts,
            )
            chosen = site_candidates[selected_type][
                int(rng.integers(0, len(site_candidates[selected_type])))
            ]
            n_dir = chosen.normal
            anchor_surf = chosen.anchor
            site_type = selected_type
        else:
            selected_type = None
            site_type = "directional_fallback"
            n_dir = random_unit_vector(rng)
            anchor_surf = outermost_point_along_normal(
                site_pos, relative_site_pos, n_dir
            )
        height = float(rng.uniform(height_min, height_max))
        trials.append(
            _PlacementTrial(
                site_type=site_type,
                selected_type=selected_type,
                n_dir=n_dir,
                anchor_surf=anchor_surf,
                height=height,
            )
        )
    return trials


def place_fragment_on_cluster(
    core: Atoms,
    fragment_template: Atoms,
    rng: Generator,
    config: ClusterAdsorbateConfig | None = None,
    *,
    anchor_index: int = 0,
    bond_axis: tuple[int, int] | None = None,
    within_structure_site_counts: dict[str, int] | None = None,
    batch_site_counts: dict[str, int] | None = None,
    placement_metadata: dict[str, str] | None = None,
    site_core: Atoms | None = None,
    clash_atoms: Atoms | None = None,
) -> Atoms | None:
    """Rigidly place a gas-phase fragment with random orientation near the cluster."""
    if config is None:
        config = ClusterAdsorbateConfig()
    if len(core) == 0:
        raise SCGOValidationError("core must contain at least one atom")
    n_frag = len(fragment_template)
    if n_frag == 0:
        raise SCGOValidationError("fragment_template must contain at least one atom")
    if not (0 <= anchor_index < n_frag):
        raise SCGOValidationError(
            f"anchor_index={anchor_index} invalid for fragment with {n_frag} atoms"
        )
    if bond_axis is not None:
        i, j = bond_axis
        if not (0 <= i < n_frag and 0 <= j < n_frag) or i == j:
            raise SCGOValidationError(
                f"bond_axis={bond_axis} invalid for this fragment"
            )

    site_atoms = site_core if site_core is not None else core
    clash_target = clash_atoms if clash_atoms is not None else core
    if len(site_atoms) == 0:
        raise SCGOValidationError("site_core must contain at least one atom")
    if len(clash_target) == 0:
        raise SCGOValidationError("clash_atoms must contain at least one atom")

    site_pos = site_atoms.get_positions()
    com = np.mean(site_pos, axis=0)
    relative_site_pos = site_pos - com
    symbols = fragment_template.get_chemical_symbols()
    site_candidates = compute_surface_site_candidates(site_atoms)
    flat_candidates = [
        candidate for entries in site_candidates.values() for candidate in entries
    ]

    base_frag_pos = fragment_template.get_positions().astype(float).copy()
    base_frag_pos -= base_frag_pos[anchor_index]
    clash_positions = clash_target.get_positions()
    clash_numbers = clash_target.get_atomic_numbers()
    frag_numbers = fragment_template.get_atomic_numbers()
    blmin_zs = list(clash_numbers) + list(frag_numbers)
    blmin_base = build_blmin_from_zs(blmin_zs, ratio=1.0)

    for attempt in range(config.max_placement_attempts):
        attempt_ratio = attempt / max(1, config.max_placement_attempts - 1)
        height_min, height_max, blmin_ratio, min_dist_factor = (
            _compute_effective_placement_params(
                config, attempt_ratio, fragment_template, core, anchor_index
            )
        )
        if blmin_ratio == 1.0:
            blmin = blmin_base
        else:
            blmin = {pair: dist * blmin_ratio for pair, dist in blmin_base.items()}

        trials = _generate_placement_trials(
            rng,
            site_candidates,
            flat_candidates,
            site_pos,
            relative_site_pos,
            height_min,
            height_max,
            within_structure_site_counts,
            batch_site_counts,
            _RANKED_CANDIDATES_PER_ATTEMPT,
        )

        ranked: list[tuple[float, _PlacementTrial, np.ndarray]] = []
        for trial in trials:
            target = trial.anchor_surf + trial.height * trial.n_dir
            pos = _build_fragment_positions(
                base_frag_pos,
                n_frag,
                trial.n_dir,
                target,
                bond_axis,
                rng,
                config.random_spin_about_normal,
            )
            if pos is None:
                continue
            score = steric_deficit_two_sets(
                pos,
                frag_numbers,
                clash_positions,
                clash_numbers,
                blmin,
            )
            ranked.append((score, trial, pos))

        ranked.sort(key=lambda item: item[0])

        accepted: tuple[_PlacementTrial, np.ndarray, Atoms] | None = None
        for _score, trial, pos in ranked[:3]:
            frag = Atoms(
                symbols=symbols,
                positions=pos,
                cell=core.get_cell(),
                pbc=core.get_pbc(),
            )
            if atoms_too_close_two_sets(frag, clash_target, blmin):
                continue
            accepted = (trial, pos, frag)
            break

        if accepted is None:
            continue

        trial, _pos, frag = accepted
        if config.validate_combined_structure:
            trial_combined = combine_core_adsorbate(clash_target, frag)
            ok, _msg = validate_combined_cluster_structure(
                trial_combined,
                min_distance_factor=min_dist_factor,
                connectivity_factor=config.structure_connectivity_factor,
                check_clashes=config.structure_check_clashes,
                check_connectivity=config.structure_check_connectivity,
            )
            if not ok:
                continue

        if trial.selected_type is not None and within_structure_site_counts is not None:
            within_structure_site_counts[trial.selected_type] = (
                int(within_structure_site_counts.get(trial.selected_type, 0)) + 1
            )
        if placement_metadata is not None:
            placement_metadata["site_type"] = trial.site_type
        return frag

    logger.warning(
        "place_fragment_on_cluster: exceeded max_placement_attempts=%s",
        config.max_placement_attempts,
    )
    return None
