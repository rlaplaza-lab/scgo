"""Place molecular fragments near the surface of a gas-phase cluster."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms
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
    get_or_compute_surface_site_candidates,
)
from scgo.cluster_adsorbate.validation import validate_combined_cluster_structure
from scgo.exceptions import SCGOValidationError
from scgo.initialization.atomic_radii import build_blmin_from_zs, get_covalent_radius
from scgo.initialization.geometry_helpers import _should_check_connectivity
from scgo.initialization.initialization_config import BLMIN_RATIO_DEFAULT
from scgo.initialization.steric_scoring import steric_deficit_two_sets
from scgo.system_types.connectivity_factor import (
    ConnectivityFactorInput,
    NormalizedConnectivityFactor,
    normalize_connectivity_factor,
    pair_bond_threshold,
)
from scgo.utils.logging import get_logger

logger = get_logger(__name__)

_RANKED_CANDIDATES_PER_ATTEMPT = 12


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
    *,
    site_index: int | None = None,
    connectivity_factor: ConnectivityFactorInput
    | NormalizedConnectivityFactor
    | None = None,
) -> tuple[float, float]:
    """Height window from the anchor atom and one core site atom.

    The floor is the steric scale times the covalent sum. The ceiling is that
    pair's bond threshold. When ``site_index`` is omitted the core atom farthest
    from the center of mass is used.
    """
    if len(core) == 0:
        raise SCGOValidationError("core must contain at least one atom")
    anchor_sym = fragment_template.get_chemical_symbols()[anchor_index]
    r_anchor = get_covalent_radius(anchor_sym)
    if site_index is None:
        positions = core.get_positions()
        com = np.mean(positions, axis=0)
        site_index = int(np.argmax(np.linalg.norm(positions - com, axis=1)))
    site_sym = core.get_chemical_symbols()[int(site_index)]
    r_site = get_covalent_radius(site_sym)
    spec = normalize_connectivity_factor(connectivity_factor)
    ceiling = pair_bond_threshold(r_anchor, r_site, anchor_sym, site_sym, spec)
    floor = (r_anchor + r_site) * BLMIN_RATIO_DEFAULT
    if ceiling < floor:
        ceiling = floor
    return float(floor), float(ceiling)


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
    site_atoms: Atoms,
    site_pos: np.ndarray,
    relative_site_pos: np.ndarray,
    fragment_template: Atoms,
    anchor_index: int,
    config: ClusterAdsorbateConfig,
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
        site_index = int(np.argmin(np.linalg.norm(site_pos - anchor_surf, axis=1)))
        floor, ceiling = radii_derived_height_bounds(
            fragment_template,
            site_atoms,
            anchor_index,
            site_index=site_index,
            connectivity_factor=config.structure_connectivity_factor,
        )
        height_min = (
            floor if config.height_min is None else max(floor, float(config.height_min))
        )
        height_max = (
            ceiling
            if config.height_max is None
            else min(ceiling, float(config.height_max))
        )
        if height_max < height_min:
            height_max = height_min
        height = (
            float(rng.uniform(height_min, height_max))
            if height_max > height_min
            else height_min
        )
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


def infer_fragment_bond_axis(
    fragment: Atoms,
    anchor_index: int,
    connectivity_factor: ConnectivityFactorInput
    | NormalizedConnectivityFactor
    | None = None,
) -> tuple[int, int] | None:
    """Return ``(anchor, neighbor)`` when the anchor has exactly one bonded neighbor.

    OH, CO, and the O-O bond of OOH qualify. Water and other multi-coordinate
    anchors return ``None``.
    """
    if len(fragment) < 2:
        return None
    spec = normalize_connectivity_factor(connectivity_factor)
    symbols = fragment.get_chemical_symbols()
    positions = fragment.get_positions()
    anchor_symbol = symbols[anchor_index]
    anchor_radius = get_covalent_radius(anchor_symbol)
    anchor_pos = positions[anchor_index]
    neighbors: list[int] = []
    for index, symbol in enumerate(symbols):
        if index == anchor_index:
            continue
        distance = float(np.linalg.norm(positions[index] - anchor_pos))
        threshold = pair_bond_threshold(
            anchor_radius,
            get_covalent_radius(symbol),
            anchor_symbol,
            symbol,
            spec,
        )
        if distance <= threshold:
            neighbors.append(index)
    if len(neighbors) != 1:
        return None
    return anchor_index, neighbors[0]


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
    site_candidates: dict[SiteType, list[SurfaceSiteCandidate]] | None = None,
    uses_surface: bool = False,
) -> Atoms | None:
    """Rigidly place a fragment at a ranked adsorption site near the cluster.

    Each attempt scores several site/height trials by steric deficit and takes
    the first clash-free trial among the best-ranked ones. Multi-atom fragments
    are oriented along ``bond_axis`` when given (aligned to the site normal,
    optionally spun about it). When ``bond_axis`` is omitted,
    :func:`infer_fragment_bond_axis` supplies one if the anchor has a single
    bonded neighbor; otherwise the fragment is randomly rotated.

    Returns:
        The placed fragment, or ``None`` if every attempt fails.

    Raises:
        SCGOValidationError: If ``core``, ``fragment_template``, ``site_core``
            or ``clash_atoms`` is empty, or if ``anchor_index``/``bond_axis``
            is out of range for the fragment.
    """
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
    else:
        bond_axis = infer_fragment_bond_axis(
            fragment_template,
            anchor_index,
            config.structure_connectivity_factor,
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
    if site_candidates is None:
        site_candidates = get_or_compute_surface_site_candidates(site_atoms)
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

    blmin_ratio = max(float(config.blmin_ratio), BLMIN_RATIO_DEFAULT)
    min_dist_factor = max(
        float(config.structure_min_distance_factor), BLMIN_RATIO_DEFAULT
    )
    if blmin_ratio == 1.0:
        blmin = blmin_base
    else:
        blmin = {pair: dist * blmin_ratio for pair, dist in blmin_base.items()}

    for _attempt in range(config.max_placement_attempts):
        trials = _generate_placement_trials(
            rng,
            site_candidates,
            flat_candidates,
            site_atoms,
            site_pos,
            relative_site_pos,
            fragment_template,
            anchor_index,
            config,
            within_structure_site_counts,
            batch_site_counts,
            _RANKED_CANDIDATES_PER_ATTEMPT,
        )

        cell = core.get_cell()
        pbc = core.get_pbc()
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
            score, is_hard_clash = steric_deficit_two_sets(
                pos,
                frag_numbers,
                clash_positions,
                clash_numbers,
                blmin,
                cell=cell,
                pbc=pbc,
            )
            if is_hard_clash:
                continue
            ranked.append((score, trial, pos))

        ranked.sort(key=lambda item: item[0])
        if not ranked:
            continue

        _score, trial, pos = ranked[0]
        frag = Atoms(
            symbols=symbols,
            positions=pos,
            cell=cell,
            pbc=pbc,
        )
        if config.validate_combined_structure:
            trial_combined = combine_core_adsorbate(clash_target, frag)
            ok, _msg = validate_combined_cluster_structure(
                trial_combined,
                min_distance_factor=min_dist_factor,
                connectivity_factor=config.structure_connectivity_factor,
                check_clashes=config.structure_check_clashes,
                check_connectivity=(
                    config.structure_check_connectivity
                    and _should_check_connectivity(
                        trial_combined, uses_surface=uses_surface
                    )
                ),
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

    logger.debug(
        "Reached max_placement_attempts=%s in place_fragment_on_cluster (bond_axis=%s)",
        config.max_placement_attempts,
        bond_axis,
    )
    return None
