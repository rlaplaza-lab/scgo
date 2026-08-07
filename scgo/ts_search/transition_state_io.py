"""I/O helpers for transition-state search outputs."""

from __future__ import annotations

import json
import math
import os
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import write as ase_write

from scgo.constants import DEFAULT_COMPARATOR_TOL, DEFAULT_ENERGY_TOLERANCE
from scgo.database import (
    extract_minima_from_database_file,
)
from scgo.database.discovery import list_discovered_db_paths_with_run
from scgo.metadata.atoms import set_tags
from scgo.metadata.provenance import output_json_provenance
from scgo.surface.validation import (
    validate_stored_mobile_partition_metadata,
    validate_stored_slab_adsorbate_metadata,
)
from scgo.ts_search.ts_statistics import compute_ts_statistics
from scgo.utils.comparators import PureInteratomicDistanceComparator
from scgo.utils.helpers import copy_atoms, get_cluster_formula, validate_pair_id
from scgo.utils.logging import get_logger

from .transition_state import (
    _permute_atoms_block_to_match,
    calculate_structure_similarity,
    minima_provenance_dict,
)

# Absolute ceiling for adsorbate pair oversample before IDPP re-rank.
_ADSORBATE_PAIR_OVERSAMPLE_CAP = 50


def adsorbate_pair_select_cap(max_pairs: int) -> int:
    """Return oversample size: ``min(max_pairs * 10, max(max_pairs, 50))``."""
    mp = int(max_pairs)
    return min(mp * 10, max(mp, _ADSORBATE_PAIR_OVERSAMPLE_CAP))


def load_minima_by_composition(
    base_dir: str,
    composition: list[str] | None = None,
    prefer_final_unique: bool = True,
) -> dict[str, list[tuple[float, Atoms]]]:
    """Load minima from all runs, optionally filtered by composition.

    Scans base_dir for run_*/ subdirectories containing *.db database files.
    Extracts minima from all databases and groups by chemical formula.

    By default only ``final_unique_minimum`` rows are loaded (canonical GO
    output). Structural deduplication across runs is left to callers (e.g.
    :func:`scgo.utils.helpers.filter_unique_minima` in TS search).

    Args:
        base_dir: Root directory containing run_*/ subdirectories.
        composition: Optional list of atomic symbols to filter results (e.g., ["Pt", "Au"]).
            If provided, only minima matching this composition are returned.
        prefer_final_unique: If True (default), only final-tagged minima; set False
            to load all relaxed non-TS rows.

    Returns:
        Dictionary mapping composition formula strings to lists of (energy, Atoms) tuples,
        each sorted by energy (lowest first). Returns empty dict if no minima found.

    Example:
        >>> minima = load_minima_by_composition("Pt3_searches", ["Pt", "Pt", "Pt"])
        >>> list(minima.keys())
        ['Pt3']
    """
    logger = get_logger(__name__)

    if not os.path.exists(base_dir):
        logger.warning("Output directory does not exist: %s", base_dir)
        return {}

    minima_by_formula: dict[str, list[tuple[float, Atoms]]] = {}

    target_formula = get_cluster_formula(composition) if composition else None

    db_files_with_run = list_discovered_db_paths_with_run(
        base_dir, composition=composition, use_cache=True
    )

    for db_file, run_id in db_files_with_run:
        try:
            try:
                db_relpath = os.path.relpath(db_file, base_dir)
            except (OSError, ValueError):
                db_relpath = os.path.basename(db_file)
            # When prefer_final_unique=True, require_final=True so we only load
            # GO's canonical final unique minima (DB rows tagged final_unique_minimum).
            minima = extract_minima_from_database_file(
                db_file,
                run_id=run_id,
                require_final=prefer_final_unique,
                source_db_relpath=db_relpath,
            )

            if not minima:
                continue

            # Get composition from first structure
            first_atoms = minima[0][1]
            symbols = first_atoms.get_chemical_symbols()
            formula = get_cluster_formula(symbols)

            # Filter by target composition if specified
            if target_formula and formula != target_formula:
                continue

            # Add to results with run_id in provenance
            if formula not in minima_by_formula:
                minima_by_formula[formula] = []

            for energy, atoms in minima:
                atoms_copy = copy_atoms(atoms)
                set_tags(
                    atoms_copy,
                    run_id=run_id,
                    source_db=os.path.basename(db_file),
                    source_db_relpath=db_relpath,
                )
                validate_stored_slab_adsorbate_metadata(atoms_copy)
                validate_stored_mobile_partition_metadata(atoms_copy)
                minima_by_formula[formula].append((energy, atoms_copy))

        except (ValueError, OSError) as e:
            logger.warning(
                "Failed to load minima from %s: %s: %s",
                db_file,
                type(e).__name__,
                e,
            )

    # Sort each formula's minima by energy
    for formula in minima_by_formula:
        minima_by_formula[formula] = sorted(
            minima_by_formula[formula], key=lambda x: x[0]
        )

    return minima_by_formula


def _core_rms_displacement(
    atoms_i: Atoms,
    atoms_j: Atoms,
    *,
    n_slab: int,
    n_core: int,
    use_mic: bool,
) -> float:
    """RMS Cartesian displacement of the core block after spatial matching.

    Same-element core permutations (common from GA) are resolved with a
    Hungarian spatial match on copies so the gate is order-invariant. Minima
    atoms are never mutated.
    """
    if n_core <= 0:
        return 0.0
    i0 = max(0, int(n_slab))
    i1 = i0 + int(n_core)
    if i1 > len(atoms_i) or i1 > len(atoms_j):
        return 0.0
    # Array views → thin Atoms (avoid full ASE slice copies with constraints/info).
    pos_i = np.asarray(atoms_i.get_positions()[i0:i1], dtype=float)
    pos_j = np.asarray(atoms_j.get_positions()[i0:i1], dtype=float)
    nums_i = np.asarray(atoms_i.numbers[i0:i1], dtype=int)
    nums_j = np.asarray(atoms_j.numbers[i0:i1], dtype=int)
    core_i = Atoms(numbers=nums_i, positions=pos_i, cell=atoms_i.cell, pbc=atoms_i.pbc)
    core_j = Atoms(numbers=nums_j, positions=pos_j, cell=atoms_j.cell, pbc=atoms_j.pbc)
    mic_cell = None
    mic_pbc = None
    if use_mic and bool(np.any(atoms_i.pbc)):
        mic_cell = np.asarray(atoms_i.cell.array, dtype=float)
        mic_pbc = np.asarray(atoms_i.pbc, dtype=bool)
    matched_pos, _matched_nums = _permute_atoms_block_to_match(
        core_i,
        core_j,
        mic_cell=mic_cell,
        mic_pbc=mic_pbc,
        method="spatial",
    )
    dlt = matched_pos - pos_i
    if mic_cell is not None and mic_pbc is not None:
        inv = np.linalg.inv(mic_cell)
        frac = dlt @ inv.T
        frac -= np.round(frac)
        dlt = frac @ mic_cell
    return float(np.sqrt(np.mean(np.sum(dlt * dlt, axis=1))))


def select_structure_pairs(
    minima: list[tuple[float, Atoms]],
    max_pairs: int | None = None,
    energy_gap_threshold: float | None = None,
    similarity_tolerance: float = DEFAULT_COMPARATOR_TOL,
    similarity_pair_cor_max: float = 0.1,
    surface_aware: bool = False,
    *,
    use_mic: bool,
    n_slab: int | None = None,
    max_endpoint_mismatch: float | None = None,
    adsorbate_aware: bool = False,
    n_core_mobile: int | None = None,
) -> list[tuple[int, int]]:
    """Select pairs of minima for TS calculations.

    Pairs nearby minima in energy space, using permutation-invariant structural
    comparison to avoid pairing very similar structures. When ``max_pairs`` is
    set, candidates are ranked by a physics-guided score (energy gap and
    structural dissimilarity) before taking the top N.

    Args:
        minima: List of (energy, Atoms) tuples, sorted by energy.
        max_pairs: Maximum number of pairs to generate. If None, generates all pairs.
            Default None.
        energy_gap_threshold: Only pair structures with energy gap below this threshold (eV).
            If None, pairs all structures. Default None.
        similarity_tolerance: Cumulative difference tolerance for structure comparison.
            Structures with cumulative difference below this value are considered too similar
            to pair. Default `DEFAULT_COMPARATOR_TOL` (tighter than GA duplicate detection).
        similarity_pair_cor_max: Maximum single distance difference tolerance.
            Default 0.1 Å (tighter than GA to ensure truly distinct structures).
        surface_aware: Use slightly looser scoring scales (slab / periodic systems).
        use_mic: MIC for distance/similarity geometry.
        n_slab: When set (from ``SurfaceSystemConfig.slab``), structural comparison
            uses only atoms ``n_slab:`` so pair selection ignores frozen slab motion.
        max_endpoint_mismatch: Optional Å geometric gate on comparator ``max_diff``.
            ``None`` disables the gate (bare-cluster default). When set (adsorbate
            presets), also enables pre-NEB clash and IDPP energy-profile checks.
        adsorbate_aware: Prefer modest core RMS and weight mismatch more heavily.
        n_core_mobile: Core atom count for adsorbate-aware core-RMS scoring.

    Returns:
        List of (index1, index2) tuples where index1 < index2, indicating which minima to pair.
    """
    logger = get_logger(__name__)
    mic = bool(use_mic)

    if len(minima) < 2:
        logger.info("Only %d minima, need at least 2 to pair", len(minima))
        return []

    scored_pairs: list[tuple[float, int, int]] = []
    n_skipped_similar = 0
    n_skipped_mismatch = 0
    slab_len = int(n_slab) if n_slab is not None else 0
    # Reuse one comparator when mobile counts are uniform (typical).
    shared_n_top = max(0, len(minima[0][1]) - slab_len)
    shared_comparator = PureInteratomicDistanceComparator(
        n_top=shared_n_top,
        tol=similarity_tolerance,
        pair_cor_max=similarity_pair_cor_max,
        mic=mic,
    )

    def _score_candidate(
        gap: float,
        cum_diff: float,
        max_diff: float,
        core_rms: float | None,
    ) -> float:
        """Return higher-is-better physics-guided priority score.

        Uses a compact blend of:
        - energy-gap proximity to a regime-dependent target window,
        - moderate structural dissimilarity preference,
        - endpoint mismatch penalty to avoid overly-discontinuous paths,
        - optional preference for modest core RMS (adsorbate systems).
        """
        # Surface systems often tolerate/require slightly larger endpoint deltas.
        gap_center = 0.45 if surface_aware else 0.30
        gap_width = 0.55 if surface_aware else 0.40
        gap_score = math.exp(-(((gap - gap_center) / max(1e-8, gap_width)) ** 2))

        if adsorbate_aware:
            # Prefer activated local hops: mid energy gaps, moderate endpoint
            # mismatch (~0.35–0.7 Å), and core RMS ~0.5 Å. Tiny-mismatch /
            # tiny-core pairs are often barrierless endothermic slides whose
            # IDPP bands are endpoint-max and CI-NEB climbs into junk saddles.
            mismatch_weight = 0.25
            distinct_weight = 0.20
            gap_weight = 0.25
            core_weight = 0.30
            cum_scale = 0.10 if surface_aware else 0.08
            gap_center = 0.55 if surface_aware else 0.50
            gap_width = 0.50 if surface_aware else 0.45
            gap_score = math.exp(-(((gap - gap_center) / max(1e-8, gap_width)) ** 2))
            # Peak distinctness above near-isomer noise.
            distinct_center = 0.12 if surface_aware else 0.10
            distinct_score = math.exp(
                -(((cum_diff - distinct_center) / max(1e-8, cum_scale)) ** 2)
            )
            mismatch_center = 0.55 if surface_aware else 0.50
            mismatch_width = 0.35 if surface_aware else 0.28
            mismatch_score = math.exp(
                -(((max_diff - mismatch_center) / max(1e-8, mismatch_width)) ** 2)
            )
        else:
            mismatch_scale = 0.45 if surface_aware else 0.35
            mismatch_weight = 0.15
            distinct_weight = 0.35
            gap_weight = 0.50
            core_weight = 0.0
            cum_scale = 0.12 if surface_aware else 0.09
            distinct_score = 1.0 - math.exp(-max(0.0, cum_diff) / max(1e-8, cum_scale))
            mismatch_score = math.exp(-max(0.0, max_diff) / max(1e-8, mismatch_scale))

        score = (
            gap_weight * gap_score
            + distinct_weight * distinct_score
            + mismatch_weight * mismatch_score
        )
        if core_weight > 0.0 and core_rms is not None:
            # Prefer core rearrangements that accompany real adsorbate hops.
            core_center = 0.50 if surface_aware else 0.55
            core_width = 0.45 if surface_aware else 0.35
            core_score = math.exp(
                -(((core_rms - core_center) / max(1e-8, core_width)) ** 2)
            )
            score += core_weight * core_score
        return score

    for i in range(len(minima)):
        for j in range(i + 1, len(minima)):
            energy_i, atoms_i = minima[i]
            energy_j, atoms_j = minima[j]

            # Energy gap filter
            gap = abs(energy_j - energy_i)
            if energy_gap_threshold is not None and gap > energy_gap_threshold:
                # Minima are energy-sorted; once the gap is too large, later j
                # for this i can only increase it.
                break

            # Permutation-invariant similarity filter
            try:
                cum_diff, max_diff, are_similar = calculate_structure_similarity(
                    atoms_i,
                    atoms_j,
                    tolerance=similarity_tolerance,
                    pair_cor_max=similarity_pair_cor_max,
                    use_mic=mic,
                    n_slab=n_slab,
                    comparator=shared_comparator,
                )

                if are_similar:
                    n_skipped_similar += 1
                    logger.debug(
                        "Skipping pair (%s, %s): structures too similar "
                        "(cum_diff=%.4f, max_diff=%.3f Å)",
                        i,
                        j,
                        cum_diff,
                        max_diff,
                    )
                    continue
                # Adsorbate near-isomers with tiny max displacement are usually
                # barrierless endothermic slides; climb cannot salvage them.
                if adsorbate_aware and float(max_diff) < 0.20:
                    n_skipped_similar += 1
                    logger.debug(
                        "Skipping pair (%s, %s): adsorbate hop too small "
                        "(max_diff=%.3f Å < 0.20 Å)",
                        i,
                        j,
                        max_diff,
                    )
                    continue
                if max_endpoint_mismatch is not None and float(max_diff) > float(
                    max_endpoint_mismatch
                ):
                    n_skipped_mismatch += 1
                    logger.debug(
                        "Skipping pair (%s, %s): endpoint mismatch too large "
                        "(max_diff=%.3f Å > %.3f Å)",
                        i,
                        j,
                        max_diff,
                        max_endpoint_mismatch,
                    )
                    continue
            except (ValueError, RuntimeError) as e:
                logger.warning(
                    f"Failed to calculate similarity for pair ({i}, {j}): {type(e).__name__}: {e}"
                )
                continue

            core_rms: float | None = None
            if adsorbate_aware and n_core_mobile is not None and int(n_core_mobile) > 0:
                core_rms = _core_rms_displacement(
                    atoms_i,
                    atoms_j,
                    n_slab=slab_len,
                    n_core=int(n_core_mobile),
                    use_mic=mic,
                )
                # Large core rearrangements rarely yield usable adsorbate NEBs.
                core_rms_limit = 2.0 if surface_aware else 1.5
                if core_rms > core_rms_limit:
                    n_skipped_mismatch += 1
                    logger.debug(
                        "Skipping pair (%s, %s): core RMS too large "
                        "(core_rms=%.3f Å > %.3f Å)",
                        i,
                        j,
                        core_rms,
                        core_rms_limit,
                    )
                    continue

            scored_pairs.append(
                (
                    _score_candidate(gap, float(cum_diff), float(max_diff), core_rms),
                    i,
                    j,
                )
            )

    if n_skipped_similar:
        logger.debug(
            "Pair selection: skipped %d too-similar candidate pairs",
            n_skipped_similar,
        )
    if n_skipped_mismatch:
        logger.debug(
            "Pair selection: skipped %d high-mismatch candidate pairs",
            n_skipped_mismatch,
        )

    if not scored_pairs:
        return []

    # Deterministic ordering: higher score first, then stable index tie-break.
    scored_pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    ranked_pairs = [(i, j) for _score, i, j in scored_pairs]
    if max_pairs is None:
        return ranked_pairs

    return ranked_pairs[:max_pairs]


def save_transition_state_results(
    ts_results: list[dict[str, Any]],
    output_dir: str,
    composition: list[str],
    run_context: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> str:
    """Save all transition state results to ``results_summary.json``.

    Args:
        ts_results: List of result dictionaries from find_transition_state().
        output_dir: TS results root directory where summary will be saved.
        composition: List of atomic symbols for the composition.
        run_context: Optional NEB/search context merged into the summary.
        run_id: Optional run ID for the current TS search invocation.

    Returns:
        Path to saved summary file.
    """
    logger = get_logger(__name__)

    os.makedirs(output_dir, exist_ok=True)

    formula = get_cluster_formula(composition)

    summary = output_json_provenance(extra=run_context or {})
    summary.update(
        {
            "composition": composition,
            "formula": formula,
            "num_total_pairs": len(ts_results),
            "num_successful": sum(1 for r in ts_results if r["status"] == "success"),
            "num_converged": sum(1 for r in ts_results if r.get("neb_converged")),
            "current_run_id": run_id,
            "run_metadata_relpath": (
                f"{run_id}/metadata.json" if run_id is not None else None
            ),
            "run_timing_relpath": (
                f"{run_id}/timing.json"
                if run_id is not None
                and os.path.isfile(os.path.join(output_dir, run_id, "timing.json"))
                else None
            ),
            "results": [],
        }
    )

    for result in ts_results:
        # Create JSON-serializable result (remove Atoms objects)
        result_json = {
            "pair_id": result["pair_id"],
            "status": result["status"],
            "neb_converged": result.get("neb_converged", False),
            "n_images": result.get("n_images"),
            "spring_constant": result.get("spring_constant"),
            "reactant_energy": result.get("reactant_energy"),
            "product_energy": result.get("product_energy"),
            "ts_energy": result.get("ts_energy"),
            "barrier_height": result.get("barrier_height"),
            "error": result.get("error"),
        }
        if result.get("minima_indices") is not None:
            result_json["minima_indices"] = result["minima_indices"]
        if result.get("minima_provenance") is not None:
            result_json["minima_provenance"] = result["minima_provenance"]
        if result["status"] == "success":
            result_json["ts_image_index"] = result.get("ts_image_index")

        summary["results"].append(result_json)

    # Keep statistics aligned with ts_network metadata output.
    summary["statistics"] = compute_ts_statistics(ts_results)

    summary_path = os.path.join(output_dir, "results_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "TS summary %s (success %s/%s, converged %s/%s)",
        summary_path,
        summary["num_successful"],
        summary["num_total_pairs"],
        summary["num_converged"],
        summary["num_total_pairs"],
    )

    return summary_path


def _cluster_ts_candidates_globally(
    candidates: list[tuple[float, Atoms, str, tuple[int, int], dict[str, Any]]],
    energy_tolerance: float,
    similarity_tolerance: float,
    similarity_pair_cor_max: float,
    *,
    use_mic: bool = False,
    n_slab: int | None = None,
) -> list[list[tuple[float, Atoms, str, tuple[int, int], dict[str, Any]]]]:
    """Cluster TS candidates by energy + geometry in one deterministic pass."""
    if not candidates:
        return []

    sorted_candidates = sorted(candidates, key=lambda c: c[0])
    clusters: list[list[tuple[float, Atoms, str, tuple[int, int], dict[str, Any]]]] = []
    representatives: list[tuple[float, Atoms]] = []

    for cand in sorted_candidates:
        energy, atoms, *_ = cand
        matched_idx: int | None = None

        for idx, (rep_energy, rep_atoms) in enumerate(representatives):
            if abs(float(energy) - float(rep_energy)) > energy_tolerance:
                continue
            _cum, _maxd, are_similar = calculate_structure_similarity(
                rep_atoms,
                atoms,
                tolerance=similarity_tolerance,
                pair_cor_max=similarity_pair_cor_max,
                use_mic=use_mic,
                n_slab=n_slab,
            )
            if are_similar:
                matched_idx = idx
                break

        if matched_idx is None:
            clusters.append([cand])
            representatives.append((float(energy), atoms))
        else:
            clusters[matched_idx].append(cand)

    return clusters


def write_final_unique_ts(
    ts_results: list[dict[str, Any]],
    output_dir: str,
    composition: list[str],
    energy_tolerance: float = DEFAULT_ENERGY_TOLERANCE,
    similarity_tolerance: float = DEFAULT_COMPARATOR_TOL,
    similarity_pair_cor_max: float = 0.1,
    minima: list | None = None,
    minima_base_dir: str | None = None,
    run_context: dict[str, Any] | None = None,
    surface_aware: bool = False,
    n_slab: int | None = None,
    path_key: str | None = None,
) -> list[dict[str, Any]]:
    """Deduplicate successful TS geometries globally and write unique `.xyz` files.

    Structures that are the same across different minima pairs (e.g. a
    bifurcation TS) are merged into one file. Each returned dict includes
    ``connected_edges`` listing every ``pair_id`` / ``minima_indices`` that
    produced that geometry.

    Returns a list of dictionaries with keys including:
      - ``connected_edges``, ``connected_minima``
      - ``pair_id``, ``minima_indices`` (first edge)
      - ``ts_energy``, ``barrier_height`` (from lowest-energy cluster member)
      - ``filename``, ``neb_converged``

    This function is best-effort and will not raise on IO errors.
    """
    logger = get_logger(__name__)

    os.makedirs(output_dir, exist_ok=True)
    formula = path_key or get_cluster_formula(composition)

    # Collect successful TS candidates
    candidates: list[tuple[float, Atoms, str, tuple[int, int], dict[str, Any]]] = []
    for result in ts_results:
        if result.get("status") != "success":
            continue
        if not result.get("neb_converged", False):
            continue
        ts_atoms = result.get("transition_state")
        ts_energy = result.get("ts_energy")
        pair_id = result.get("pair_id")
        if ts_atoms is None or ts_energy is None or pair_id is None:
            continue
        # Parse minima indices from pair_id (strict validation)

        minima_indices = validate_pair_id(pair_id)

        candidates.append(
            (float(ts_energy), ts_atoms.copy(), pair_id, minima_indices, result)
        )

    final_dir = os.path.join(output_dir, "final_unique_ts")
    os.makedirs(final_dir, exist_ok=True)

    summary_list: list[dict[str, Any]] = []

    if not candidates:
        # Write empty summary
        summary_path = os.path.join(final_dir, "final_unique_ts_summary.json")
        empty_data: dict[str, Any] = output_json_provenance(extra=run_context or {})
        empty_data.update({"formula": formula, "unique_ts": []})
        if minima_base_dir is not None:
            empty_data["minima_base_dir"] = minima_base_dir
        with open(summary_path, "w") as f:
            json.dump(empty_data, f, indent=2)
        logger.info("No successful TSs to deduplicate for %s", formula)
        return []

    clusters = _cluster_ts_candidates_globally(
        candidates,
        energy_tolerance,
        similarity_tolerance,
        similarity_pair_cor_max,
        use_mic=surface_aware,
        n_slab=n_slab,
    )

    rank = 0
    for cluster in clusters:
        cluster_sorted = sorted(cluster, key=lambda c: c[0])
        seen_pair: set[str] = set()
        connected_edges: list[dict[str, Any]] = []
        for _energy, _atoms, pair_id, minima_indices, result in cluster_sorted:
            if pair_id in seen_pair:
                continue
            seen_pair.add(pair_id)
            edge: dict[str, Any] = {
                "pair_id": pair_id,
                "minima_indices": [int(minima_indices[0]), int(minima_indices[1])],
                "barrier_height": result.get("barrier_height"),
                "neb_converged": bool(result.get("neb_converged", False)),
                "reactant_energy": result.get("reactant_energy"),
                "product_energy": result.get("product_energy"),
                "barrier_forward": result.get("barrier_forward"),
                "barrier_reverse": result.get("barrier_reverse"),
            }
            if minima is not None:
                i, j = minima_indices
                edge["minima_provenance"] = [
                    minima_provenance_dict(minima, i),
                    minima_provenance_dict(minima, j),
                ]
            connected_edges.append(edge)

        connected_edges.sort(
            key=lambda e: (e["minima_indices"][0], e["minima_indices"][1])
        )

        energy, atoms, _pid, _mi, result = min(cluster, key=lambda c: c[0])

        first_edge = connected_edges[0]
        pair_id = str(first_edge["pair_id"])
        minima_indices = [
            int(first_edge["minima_indices"][0]),
            int(first_edge["minima_indices"][1]),
        ]

        connected_minima_sorted = sorted(
            {idx for e in connected_edges for idx in e["minima_indices"]}
        )

        rank += 1
        atoms_clean = atoms.copy()
        atoms_clean.calc = None
        if not surface_aware:
            atoms_clean.center()
        if "tags" in atoms_clean.arrays:
            del atoms_clean.arrays["tags"]

        if len(connected_edges) > 1:
            filename = f"{formula}_ts_{rank:02d}.xyz"
        else:
            filename = f"{formula}_ts_{rank:02d}_pair_{first_edge['pair_id']}.xyz"
        filepath = os.path.join(final_dir, filename)
        ase_write(filepath, atoms_clean)

        item: dict[str, Any] = {
            "pair_id": pair_id,
            "ts_energy": float(energy),
            "barrier_height": result.get("barrier_height"),
            "minima_indices": minima_indices,
            "connected_edges": connected_edges,
            "connected_minima": connected_minima_sorted,
            "filename": filepath,
            "neb_converged": bool(result.get("neb_converged", False)),
            "_atoms_obj": atoms,
        }
        if minima is not None:
            i, j = minima_indices
            item["minima_provenance"] = [
                minima_provenance_dict(minima, i),
                minima_provenance_dict(minima, j),
            ]
        summary_list.append(item)

    # Write summary (serialize without Atoms objects)
    serializable_summary = []
    for item in summary_list:
        serial_item = {k: v for k, v in item.items() if k != "_atoms_obj"}
        serializable_summary.append(serial_item)

    summary_path = os.path.join(final_dir, "final_unique_ts_summary.json")
    summary_data: dict[str, Any] = output_json_provenance(extra=run_context or {})
    summary_data.update({"formula": formula, "unique_ts": serializable_summary})
    if minima_base_dir is not None:
        summary_data["minima_base_dir"] = minima_base_dir
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    logger.info(
        "Unique TS: %d structures in %s, summary %s",
        len(summary_list),
        final_dir,
        summary_path,
    )

    return summary_list
