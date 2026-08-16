"""Hierarchical (core + rigid fragment) gas-phase cluster building for GA seeds."""

from __future__ import annotations

from collections.abc import Sequence

from ase import Atoms
from numpy.random import Generator

from scgo.cluster_adsorbate.combine import combine_core_adsorbate
from scgo.cluster_adsorbate.config import (
    ClusterAdsorbateConfig,
    resolve_cluster_adsorbate_config,
)
from scgo.cluster_adsorbate.helpers import resolve_fragment_anchor_and_bond_axis
from scgo.cluster_adsorbate.placement import place_fragment_on_cluster
from scgo.cluster_adsorbate.sites import get_or_compute_surface_site_candidates
from scgo.exceptions import SCGORuntimeError, SCGOValidationError
from scgo.initialization import (
    BatchInitPlan,
    create_initial_cluster,
    create_initial_cluster_batch,
    emit_init_diagnostics,
    plan_batch_initialization,
    reset_init_diagnostics,
)
from scgo.initialization.geometry_helpers import reorder_cluster_to_composition
from scgo.metadata.atoms import set_tags
from scgo.system_types import (
    AdsorbateDefinition,
    AdsorbateFragmentInput,
    resolve_adsorbate_fragments,
)
from scgo.utils.logging import get_logger
from scgo.utils.phase_logging import format_count_summary
from scgo.utils.rng_helpers import create_child_rng
from scgo.utils.site_counts import increment_site_type_count

logger = get_logger(__name__)


def _stamp_site_metadata(combined: Atoms, site_types: list[str]) -> None:
    if site_types:
        set_tags(
            combined,
            adsorbate_site_types_json=site_types,
            adsorbate_site_type=site_types[-1],
        )


def _place_fragments_on_core(
    core: Atoms,
    fragments: Sequence[Atoms],
    rng: Generator,
    ca: ClusterAdsorbateConfig,
    anchor: int,
    bond_axis: tuple[float, float, float] | None,
    batch_site_counts: dict[str, int] | None,
) -> tuple[Atoms, list[str]] | None:
    """Place every fragment onto a single core.

    Previously placed fragments stay fixed and act as clash partners; site
    candidates are computed once for the bare core.

    Returns:
        The combined structure and its per-fragment site types, or ``None`` if
        any fragment cannot be placed (the caller should retry on a fresh core).
    """
    combined = core
    precomputed_sites = get_or_compute_surface_site_candidates(core)
    within_structure_site_counts: dict[str, int] = {}
    site_types: list[str] = []
    for frag_tmpl in fragments:
        frag_metadata: dict[str, str] = {}
        frag = place_fragment_on_cluster(
            core,
            frag_tmpl,
            rng,
            ca,
            anchor_index=anchor,
            bond_axis=bond_axis,
            within_structure_site_counts=within_structure_site_counts,
            batch_site_counts=batch_site_counts,
            placement_metadata=frag_metadata,
            site_core=core,
            clash_atoms=combined,
            site_candidates=precomputed_sites,
        )
        if frag is None:
            return None
        site_types.append(frag_metadata.get("site_type", "directional_fallback"))
        combined = combine_core_adsorbate(combined, frag)
    _stamp_site_metadata(combined, site_types)
    return combined, site_types


def _record_batch_site_type(
    batch_site_counts: dict[str, int] | None, site_types: list[str]
) -> None:
    """Count one structure against its representative (last placed) site type.

    Matches the surface deposition batch, which counts the single
    ``adsorbate_site_type`` tag stamped on each accepted structure.
    """
    if batch_site_counts is None or not site_types:
        return
    increment_site_type_count(batch_site_counts, site_types[-1])


def build_adsorbate_only_cluster(
    fragment_templates: Sequence[Atoms],
    rng: Generator,
    cluster_adsorbate_config: ClusterAdsorbateConfig | None,
    *,
    adsorbate_definition: AdsorbateDefinition | None = None,
    max_placement_attempts: int = 200,
    batch_site_counts: dict[str, int] | None = None,
) -> Atoms | None:
    """Place one or more molecular fragments without a metal core.

    Returns:
        The combined fragment structure, or ``None`` if every attempt fails.

    Raises:
        SCGOValidationError: If ``fragment_templates`` is empty.
    """
    if not fragment_templates:
        raise SCGOValidationError(
            "fragment_templates must contain at least one fragment"
        )

    ca = resolve_cluster_adsorbate_config(cluster_adsorbate_config)
    anchor, bond_axis = (
        resolve_fragment_anchor_and_bond_axis(adsorbate_definition)
        if adsorbate_definition is not None
        else (0, None)
    )

    first = fragment_templates[0].copy()
    first.center()
    if len(fragment_templates) == 1:
        return first

    for _ in range(max_placement_attempts):
        combined = first.copy()
        site_core = combined
        within_structure_site_counts: dict[str, int] = {}
        site_types: list[str] = []
        all_ok = True
        for frag_tmpl in fragment_templates[1:]:
            frag_metadata: dict[str, str] = {}
            placed = place_fragment_on_cluster(
                site_core,
                frag_tmpl,
                rng,
                ca,
                anchor_index=anchor,
                bond_axis=bond_axis,
                site_core=site_core,
                clash_atoms=combined,
                within_structure_site_counts=within_structure_site_counts,
                batch_site_counts=batch_site_counts,
                placement_metadata=frag_metadata,
            )
            if placed is None:
                all_ok = False
                break
            site_types.append(frag_metadata.get("site_type", "directional_fallback"))
            combined = combine_core_adsorbate(combined, placed)
            site_core = combined
        if all_ok:
            _stamp_site_metadata(combined, site_types)
            return combined
    logger.warning(
        "Reached max_placement_attempts=%s in build_adsorbate_only_cluster",
        max_placement_attempts,
    )
    return None


def build_hierarchical_core_fragment_cluster(
    adsorbate_definition: AdsorbateDefinition,
    rng: Generator,
    previous_search_glob: str,
    fragment_templates: AdsorbateFragmentInput | None,
    cluster_adsorbate_config: ClusterAdsorbateConfig | None,
    *,
    cluster_init_vacuum: float = 8.0,
    init_mode: str = "smart",
    max_placement_attempts: int = 200,
    batch_site_counts: dict[str, int] | None = None,
    placement_metadata: dict[str, str] | None = None,
    plan: BatchInitPlan | None = None,
    allocation: tuple[str, int | None] | None = None,
    emit_diagnostics: bool = True,
    verbosity: int = 1,
) -> Atoms | None:
    """Build core cluster, place rigid fragment(s), return gas-phase structure.

    Fragments are placed sequentially on adsorption sites of the metal core,
    with anti-repetition weighting across site types; previously placed
    fragments are kept fixed and are used as clash partners.

    Args:
        plan: Pre-computed :class:`~scgo.initialization.BatchInitPlan` (discovery + allocation) so the
            noisy DB scan runs once per batch instead of once per candidate.
        allocation: Override the ``(strategy, template_index)`` allocation for
            this single core (used with ``plan`` to preserve diversity).
        emit_diagnostics: When ``False``, suppress the per-call diagnostic
            summary (the batch owner emits the aggregate summary).
        verbosity: Verbosity for initialization diagnostic summaries (0-3).

    Returns:
        The combined structure, or ``None`` if every attempt fails.
    """
    core_list = [str(s) for s in adsorbate_definition.core_symbols]
    fragments = resolve_adsorbate_fragments(
        fragment_templates,
        adsorbate_definition,
        context="build_hierarchical_core_fragment_cluster",
    )

    if not core_list:
        return build_adsorbate_only_cluster(
            fragments,
            rng,
            cluster_adsorbate_config,
            adsorbate_definition=adsorbate_definition,
            max_placement_attempts=max_placement_attempts,
        )

    ca = resolve_cluster_adsorbate_config(cluster_adsorbate_config)
    anchor, bond_axis = resolve_fragment_anchor_and_bond_axis(adsorbate_definition)
    for _ in range(max_placement_attempts):
        core = create_initial_cluster(
            list(core_list),
            vacuum=cluster_init_vacuum,
            rng=rng,
            previous_search_glob=previous_search_glob,
            mode=init_mode,
            plan=plan,
            allocation=allocation,
            emit_diagnostics=emit_diagnostics,
            verbosity=verbosity,
        )
        core = reorder_cluster_to_composition(core, core_list)
        placed = _place_fragments_on_core(
            core, fragments, rng, ca, anchor, bond_axis, batch_site_counts
        )
        if placed is None:
            continue

        combined, site_types = placed
        if placement_metadata is not None and site_types:
            placement_metadata["site_types"] = ",".join(site_types)
            placement_metadata["site_type"] = site_types[-1]
        return combined
    logger.warning(
        "Reached max_placement_attempts=%s in build_hierarchical_core_fragment_cluster",
        max_placement_attempts,
    )
    return None


def build_hierarchical_core_fragment_cluster_batch(
    adsorbate_definition: AdsorbateDefinition,
    rng: Generator,
    previous_search_glob: str,
    fragment_templates: AdsorbateFragmentInput | None,
    cluster_adsorbate_config: ClusterAdsorbateConfig | None,
    *,
    cluster_init_vacuum: float = 8.0,
    init_mode: str = "smart",
    n_structures: int = 1,
    max_placement_attempts: int = 200,
    batch_site_counts: dict[str, int] | None = None,
    n_jobs: int | None = None,
    plan: BatchInitPlan | None = None,
    verbosity: int = 1,
) -> list[Atoms]:
    """Build an entire batch of hierarchical (core + fragment) clusters.

    Discovery (previous-search DB scan) and strategy allocation run exactly once
    for the whole batch via a shared :class:`~scgo.initialization.BatchInitPlan`, and all metal cores
    are generated in parallel. Fragment placement then runs on each core,
    retrying with freshly generated cores (reusing the same plan) when a
    placement fails.

    This is the discovery-once, parallel counterpart of
    :func:`build_hierarchical_core_fragment_cluster` and must be used for batch
    population initialization so discovery + allocation are not re-run per
    candidate.

    Args:
        n_jobs: Parallelism for core generation; ``None`` uses the project
            default (single worker; opt in with -1/-2 for parallelism).
        plan: Pre-computed :class:`~scgo.initialization.BatchInitPlan`; built here when omitted.
        verbosity: Verbosity for the single aggregate initialization summary
            this function emits (0-3).

    Raises:
        SCGORuntimeError: If fewer than ``n_structures`` structures can be built.
    """
    core_list = [str(s) for s in adsorbate_definition.core_symbols]
    fragments = resolve_adsorbate_fragments(
        fragment_templates,
        adsorbate_definition,
        context="build_hierarchical_core_fragment_cluster_batch",
    )

    if not core_list:
        # No metal core: place fragments directly (no DB discovery involved).
        out: list[Atoms] = []
        for _ in range(n_structures):
            combined = build_adsorbate_only_cluster(
                fragments,
                create_child_rng(rng),
                cluster_adsorbate_config,
                adsorbate_definition=adsorbate_definition,
                max_placement_attempts=max_placement_attempts,
                batch_site_counts=batch_site_counts,
            )
            if combined is None:
                raise SCGORuntimeError(
                    "build_hierarchical_core_fragment_cluster_batch: could not place "
                    "adsorbate-only fragments; increase max_placement_attempts."
                )
            out.append(combined)
        return out

    ca = resolve_cluster_adsorbate_config(cluster_adsorbate_config)
    anchor, bond_axis = resolve_fragment_anchor_and_bond_axis(adsorbate_definition)

    reset_init_diagnostics()
    if plan is None:
        plan = plan_batch_initialization(
            core_list,
            n_structures,
            rng,
            vacuum=cluster_init_vacuum,
            previous_search_glob=previous_search_glob,
            mode=init_mode,
        )

    # Generate all cores in parallel, reusing the single batch plan.
    cores = create_initial_cluster_batch(
        composition=list(core_list),
        n_structures=n_structures,
        rng=rng,
        vacuum=cluster_init_vacuum,
        previous_search_glob=previous_search_glob,
        mode=init_mode,
        n_jobs=n_jobs,
        plan=plan,
        emit_diagnostics=False,
        verbosity=verbosity,
    )

    results: list[Atoms] = []
    placement_rng = create_child_rng(rng)
    extra_cores = 0
    max_extra_cores = n_structures * max_placement_attempts
    core_iter = iter(cores)
    while len(results) < n_structures:
        core = next(core_iter, None)
        if core is None:
            # The pre-generated cores are exhausted because placement failed on
            # some of them; top up one core at a time, cycling the plan's
            # strategy mix so retries do not all reuse the same allocation.
            if extra_cores >= max_extra_cores:
                break
            core = create_initial_cluster(
                list(core_list),
                rng=placement_rng,
                vacuum=cluster_init_vacuum,
                previous_search_glob=previous_search_glob,
                mode=init_mode,
                plan=plan,
                allocation=plan.allocation_for(extra_cores),
                emit_diagnostics=False,
                verbosity=verbosity,
            )
            extra_cores += 1
        core = reorder_cluster_to_composition(core, core_list)
        placed = _place_fragments_on_core(
            core, fragments, placement_rng, ca, anchor, bond_axis, batch_site_counts
        )
        if placed is None:
            continue
        combined, site_types = placed
        _record_batch_site_type(batch_site_counts, site_types)
        results.append(combined)

    if len(results) < n_structures:
        raise SCGORuntimeError(
            "build_hierarchical_core_fragment_cluster_batch: could not place "
            f"{len(results)} of {n_structures} structures after retries; increase "
            "max_placement_attempts or relax ClusterAdsorbateConfig."
        )

    site_summary = format_count_summary(batch_site_counts) if batch_site_counts else ""
    emit_init_diagnostics(
        n_structures,
        verbosity=verbosity,
        extra=f"site types {site_summary}" if site_summary else "",
    )
    return results
