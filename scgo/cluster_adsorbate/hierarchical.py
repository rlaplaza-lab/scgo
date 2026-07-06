"""Hierarchical (core + rigid fragment) gas-phase cluster building for GA seeds."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING

from ase import Atoms
from numpy.random import Generator

from scgo.cluster_adsorbate.combine import combine_core_adsorbate
from scgo.cluster_adsorbate.config import ClusterAdsorbateConfig
from scgo.cluster_adsorbate.helpers import resolve_fragment_anchor_and_bond_axis
from scgo.cluster_adsorbate.placement import place_fragment_on_cluster
from scgo.initialization import create_initial_cluster
from scgo.initialization.geometry_helpers import reorder_cluster_to_composition
from scgo.utils.logging import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from scgo.system_types import AdsorbateDefinition, AdsorbateFragmentInput


def _stamp_site_metadata(combined: Atoms, site_types: list[str]) -> None:
    if site_types:
        combined.info["adsorbate_site_types_json"] = json.dumps(site_types)
        combined.info["adsorbate_site_type"] = site_types[-1]


def build_adsorbate_only_cluster(
    fragment_templates: Sequence[Atoms],
    rng: Generator,
    cluster_adsorbate_config: ClusterAdsorbateConfig | None,
    *,
    adsorbate_definition: AdsorbateDefinition | None = None,
    max_placement_attempts: int = 200,
    batch_site_counts: dict[str, int] | None = None,
) -> Atoms | None:
    """Place one or more molecular fragments without a metal core."""
    if not fragment_templates:
        raise ValueError("fragment_templates must contain at least one fragment")

    ca = cluster_adsorbate_config or ClusterAdsorbateConfig()
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
        "build_adsorbate_only_cluster: exceeded max_placement_attempts=%s",
        max_placement_attempts,
    )
    return None


def build_hierarchical_core_fragment_cluster(
    full_composition: Sequence[str],
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
) -> Atoms | None:
    """Build core cluster, place rigid fragment(s), return gas-phase structure.

    Each entry in ``fragment_templates`` is placed sequentially on distinct
    adsorption sites while preserving previously placed fragments.
    """
    from scgo.system_types import resolve_adsorbate_fragments

    core_list = [str(s) for s in adsorbate_definition["core_symbols"]]
    ads_list = [str(s) for s in adsorbate_definition["adsorbate_symbols"]]
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

    ca = cluster_adsorbate_config or ClusterAdsorbateConfig()
    expected_mobile = list(core_list) + list(ads_list)
    if list(full_composition) != expected_mobile:
        raise ValueError(
            "Hierarchical init requires the mobile composition to be "
            "core_symbols (in order) then adsorbate_symbols (in order). "
            f"Got {list(full_composition)!r}, expected {expected_mobile!r}."
        )

    anchor, bond_axis = resolve_fragment_anchor_and_bond_axis(adsorbate_definition)
    within_structure_site_counts: dict[str, int] = {}
    for _ in range(max_placement_attempts):
        core = create_initial_cluster(
            list(core_list),
            vacuum=cluster_init_vacuum,
            rng=rng,
            previous_search_glob=previous_search_glob,
            mode=init_mode,
        )
        core = reorder_cluster_to_composition(core, core_list)
        combined = core
        metal_core = core
        site_types: list[str] = []
        placement_failed = False
        for frag_tmpl in fragments:
            frag_metadata: dict[str, str] = {}
            frag = None
            for _frag_attempt in range(ca.max_placement_attempts):
                frag = place_fragment_on_cluster(
                    metal_core,
                    frag_tmpl,
                    rng,
                    ca,
                    anchor_index=anchor,
                    bond_axis=bond_axis,
                    within_structure_site_counts=within_structure_site_counts,
                    batch_site_counts=batch_site_counts,
                    placement_metadata=frag_metadata,
                    site_core=metal_core,
                    clash_atoms=combined,
                )
                if frag is not None:
                    break
            if frag is None:
                placement_failed = True
                break
            site_types.append(frag_metadata.get("site_type", "directional_fallback"))
            combined = combine_core_adsorbate(combined, frag)
            metal_core = combined

        if placement_failed:
            continue

        _stamp_site_metadata(combined, site_types)
        if placement_metadata is not None and site_types:
            placement_metadata["site_types"] = ",".join(site_types)
            placement_metadata["site_type"] = site_types[-1]
        return combined
    logger.warning(
        "build_hierarchical_core_fragment_cluster: exceeded max_placement_attempts=%s",
        max_placement_attempts,
    )
    return None
