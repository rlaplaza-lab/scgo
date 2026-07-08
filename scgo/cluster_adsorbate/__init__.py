"""Composable adsorbate placement and local relaxation on gas-phase metal clusters."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from scgo.cluster_adsorbate.combine import combine_core_adsorbate
from scgo.cluster_adsorbate.config import ClusterAdsorbateConfig
from scgo.cluster_adsorbate.constraints import (
    attach_adsorbate_internal_geometry_constraints,
    attach_fix_bond_lengths,
)
from scgo.cluster_adsorbate.hierarchical import (
    build_adsorbate_only_cluster,
    build_hierarchical_core_fragment_cluster,
)
from scgo.cluster_adsorbate.placement import (
    blmin_for_core_and_fragment,
    place_fragment_on_cluster,
)
from scgo.cluster_adsorbate.relax import relax_metal_cluster_with_adsorbate
from scgo.cluster_adsorbate.validation import validate_combined_cluster_structure
from scgo.initialization.geometry_helpers import reorder_cluster_to_composition


def __getattr__(name: str) -> Callable[..., Any]:
    if name in {
        "enforce_frozen_adsorbate_geometry",
        "restore_rigid_adsorbate_fragments",
    }:
        from scgo.cluster_adsorbate import rigid as _rigid

        return getattr(_rigid, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "build_adsorbate_only_cluster",
    "build_hierarchical_core_fragment_cluster",
    "reorder_cluster_to_composition",
    "ClusterAdsorbateConfig",
    "attach_adsorbate_internal_geometry_constraints",
    "attach_fix_bond_lengths",
    "blmin_for_core_and_fragment",
    "combine_core_adsorbate",
    "enforce_frozen_adsorbate_geometry",
    "restore_rigid_adsorbate_fragments",
    "place_fragment_on_cluster",
    "relax_metal_cluster_with_adsorbate",
    "validate_combined_cluster_structure",
]
