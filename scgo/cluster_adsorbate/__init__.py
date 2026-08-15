"""Composable adsorbate placement and local relaxation on gas-phase metal clusters.

Hierarchical builders load lazily so importing leaf helpers (config, placement,
validation) does not close the ``system_types`` ↔ ``hierarchical`` cycle via
this package ``__init__``.
"""

from __future__ import annotations

import importlib
from typing import Any

from scgo.cluster_adsorbate.combine import combine_core_adsorbate
from scgo.cluster_adsorbate.config import ClusterAdsorbateConfig
from scgo.cluster_adsorbate.constraints import (
    attach_adsorbate_internal_geometry_constraints,
    attach_fix_bond_lengths,
)
from scgo.cluster_adsorbate.placement import (
    place_fragment_on_cluster,
)
from scgo.cluster_adsorbate.relax import relax_metal_cluster_with_adsorbate
from scgo.cluster_adsorbate.validation import validate_combined_cluster_structure
from scgo.initialization.geometry_helpers import reorder_cluster_to_composition

# Rigid helpers live in ``scgo.cluster_adsorbate.rigid`` (not re-exported here):
# importing them eagerly would circular-import via ``system_types`` → ``surface``.

__all__ = [
    "build_adsorbate_only_cluster",
    "build_hierarchical_core_fragment_cluster",
    "build_hierarchical_core_fragment_cluster_batch",
    "reorder_cluster_to_composition",
    "ClusterAdsorbateConfig",
    "attach_adsorbate_internal_geometry_constraints",
    "attach_fix_bond_lengths",
    "combine_core_adsorbate",
    "place_fragment_on_cluster",
    "relax_metal_cluster_with_adsorbate",
    "validate_combined_cluster_structure",
]

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "build_adsorbate_only_cluster": (
        "scgo.cluster_adsorbate.hierarchical",
        "build_adsorbate_only_cluster",
    ),
    "build_hierarchical_core_fragment_cluster": (
        "scgo.cluster_adsorbate.hierarchical",
        "build_hierarchical_core_fragment_cluster",
    ),
    "build_hierarchical_core_fragment_cluster_batch": (
        "scgo.cluster_adsorbate.hierarchical",
        "build_hierarchical_core_fragment_cluster_batch",
    ),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRS:
        module_name, attr = _LAZY_ATTRS[name]
        module = importlib.import_module(module_name)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return sorted(__all__)
