"""Cluster-on-surface (adsorbate + slab) support for SCGO.

Heavy modules (constraints, deposition, validation) load lazily so that
``from scgo.surface.config import SurfaceSystemConfig`` does not pull the
deposition / hierarchical import graph.
"""

from __future__ import annotations

import importlib
from typing import Any

from scgo.surface.composition import full_adsorbate_slab_composition
from scgo.surface.config import (
    SurfaceSystemConfig,
    describe_surface_config,
    make_surface_config,
)
from scgo.surface.fragment_templates import build_default_fragment_template
from scgo.surface.objectives import adsorption_energy
from scgo.surface.partition import (
    SlabSearchPartition,
    prepare_slab_search_surface_config,
    resolve_slab_search_partition,
)
from scgo.surface.pbc import normalize_slab_pbc
from scgo.surface.presets import (
    DEFAULT_GRAPHITE_SLAB_LAYERS,
    DEFAULT_GRAPHITE_SLAB_REPEAT_XY,
    DEFAULT_GRAPHITE_SLAB_VACUUM,
    build_defected_graphite_slab,
    build_graphene_slab,
    build_graphite_slab,
    build_monovacancy_graphene_slab,
    build_n_doped_graphite_slab,
    make_defected_graphite_surface_config,
    make_graphene_surface_config,
    make_graphite_surface_config,
    make_n_doped_graphite_surface_config,
)

__all__ = [
    "SurfaceSystemConfig",
    "describe_surface_config",
    "make_surface_config",
    "build_default_fragment_template",
    "full_adsorbate_slab_composition",
    "adsorption_energy",
    "attach_slab_constraints",
    "attach_slab_constraints_from_surface_config",
    "surface_slab_constraint_summary",
    "combine_slab_adsorbate",
    "create_deposited_cluster",
    "create_deposited_cluster_batch",
    "slab_surface_extreme",
    "validate_stored_mobile_partition_metadata",
    "validate_stored_slab_adsorbate_metadata",
    "validate_supported_cluster_deposit",
    "validate_surface_config_slab_prefix",
    "DEFAULT_GRAPHITE_SLAB_LAYERS",
    "DEFAULT_GRAPHITE_SLAB_REPEAT_XY",
    "DEFAULT_GRAPHITE_SLAB_VACUUM",
    "build_graphite_slab",
    "build_defected_graphite_slab",
    "build_n_doped_graphite_slab",
    "build_graphene_slab",
    "build_monovacancy_graphene_slab",
    "make_graphene_surface_config",
    "make_graphite_surface_config",
    "make_defected_graphite_surface_config",
    "make_n_doped_graphite_surface_config",
    "SlabSearchPartition",
    "prepare_slab_search_surface_config",
    "resolve_slab_search_partition",
    "normalize_slab_pbc",
]

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "attach_slab_constraints": ("scgo.surface.constraints", "attach_slab_constraints"),
    "attach_slab_constraints_from_surface_config": (
        "scgo.surface.constraints",
        "attach_slab_constraints_from_surface_config",
    ),
    "surface_slab_constraint_summary": (
        "scgo.surface.constraints",
        "surface_slab_constraint_summary",
    ),
    "combine_slab_adsorbate": ("scgo.surface.deposition", "combine_slab_adsorbate"),
    "create_deposited_cluster": (
        "scgo.surface.deposition",
        "create_deposited_cluster",
    ),
    "create_deposited_cluster_batch": (
        "scgo.surface.deposition",
        "create_deposited_cluster_batch",
    ),
    # Via combine_atoms (not deposition): combine_atoms ↔ cluster_adsorbate.combine
    # must stay out of the eager surface package graph.
    "slab_surface_extreme": ("scgo.utils.combine_atoms", "slab_surface_extreme"),
    "validate_stored_mobile_partition_metadata": (
        "scgo.surface.validation",
        "validate_stored_mobile_partition_metadata",
    ),
    "validate_stored_slab_adsorbate_metadata": (
        "scgo.surface.validation",
        "validate_stored_slab_adsorbate_metadata",
    ),
    "validate_supported_cluster_deposit": (
        "scgo.surface.validation",
        "validate_supported_cluster_deposit",
    ),
    "validate_surface_config_slab_prefix": (
        "scgo.surface.validation",
        "validate_surface_config_slab_prefix",
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
