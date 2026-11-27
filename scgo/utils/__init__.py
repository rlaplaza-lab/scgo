"""Utility functions and helpers for cluster optimization.

This package contains various utility modules:

- helpers: Core utility functions for optimization and validation
- comparators: Structure comparison tools for duplicate detection
- mutation_weights: Adaptive mutation probability configuration
- atoms_helpers: ASE Atoms object manipulation utilities
- logging: Logging configuration and utilities
- rng_helpers: Random number generator utilities for reproducibility
- validation: Input validation utilities
"""

from __future__ import annotations

from scgo.utils.lazy_exports import lazy_submodule_dir, lazy_submodule_getattr

__all__ = [
    # atoms_helpers
    "parse_energy_from_xyz_comment",
    # comparators
    "PureInteratomicDistanceComparator",
    # helpers
    "auto_niter",
    "auto_niter_local_relaxation",
    "auto_population_size",
    "get_cluster_formula",
    "get_composition_counts",
    "get_ordered_formula",
    "get_system_path_key",
    "is_true_minimum",
    "perform_local_relaxation",
    # logging
    "TRACE",
    "VERBOSITY_LEVELS",
    "configure_logging",
    "get_logger",
    "log_debug_v",
    "log_info_v",
    "log_warning_v",
    "should_show_progress",
    "suppress_matching_stdout",
    # mutation_weights
    "get_adaptive_mutation_config",
    # rng_helpers
    "create_child_rng",
    "ensure_rng",
    "ensure_rng_or_create",
    "get_child_rng_or_none",
    "offspring_rng_triple",
]

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "parse_energy_from_xyz_comment": (
        "scgo.utils.atoms_helpers",
        "parse_energy_from_xyz_comment",
    ),
    "PureInteratomicDistanceComparator": (
        "scgo.utils.comparators",
        "PureInteratomicDistanceComparator",
    ),
    "auto_niter": ("scgo.utils.helpers", "auto_niter"),
    "auto_niter_local_relaxation": (
        "scgo.utils.helpers",
        "auto_niter_local_relaxation",
    ),
    "auto_population_size": ("scgo.utils.helpers", "auto_population_size"),
    "get_cluster_formula": ("scgo.utils.helpers", "get_cluster_formula"),
    "get_composition_counts": ("scgo.utils.helpers", "get_composition_counts"),
    "get_ordered_formula": ("scgo.utils.helpers", "get_ordered_formula"),
    "get_system_path_key": ("scgo.utils.helpers", "get_system_path_key"),
    "is_true_minimum": ("scgo.utils.helpers", "is_true_minimum"),
    "perform_local_relaxation": ("scgo.utils.helpers", "perform_local_relaxation"),
    "TRACE": ("scgo.utils.logging", "TRACE"),
    "VERBOSITY_LEVELS": ("scgo.utils.logging", "VERBOSITY_LEVELS"),
    "configure_logging": ("scgo.utils.logging", "configure_logging"),
    "get_logger": ("scgo.utils.logging", "get_logger"),
    "log_debug_v": ("scgo.utils.logging", "log_debug_v"),
    "log_info_v": ("scgo.utils.logging", "log_info_v"),
    "log_warning_v": ("scgo.utils.logging", "log_warning_v"),
    "should_show_progress": ("scgo.utils.logging", "should_show_progress"),
    "suppress_matching_stdout": ("scgo.utils.logging", "suppress_matching_stdout"),
    "get_adaptive_mutation_config": (
        "scgo.utils.mutation_weights",
        "get_adaptive_mutation_config",
    ),
    "create_child_rng": ("scgo.utils.rng_helpers", "create_child_rng"),
    "ensure_rng": ("scgo.utils.rng_helpers", "ensure_rng"),
    "ensure_rng_or_create": ("scgo.utils.rng_helpers", "ensure_rng_or_create"),
    "get_child_rng_or_none": ("scgo.utils.rng_helpers", "get_child_rng_or_none"),
    "offspring_rng_triple": ("scgo.utils.rng_helpers", "offspring_rng_triple"),
}

__getattr__ = lazy_submodule_getattr(globals(), _LAZY_ATTRS, cache=True)
__dir__ = lazy_submodule_dir(__all__)
