"""SCGO: global optimization and TS search tools. See README.md and examples/."""

from __future__ import annotations

import os


def configure() -> None:
    """Apply SCGO runtime configuration.

    Currently sets ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` unless
    already present in the environment. Skipped entirely when
    ``SCGO_SKIP_PYTORCH_CONFIG`` is set.
    """
    if os.environ.get("SCGO_SKIP_PYTORCH_CONFIG"):
        return
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# Algorithms
from scgo._version import __version__
from scgo.algorithms import bh_go, ga_go

# Cluster + adsorbate (composable local relax)
from scgo.cluster_adsorbate.combine import combine_core_adsorbate
from scgo.cluster_adsorbate.config import ClusterAdsorbateConfig
from scgo.cluster_adsorbate.constraints import attach_fix_bond_lengths
from scgo.cluster_adsorbate.placement import place_fragment_on_cluster
from scgo.cluster_adsorbate.relax import relax_metal_cluster_with_adsorbate
from scgo.cluster_adsorbate.validation import validate_combined_cluster_structure

# Database
from scgo.database import (
    SCGODatabaseManager,
    load_previous_run_results,
    load_reference_structures,
    setup_database,
)
from scgo.exceptions import (
    SCGOConfigurationError,
    SCGODatabaseError,
    SCGOError,
    SCGOFileError,
    SCGONotImplementedError,
    SCGORuntimeError,
    SCGOValidationError,
)

# Initialization
from scgo.initialization import (
    create_initial_cluster,
    generate_template_structure,
    validate_cluster_structure,
)

# Parameter presets
from scgo.param_presets import (
    AVAILABLE_MACE_MODELS,
    AVAILABLE_UMA_MODELS,
    AVAILABLE_UPET_MODELS,
    get_default_params,
    get_default_uma_params,
    get_default_upet_params,
    get_diversity_params,
    get_high_energy_params,
    get_low_effort_torchsim_ga_params,
    get_low_effort_ts_search_params,
    get_low_effort_uma_ga_params,
    get_low_effort_upet_ga_params,
    get_minimal_ga_params,
    get_testing_params,
    get_torchsim_ga_params,
    get_ts_search_params,
    get_uma_ga_benchmark_params,
    get_upet_ga_benchmark_params,
)
from scgo.runner_api import (
    CompositionInput,
    log_go_ts_summary,
    parse_composition_arg,
    resolve_workflow_seed,
    run_go,
    run_go_campaign,
    run_go_ts,
    run_go_ts_campaign,
    run_ts_campaign,
    run_ts_search,
)

# Surface / adsorption
from scgo.surface.config import SurfaceSystemConfig, make_surface_config
from scgo.surface.objectives import adsorption_energy
from scgo.surface.presets import (
    make_defected_graphite_surface_config,
    make_graphene_surface_config,
    make_graphite_surface_config,
    make_hopg_5x5_defected_graphite_surface_config,
    make_hopg_5x5_graphite_surface_config,
    make_n_doped_graphite_surface_config,
)

# Utilities
from scgo.utils.helpers import (
    get_cluster_formula,
    get_ordered_formula,
    get_system_path_key,
    is_true_minimum,
    perform_local_relaxation,
)
from scgo.utils.logging import (
    configure_logging,
    get_logger,
)
from scgo.utils.rng_helpers import get_child_rng_or_none


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    # Version and capabilities
    "__version__",
    "AVAILABLE_MACE_MODELS",
    "AVAILABLE_UMA_MODELS",
    "AVAILABLE_UPET_MODELS",
    # Exceptions
    "SCGOError",
    "SCGOConfigurationError",
    "SCGOValidationError",
    "SCGORuntimeError",
    "SCGODatabaseError",
    "SCGONotImplementedError",
    "SCGOFileError",
    # Runtime configuration
    "configure",
    # Algorithms (for advanced users)
    "bh_go",
    "ga_go",
    # Database
    "load_previous_run_results",
    "load_reference_structures",
    "SCGODatabaseManager",
    "setup_database",
    # Initialization
    "create_initial_cluster",
    "generate_template_structure",
    "validate_cluster_structure",
    # Surface
    "SurfaceSystemConfig",
    "adsorption_energy",
    "make_graphene_surface_config",
    "make_graphite_surface_config",
    "make_hopg_5x5_graphite_surface_config",
    "make_hopg_5x5_defected_graphite_surface_config",
    "make_defected_graphite_surface_config",
    "make_n_doped_graphite_surface_config",
    "make_surface_config",
    # Cluster + adsorbate
    "ClusterAdsorbateConfig",
    "attach_fix_bond_lengths",
    "combine_core_adsorbate",
    "place_fragment_on_cluster",
    "relax_metal_cluster_with_adsorbate",
    "validate_combined_cluster_structure",
    # Logging
    "configure_logging",
    "get_logger",
    # Parameter presets
    "get_default_params",
    "get_diversity_params",
    "get_high_energy_params",
    "get_low_effort_torchsim_ga_params",
    "get_low_effort_upet_ga_params",
    "get_low_effort_uma_ga_params",
    "get_low_effort_ts_search_params",
    "get_minimal_ga_params",
    "get_testing_params",
    "get_ts_search_params",
    "get_default_uma_params",
    "get_default_upet_params",
    "get_torchsim_ga_params",
    "get_uma_ga_benchmark_params",
    "get_upet_ga_benchmark_params",
    # Main run API (see scgo.runner_api)
    "CompositionInput",
    "resolve_workflow_seed",
    "run_go",
    "run_go_campaign",
    "run_go_ts",
    "run_go_ts_campaign",
    "log_go_ts_summary",
    "parse_composition_arg",
    "run_ts_campaign",
    "run_ts_search",
    # Utilities
    "get_child_rng_or_none",
    "get_cluster_formula",
    "get_ordered_formula",
    "get_system_path_key",
    "is_true_minimum",
    "perform_local_relaxation",
]
