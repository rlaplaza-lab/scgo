"""Shared validation / resolution preamble for BH and GA runs."""

from __future__ import annotations

from dataclasses import dataclass

from scgo.cluster_adsorbate.config import ClusterAdsorbateConfig
from scgo.surface.config import SurfaceSystemConfig
from scgo.system_types import (
    ConnectivityFactorInput,
    NormalizedConnectivityFactor,
    SystemPolicy,
    SystemType,
    get_system_policy,
    resolve_connectivity_factor,
    validate_system_type_settings,
)
from scgo.utils.fitness_strategies import (
    FitnessStrategy,
    resolve_fitness_strategy,
)

__all__ = [
    "ResolvedRunContext",
    "validate_and_resolve_run_context",
]


@dataclass(frozen=True)
class ResolvedRunContext:
    """Shared fields resolved once at the start of a BH/GA run."""

    system_type: SystemType
    policy: SystemPolicy
    connectivity_factor: NormalizedConnectivityFactor
    fitness_strategy: FitnessStrategy


def validate_and_resolve_run_context(
    *,
    system_type: SystemType,
    surface_config: SurfaceSystemConfig | None = None,
    connectivity_factor: ConnectivityFactorInput
    | NormalizedConnectivityFactor
    | None = None,
    cluster_adsorbate_config: ClusterAdsorbateConfig | None = None,
    fitness_strategy: str | FitnessStrategy | None = None,
) -> ResolvedRunContext:
    """Validate system settings and resolve policy, connectivity, and fitness.

    Call sites remain responsible for algorithm-specific checks (atoms, GA
    population params, surface partition setup, etc.).
    """
    validate_system_type_settings(
        system_type=system_type, surface_config=surface_config
    )
    resolved_cf = resolve_connectivity_factor(
        connectivity_factor,
        cluster_adsorbate_config=cluster_adsorbate_config,
        surface_config=surface_config,
    )

    policy = get_system_policy(system_type)
    strategy_arg = str(fitness_strategy) if fitness_strategy is not None else None
    resolved_fitness = FitnessStrategy(
        resolve_fitness_strategy(strategy_arg, allow_none=False)
    )
    return ResolvedRunContext(
        system_type=system_type,
        policy=policy,
        connectivity_factor=resolved_cf,
        fitness_strategy=resolved_fitness,
    )
