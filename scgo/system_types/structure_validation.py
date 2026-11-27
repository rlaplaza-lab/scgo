"""Frozen structural-validation flags shared by GO and TS gates."""

from __future__ import annotations

from dataclasses import dataclass

from scgo.cluster_adsorbate.config import ClusterAdsorbateConfig
from scgo.system_types.connectivity_factor import (
    ConnectivityFactorInput,
    NormalizedConnectivityFactor,
)


@dataclass(frozen=True)
class StructureValidationPolicy:
    """Flags for system-type structural validation.

    Public runners still take these as keywords and build one policy.
    Internal gates take the policy so a new flag is added here once.
    """

    connectivity_factor: (
        ConnectivityFactorInput | NormalizedConnectivityFactor | None
    ) = None
    cluster_adsorbate_config: ClusterAdsorbateConfig | None = None
    allow_cluster_fragmentation: bool = False
    allow_adsorbate_surface_detachment: bool = False
    enforce_adsorbate_subgraph_integrity: bool = True
    binding_penetration_tolerance_a: float = 0.1
    n_slab_deposit: int | None = None

    @classmethod
    def from_search_flags(
        cls,
        *,
        connectivity_factor: (
            ConnectivityFactorInput | NormalizedConnectivityFactor | None
        ),
        cluster_adsorbate_config: ClusterAdsorbateConfig | None,
        allow_cluster_fragmentation: bool,
        allow_adsorbate_surface_detachment: bool,
        enforce_adsorbate_subgraph_integrity: bool,
        n_slab_deposit: int | None = None,
    ) -> StructureValidationPolicy:
        """Build the policy shared by GA, basin hopping, and simple GO runners."""
        return cls(
            connectivity_factor=connectivity_factor,
            cluster_adsorbate_config=cluster_adsorbate_config,
            allow_cluster_fragmentation=allow_cluster_fragmentation,
            allow_adsorbate_surface_detachment=allow_adsorbate_surface_detachment,
            enforce_adsorbate_subgraph_integrity=enforce_adsorbate_subgraph_integrity,
            n_slab_deposit=n_slab_deposit,
        )
