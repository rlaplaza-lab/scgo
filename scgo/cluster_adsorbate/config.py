"""Configuration for placing adsorbates on gas-phase metal clusters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from scgo.exceptions import SCGOValidationError
from scgo.initialization.initialization_config import (
    CONNECTIVITY_FACTOR,
    MIN_DISTANCE_FACTOR_DEFAULT,
)
from scgo.utils.config_aliases import _UNSET, resolve_aliased_float
from scgo.utils.validation import validate_positive


@dataclass(frozen=True, init=False)
class ClusterAdsorbateConfig:
    """Shared placement / relaxation settings for any small adsorbate fragment.

    Height may be set via ``height_*`` or alias ``adsorption_height_*``.
    """

    height_min: float
    height_max: float
    max_placement_attempts: int
    blmin_ratio: float
    cell_margin: float
    random_spin_about_normal: bool
    validate_combined_structure: bool
    structure_min_distance_factor: float
    structure_connectivity_factor: float
    structure_check_clashes: bool
    structure_check_connectivity: bool

    def __init__(
        self,
        *,
        height_min: Any = _UNSET,
        height_max: Any = _UNSET,
        adsorption_height_min: Any = _UNSET,
        adsorption_height_max: Any = _UNSET,
        max_placement_attempts: int = 80,
        blmin_ratio: float = 0.7,
        cell_margin: float = 6.0,
        random_spin_about_normal: bool = True,
        validate_combined_structure: bool = True,
        structure_min_distance_factor: float = MIN_DISTANCE_FACTOR_DEFAULT,
        structure_connectivity_factor: float = CONNECTIVITY_FACTOR,
        structure_check_clashes: bool = True,
        structure_check_connectivity: bool = True,
    ) -> None:
        object.__setattr__(
            self,
            "height_min",
            resolve_aliased_float(
                "height_min",
                height_min,
                "adsorption_height_min",
                adsorption_height_min,
                0.9,
            ),
        )
        object.__setattr__(
            self,
            "height_max",
            resolve_aliased_float(
                "height_max",
                height_max,
                "adsorption_height_max",
                adsorption_height_max,
                2.2,
            ),
        )
        object.__setattr__(self, "max_placement_attempts", max_placement_attempts)
        object.__setattr__(self, "blmin_ratio", blmin_ratio)
        object.__setattr__(self, "cell_margin", cell_margin)
        object.__setattr__(self, "random_spin_about_normal", random_spin_about_normal)
        object.__setattr__(
            self, "validate_combined_structure", validate_combined_structure
        )
        object.__setattr__(
            self, "structure_min_distance_factor", structure_min_distance_factor
        )
        object.__setattr__(
            self, "structure_connectivity_factor", structure_connectivity_factor
        )
        object.__setattr__(self, "structure_check_clashes", structure_check_clashes)
        object.__setattr__(
            self, "structure_check_connectivity", structure_check_connectivity
        )
        self.__post_init__()

    @property
    def adsorption_height_min(self) -> float:
        """Alias for :attr:`height_min`."""
        return self.height_min

    @property
    def adsorption_height_max(self) -> float:
        """Alias for :attr:`height_max`."""
        return self.height_max

    def __post_init__(self) -> None:
        validate_positive("height_min", self.height_min, strict=True)
        validate_positive("height_max", self.height_max, strict=True)
        if self.height_max < self.height_min:
            raise SCGOValidationError("height_max must be >= height_min")
        if self.max_placement_attempts < 1:
            raise SCGOValidationError("max_placement_attempts must be positive")
        validate_positive("blmin_ratio", self.blmin_ratio, strict=True)
        validate_positive("cell_margin", self.cell_margin, strict=True)
        validate_positive(
            "structure_min_distance_factor",
            self.structure_min_distance_factor,
            strict=True,
        )
        validate_positive(
            "structure_connectivity_factor",
            self.structure_connectivity_factor,
            strict=True,
        )


def resolve_cluster_adsorbate_config(
    config: ClusterAdsorbateConfig | None,
) -> ClusterAdsorbateConfig:
    """Return *config* or the package default (single allowed default site)."""
    if config is None:
        return ClusterAdsorbateConfig()
    return config
