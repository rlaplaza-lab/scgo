"""Configuration for placing adsorbates on gas-phase metal clusters."""

from __future__ import annotations

from dataclasses import dataclass

from scgo.initialization.initialization_config import (
    CONNECTIVITY_FACTOR,
    MIN_DISTANCE_FACTOR_DEFAULT,
)
from scgo.utils.validation import validate_positive


@dataclass(frozen=True)
class ClusterAdsorbateConfig:
    """Shared placement / relaxation settings for any small adsorbate fragment."""

    height_min: float = 0.9
    height_max: float = 2.2
    max_placement_attempts: int = 80
    blmin_ratio: float = 0.7
    cell_margin: float = 6.0
    random_spin_about_normal: bool = True
    validate_combined_structure: bool = True
    structure_min_distance_factor: float = MIN_DISTANCE_FACTOR_DEFAULT
    structure_connectivity_factor: float = CONNECTIVITY_FACTOR
    structure_check_clashes: bool = True
    structure_check_connectivity: bool = True

    def __post_init__(self) -> None:
        validate_positive("height_min", self.height_min, strict=True)
        validate_positive("height_max", self.height_max, strict=True)
        if self.height_max < self.height_min:
            raise ValueError("height_max must be >= height_min")
        if self.max_placement_attempts < 1:
            raise ValueError("max_placement_attempts must be positive")
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
