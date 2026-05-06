"""Structure validation for core + adsorbate (connectivity and clashes)."""

from __future__ import annotations

from ase import Atoms

from scgo.initialization.geometry_helpers import validate_cluster_structure
from scgo.initialization.initialization_config import (
    CONNECTIVITY_FACTOR,
    MIN_DISTANCE_FACTOR_DEFAULT,
)


def validate_combined_cluster_structure(
    atoms: Atoms,
    *,
    min_distance_factor: float = MIN_DISTANCE_FACTOR_DEFAULT,
    connectivity_factor: float = CONNECTIVITY_FACTOR,
    check_clashes: bool = True,
    check_connectivity: bool = True,
    use_mic: bool = False,
) -> tuple[bool, str]:
    """Validate the full system using the same rules as cluster initialization.

    Checks optional clash screening and that all atoms lie in one connected
    component under covalent-radius-based edge thresholds.

    Args:
        atoms: The Atoms object to validate
        min_distance_factor: Factor for minimum distance checks
        connectivity_factor: Factor for connectivity threshold
        check_clashes: Whether to check for atomic clashes
        check_connectivity: Whether to check connectivity
        use_mic: If True, use minimum image convention for distance calculations

    Returns:
        ``(True, "")`` if valid, else ``(False, error_message)``.
    """
    return validate_cluster_structure(
        atoms,
        min_distance_factor,
        connectivity_factor,
        check_clashes=check_clashes,
        check_connectivity=check_connectivity,
        use_mic=use_mic,
    )
