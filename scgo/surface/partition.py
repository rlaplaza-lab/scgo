"""Search-mobile partition for slab-as-target system types.

For ``surface`` / ``surface_adsorbate``, GA/BH keep the trailing-``n_top``
contract by ordering atoms as::

    [fixed bottom layers] + [mobile top N layers] (+ [adsorbates])

Search mobility matches the same layer policy used by
:func:`scgo.surface.constraints.attach_slab_constraints`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms

from scgo.exceptions import SCGOValidationError
from scgo.surface.config import SurfaceSystemConfig
from scgo.surface.constraints import _layer_indices_by_clustering

__all__ = [
    "SlabSearchPartition",
    "resolve_slab_search_partition",
    "reorder_slab_fixed_then_mobile",
    "prepare_slab_search_surface_config",
    "validate_slab_search_config",
    "count_distinct_slab_layers",
]


def count_distinct_slab_layers(
    slab: Atoms,
    *,
    surface_normal_axis: int,
) -> int:
    """Return the number of distinct coordinate layers along the surface normal."""
    if len(slab) == 0:
        return 0
    positions = np.asarray(slab.get_positions())
    rounded = np.round(positions[:, surface_normal_axis], decimals=6)
    return int(len(np.unique(rounded)))


def validate_slab_search_config(config: SurfaceSystemConfig) -> None:
    """Require a layer policy suitable for slab-as-target search modes."""
    if config.fix_all_slab_atoms:
        raise SCGOValidationError(
            "slab-as-target system types require fix_all_slab_atoms=False "
            "so top layers can be searched by GA/BH."
        )
    if (
        config.n_relax_top_slab_layers is None
        and config.n_fix_bottom_slab_layers is None
    ):
        raise SCGOValidationError(
            "slab-as-target system types require n_relax_top_slab_layers or "
            "n_fix_bottom_slab_layers so the fixed/mobile slab partition is defined."
        )


@dataclass(frozen=True)
class SlabSearchPartition:
    """Fixed vs search-mobile split of a slab (and optional adsorbates)."""

    n_slab: int
    n_fixed: int
    n_mobile_slab: int
    fixed_indices: tuple[int, ...]
    mobile_slab_indices: tuple[int, ...]
    mobile_slab_symbols: tuple[str, ...]

    @property
    def n_top_bare(self) -> int:
        """Trailing mobile count with no adsorbates."""
        return self.n_mobile_slab

    def n_top_with_adsorbates(self, n_adsorbate: int) -> int:
        """Count mobile top atoms plus adsorbates for the partition."""
        if n_adsorbate < 0:
            raise SCGOValidationError("n_adsorbate must be >= 0")
        return self.n_mobile_slab + int(n_adsorbate)


def resolve_slab_search_partition(
    config: SurfaceSystemConfig,
) -> SlabSearchPartition:
    """Resolve fixed bottom vs mobile top-layer indices for ``config.slab``."""
    validate_slab_search_config(config)
    slab = config.slab
    n_slab = len(slab)
    axis = int(config.surface_normal_axis)
    positions = np.asarray(slab.get_positions())
    symbols = list(slab.get_chemical_symbols())

    if config.n_relax_top_slab_layers is not None:
        mobile = _layer_indices_by_clustering(
            positions,
            axis,
            n_layers=int(config.n_relax_top_slab_layers),
            from_top=True,
        )
    else:
        assert config.n_fix_bottom_slab_layers is not None
        fixed = _layer_indices_by_clustering(
            positions,
            axis,
            n_layers=int(config.n_fix_bottom_slab_layers),
            from_top=False,
        )
        mobile = set(range(n_slab)) - fixed

    if not mobile:
        raise SCGOValidationError(
            "slab-as-target partition has no mobile top-layer atoms; "
            "check n_relax_top_slab_layers / n_fix_bottom_slab_layers."
        )
    if len(mobile) >= n_slab:
        raise SCGOValidationError(
            "slab-as-target partition requires at least one fixed bottom atom; "
            "increase slab layers or reduce n_relax_top_slab_layers."
        )

    # Preserve relative order within each group so reordering is stable.
    mobile_idx = tuple(i for i in range(n_slab) if i in mobile)
    fixed_idx = tuple(i for i in range(n_slab) if i not in mobile)
    return SlabSearchPartition(
        n_slab=n_slab,
        n_fixed=len(fixed_idx),
        n_mobile_slab=len(mobile_idx),
        fixed_indices=fixed_idx,
        mobile_slab_indices=mobile_idx,
        mobile_slab_symbols=tuple(symbols[i] for i in mobile_idx),
    )


def reorder_slab_fixed_then_mobile(
    slab: Atoms,
    partition: SlabSearchPartition,
) -> Atoms:
    """Return a copy of ``slab`` ordered ``[fixed...][mobile top...]``."""
    if len(slab) != partition.n_slab:
        raise SCGOValidationError(
            f"reorder_slab_fixed_then_mobile: len(slab)={len(slab)} != "
            f"partition.n_slab={partition.n_slab}"
        )
    order = list(partition.fixed_indices) + list(partition.mobile_slab_indices)
    out = slab[order]
    out.set_cell(slab.get_cell())
    out.set_pbc(slab.get_pbc())
    return out


def prepare_slab_search_surface_config(
    config: SurfaceSystemConfig,
) -> tuple[SurfaceSystemConfig, SlabSearchPartition]:
    """Return a config whose slab is ordered ``[fixed...][mobile...]`` plus partition.

    Call once at GO/TS setup for ``slab_is_search_target`` system types so GA/BH
    trailing-``n_top`` operators and FixAtoms share the same layout.
    """
    partition = resolve_slab_search_partition(config)
    reordered = reorder_slab_fixed_then_mobile(config.slab, partition)
    # Rebuild partition on the reordered slab (fixed are a contiguous prefix).
    new_config = SurfaceSystemConfig(
        slab=reordered,
        name=config.name,
        adsorption_height_min=config.adsorption_height_min,
        adsorption_height_max=config.adsorption_height_max,
        surface_normal_axis=config.surface_normal_axis,
        fix_all_slab_atoms=False,
        n_fix_bottom_slab_layers=config.n_fix_bottom_slab_layers,
        n_relax_top_slab_layers=config.n_relax_top_slab_layers,
        comparator_use_mic=config.comparator_use_mic,
        cluster_init_vacuum=config.cluster_init_vacuum,
        init_mode=config.init_mode,
        max_placement_attempts=config.max_placement_attempts,
        structure_connectivity_factor=config.structure_connectivity_factor,
    )
    new_partition = resolve_slab_search_partition(new_config)
    # After reorder, fixed indices must be 0..n_fixed-1.
    if list(new_partition.fixed_indices) != list(range(new_partition.n_fixed)):
        raise SCGOValidationError(
            "prepare_slab_search_surface_config failed to produce a contiguous "
            "fixed prefix after slab reorder."
        )
    if list(new_partition.mobile_slab_indices) != list(
        range(new_partition.n_fixed, new_partition.n_slab)
    ):
        raise SCGOValidationError(
            "prepare_slab_search_surface_config failed to produce a contiguous "
            "mobile suffix after slab reorder."
        )
    return new_config, new_partition
