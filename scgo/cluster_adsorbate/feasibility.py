"""Heuristic checks for whether adsorbate fragments can be placed on a core."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from ase import Atoms

from scgo.cluster_adsorbate.sites import (
    compute_surface_site_candidates,
    count_site_candidates,
)
from scgo.exceptions import SCGOValidationError
from scgo.initialization.atomic_radii import get_covalent_radius, get_vdw_radius


def count_adsorption_site_candidates(atoms: Atoms) -> int:
    """Return a conservative count of distinct adsorption sites on a 3D structure."""
    if len(atoms) == 0:
        return 0
    return count_site_candidates(compute_surface_site_candidates(atoms))


def estimate_fragment_footprint_radius(fragment: Atoms) -> float:
    """Estimate minimum separation radius (Å) for placing a rigid fragment."""
    if len(fragment) == 0:
        return 0.0
    pos = fragment.get_positions()
    anchor = pos[0]
    max_extent = float(np.max(np.linalg.norm(pos - anchor, axis=1)))
    symbols = fragment.get_chemical_symbols()
    vdw_pad = max(get_vdw_radius(s) for s in symbols)
    return max_extent + vdw_pad


def _estimate_symbol_sphere_radius(symbols: Sequence[str]) -> float:
    if not symbols:
        return 0.0
    return max(get_covalent_radius(s) for s in symbols)


def validate_adsorbate_placement_feasibility(
    core_symbols: Sequence[str],
    adsorbate_fragment_lengths: Sequence[int],
    adsorbate_fragments: Sequence[Atoms] | None = None,
    *,
    context: str = "",
) -> None:
    """Raise :class:`~scgo.exceptions.SCGOValidationError` when fragment count likely exceeds placement capacity.

    This is a fast, geometry-agnostic heuristic used before global optimization.
    It does not replace runtime placement validation.
    """
    prefix = f"{context}: " if context else ""
    lengths = [n for n in (int(x) for x in adsorbate_fragment_lengths) if n > 0]
    n_frags = len(lengths)
    if n_frags == 0:
        return

    n_core = len(core_symbols)

    if n_core == 0:
        if n_frags == 1:
            return
        if adsorbate_fragments is not None and len(adsorbate_fragments) == n_frags:
            radii = [
                estimate_fragment_footprint_radius(frag) for frag in adsorbate_fragments
            ]
            min_sep = 2.0 * max(radii) if radii else 0.0
            span = sum(2.0 * r for r in radii) + max(0, n_frags - 1) * min_sep
            if span > 40.0:
                raise SCGOValidationError(
                    f"{prefix}adsorbate-only system with {n_frags} fragments appears "
                    f"too extended for reliable placement (estimated span {span:.1f} Å)."
                )
        return

    max_by_size = max(1, n_core) if n_core < 4 else max(1, (n_core + 1) // 2)

    if n_frags > max_by_size:
        raise SCGOValidationError(
            f"{prefix}cannot place {n_frags} adsorbate fragments on a core with "
            f"{n_core} atoms: heuristic site capacity is about {max_by_size}."
        )

    if adsorbate_fragments is not None and len(adsorbate_fragments) == n_frags:
        core_radius = _estimate_symbol_sphere_radius(core_symbols) * (
            n_core ** (1.0 / 3.0)
        )
        frag_radii = [
            estimate_fragment_footprint_radius(frag) for frag in adsorbate_fragments
        ]
        largest_frag = max(frag_radii) if frag_radii else 0.0
        if largest_frag > 2.5 * core_radius and n_frags > 1:
            raise SCGOValidationError(
                f"{prefix}largest adsorbate fragment footprint ({largest_frag:.1f} Å) "
                f"is large compared to the {n_core}-atom core; multiple fragments "
                "are unlikely to fit without overlap."
            )
