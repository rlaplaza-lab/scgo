"""Full-system composition for slab + adsorbate (matches GA template ordering)."""

from __future__ import annotations

from scgo.surface.config import SurfaceSystemConfig


def full_adsorbate_slab_composition(
    adsorbate: list[str], surface_config: SurfaceSystemConfig
) -> list[str]:
    """Return slab chemical symbols first, then the adsorbate symbols.

    Matches the atom ordering of the :func:`scgo.algorithms.ga_go` surface
    template (slab prefix, then the mobile adsorbate).
    """
    return list(surface_config.slab.get_chemical_symbols()) + list(adsorbate)
