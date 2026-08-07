"""Constants used across SCGO."""

from __future__ import annotations

PENALTY_ENERGY: float = 1.0e6
"""Penalty energy (eV) for failed optimizations."""

MIN_ATOMIC_DISTANCE_WARNING: float = 0.5
"""Minimum atomic distance (Å) for warnings."""

BOLTZMANN_K_EV_PER_K: float = 8.617e-5
"""Boltzmann constant (eV/K)."""

DEFAULT_ENERGY_TOLERANCE: float = 0.02
"""Default energy tolerance (eV)."""

DEFAULT_COMPARATOR_TOL: float = 0.015
"""Cumulative structural-difference tolerance (dimensionless).

Population-weighted sum of relative sorted-distance deviations (eq. 2 of
Vilhelmsen & Hammer, PRL 108, 126101), not a length.
"""

DEFAULT_PAIR_COR_MAX: float = 0.7
"""Largest allowed single sorted-distance difference (Å).

Absolute distance-difference tolerance (eq. 3 of Vilhelmsen & Hammer,
PRL 108, 126101) — not a correlation coefficient.
"""

DEFAULT_PAIR_COR_CUM_DIFF: float = DEFAULT_COMPARATOR_TOL
"""Cumulative pair-correlation difference tolerance (same as DEFAULT_COMPARATOR_TOL)."""

DEFAULT_NEB_TANGENT_METHOD: str = "improvedtangent"
"""ASE :class:`ase.mep.neb.NEB` tangent method used by default."""

SURFACE_GA_MIN_LOCAL_RELAX_STEPS: int = 400
"""Minimum local-relaxation steps for GA with ``surface_config`` (slab adsorption)."""
