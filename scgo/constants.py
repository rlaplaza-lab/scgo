"""Constants used across SCGO."""

from __future__ import annotations

PENALTY_ENERGY: float = 1.0e6
"""Penalty energy (eV) for failed optimizations."""

MIN_ATOMIC_DISTANCE_WARNING: float = 0.5
"""Minimum atomic distance (Å) for warnings."""

BOLTZMANN_K_EV_PER_K: float = 8.617e-5
"""Boltzmann constant (eV/K)."""

DEFAULT_ENERGY_TOLERANCE: float = 0.1
"""Default energy tolerance (eV) for uniqueness / deduplication.

Independently relaxed copies of the same isomer often differ by tens of meV
at the default local-relaxation ``fmax`` of :data:`DEFAULT_FMAX_THRESHOLD`
(0.05 eV/Å); the wider 0.1 eV window collapses those near-copies without
merging distinct isomers that usually sit farther apart.
"""

DEFAULT_COMPARATOR_TOL: float = 0.05
"""Cumulative structure-comparator difference tolerance (normalized, unitless)."""

DEFAULT_PAIR_COR_MAX: float = 0.7
"""Max single interatomic-distance difference (Å) for GO uniqueness comparison."""

DEFAULT_TS_PAIR_COR_MAX: float = 0.1
"""Max single interatomic-distance difference (Å) for TS pair near-dupe gating.

Tighter than :data:`DEFAULT_PAIR_COR_MAX` because TS pairing must reject near-
duplicates before NEB, while GO uniqueness tolerates more structural variation.
"""

DEFAULT_CROSS_WEIGHT: float = 1.0
"""Default weight on cross-block distance terms in block-aware uniqueness."""

DEFAULT_SUPPORTED_SLAB_WEIGHT: float = 0.2
"""Default mobile-slab weight for supported-deposit (``surface_cluster*``) types.

Relaxed support layers barely move relative to globally optimized deposits, so
their near-constant distances must not dominate deposit/adsorbate uniqueness.
Set ``comparator_component_weights={"mobile_slab": 0.0}`` to exclude them.
"""

SUPPORTED_CLUSTER_PAIR_COR_MAX: float = 0.45
"""Tighter max-distance gate for supported-deposit (``surface_cluster*``) types.

Block-aware fingerprints keep deposit/adsorbate differences undiluted by slab
padding, so a tighter single-distance gate is useful. Applied only when the
effective value still equals :data:`DEFAULT_PAIR_COR_MAX`. Cumulative
tolerance uses the shared :data:`DEFAULT_COMPARATOR_TOL`.
"""

DEFAULT_FMAX_THRESHOLD: float = 0.05
"""Default local-relaxation / Hessian-validation force threshold (eV/Å)."""

DEFAULT_IMAG_FREQ_THRESHOLD: float = 50.0
"""Default imaginary-frequency magnitude cutoff for Hessian validation (cm^-1)."""

DEFAULT_NEB_TANGENT_METHOD: str = "improvedtangent"
"""ASE :class:`ase.mep.neb.NEB` tangent method used by default."""

NEB_RIGID_FRAGMENT_TOLERANCE_A: float = 0.05
"""Endpoint distance agreement (Å) for treating an adsorbate fragment as rigid.

Used by rigid-pose interpolation and FixBondLengths restore. Distinct from the
diagnostic ``neb_interpolation_bond_tolerance_a`` (default 0.5 Å), which only
warns when interior images stretch frozen bonds.
"""

SURFACE_GA_MIN_LOCAL_RELAX_STEPS: int = 400
"""Minimum local-relaxation steps for GA with ``surface_config`` (slab adsorption)."""
