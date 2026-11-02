"""Centralized configuration constants for cluster initialization.

This module provides named constants for magic numbers scattered throughout
the initialization system, improving maintainability and documentation.

Notable tunables (library-internal, not GO preset keys):

- ``MASS_FIRST_PLACEMENT_PROB``: fraction of placement attempts that use
  heavy-element-first growth order; the remainder use exploratory strategies.
- ``CONNECTIVITY_FACTOR`` (1.2), ``BLMIN_RATIO_DEFAULT`` (0.7),
  ``CLASH_HARD_CAP_FACTOR`` (0.4), ``BOND_LENGTH_SCALE`` (1.0).
"""

from __future__ import annotations

# Placement and retry parameters
MAX_PLACEMENT_ATTEMPTS_PER_ATOM = 500
MAX_CONNECTIVITY_RETRIES = 10
MAX_CONSECUTIVE_FAILURES = 50

# Bonded when d <= (r_i + r_j) * factor. Clash when d < (r_i + r_j) * floor.
CONNECTIVITY_FACTOR = 1.2
BOND_LENGTH_SCALE = 1.0  # placement and template nearest-neighbor length
BLMIN_RATIO_DEFAULT = 0.7  # steric floor for growth, GA, deposition, stored minima
CLASH_HARD_CAP_FACTOR = 0.4  # prefilter and NEB prescreen; never go below
MIN_DISTANCE_FACTOR_DEFAULT = CLASH_HARD_CAP_FACTOR
PLACEMENT_RADIUS_SCALING_DEFAULT = BOND_LENGTH_SCALE
SEED_CLASH_FACTOR = BLMIN_RATIO_DEFAULT
NONBONDED_GAP_SCALE = 1.5  # unbound fragment gap, in bond-cutoff units
SURFACE_LAYER_THICKNESS_SCALE = 2.0  # top layer, in radii of the largest slab species
BOND_REPAIR_INSET = 0.95  # pull a repaired contact inside the bond cutoff
# Fraction of placements that use mass-biased ordering; remainder uses exploratory strategies.
MASS_FIRST_PLACEMENT_PROB = 0.65

# Cell and vacuum
VACUUM_DEFAULT = 10.0
MAX_REASONABLE_CELL_SIDE = 1000.0  # Maximum reasonable cell side (Å)

# Boltzmann sampling
BOLTZMANN_TEMPERATURE_MIN = 0.05
BOLTZMANN_TEMPERATURE_MAX = 0.5
ENERGY_SPREAD_TOLERANCE = 1e-6  # Tolerance for energy spread comparison
ENERGY_SPREAD_DIVISOR = 10.0  # Divisor for adaptive temperature calculation

# Convex hull and geometry
CONVEX_HULL_PERTURBATION_SCALE = 0.1
CONVEX_HULL_VOLUME_TOLERANCE = 1e-6  # Tolerance for degenerate convex hulls

# Magic numbers and templates
# Magic number detection tolerance (atoms)
# Clusters within this many atoms of a magic number are considered "near"
MAGIC_NUMBER_TOLERANCE = 2

# Magic numbers for high-symmetry cluster sizes, grouped by structure family.
# Membership is what matters (the original combined these as a ``sorted(set(...))``),
# so this single literal preserves every number from the per-family lists above.
# Note that the decahedral group is unordered and includes both 54 and 55.
#
# Provenance: the icosahedral/decahedral/cuboctahedral families follow the LJ/Doye
# global-minimum and magic-number surveys (Wales, Doye, etc.). These numbers only
# gate the OPTIONAL high-symmetry templates and their allocation weights; usage is
# bounded (a small fixed template set), so they never drive unbounded allocation
# or physics defaults (CONNECTIVITY_FACTOR, MAGIC_NUMBERS-independent).
MAGIC_NUMBERS = {
    # Platonic solid vertices: tetrahedron, octahedron, cube, icosahedron, dodecahedron
    4,
    6,
    8,
    12,
    20,
    # Tetrahedral shells (vertices +1/+2/+3 shells)
    10,
    35,
    # Octahedral (ASE Octahedron generator: length=2/3/4/5, cutoff=0)
    19,
    44,
    85,
    # Cubic (n×n×n cubes)
    27,
    64,
    125,
    # Icosahedral (Doye group study)
    13,
    23,
    26,
    29,
    34,
    39,
    45,
    46,
    49,
    55,
    58,
    61,
    71,
    78,
    127,
    147,
    309,
    # Decahedral (unordered: includes both 54 and 55)
    7,
    54,
    105,
    116,
    156,
    207,
    # Archimedean: cuboctahedron (12), cuboctahedron+center (13), truncated octahedron (24)
    24,
}

# Strategy and diversity
# Number of seed combination strategies available
SEED_COMBINATION_STRATEGY_COUNT = (
    5  # 0=Boltzmann, 1=low-energy, 2=high-energy, 3=mid-energy, 4=random
)

# Number of growth order strategies available
GROWTH_ORDER_STRATEGY_COUNT = 6  # 0=random, 1=by_element, 2=alternating, 3=size_based, 4=element_clustering, 5=composition_balance

# Metropolis allocation scaling parameters
# These control logarithmic scaling for strategy allocation
TEMPLATE_BASE_PCT = 0.10  # Base percentage for template allocation
TEMPLATE_PREFACTOR = 2.0  # Scaling prefactor for templates (higher = more allocation with more templates)
SEED_BASE_PCT = 0.10  # Base percentage for seed+growth allocation
SEED_PREFACTOR = (
    1.5  # Scaling prefactor for seeds (higher = more allocation with more combinations)
)

# Internal caching and selection
_FIND_SMALLER_CANDIDATES_CACHE_VERSION = 4
_MAX_CANDIDATES_PER_FORMULA = 10000
_COMPOSITION_CACHE_NS = "composition"

# Exact-match reuse tier scaling (mirrors SEED_* for the bounded additive tier)
EXACT_BASE_PCT = 0.10
EXACT_PREFACTOR = 1.5

# Template diversity enhancement
TEMPLATE_ROTATION_CANDIDATES = (
    3  # Number of rotation variants to generate per template for diversity
)

# Template weight configuration (base weights per template type)
TEMPLATE_BASE_WEIGHTS: dict[str, float] = {
    "icosahedron": 1.5,
    "decahedron": 1.3,
    "cuboctahedron": 1.0,
    "octahedron": 1.0,
    "truncated_octahedron": 1.0,
    "cube": 0.8,
    "tetrahedron": 0.8,
}

# Multi-element composition penalty factor
# Subtracted from every template weight for multi-element clusters
# (high-symmetry motifs are less favorable for mixed compositions)
MULTI_ELEMENT_TEMPLATE_PENALTY = 0.9

# Diversity boost factor for underrepresented template types
# Promotes exploration of less common template types
TEMPLATE_DIVERSITY_BOOST_FACTOR = 0.15

# Tolerance and thresholds
LINEAR_GEOMETRY_TOLERANCE = 1e-4  # Tolerance for linear geometry detection
ROTATION_AXIS_TOLERANCE = 1e-10  # Tolerance for rotation axis normalization
POSITION_COMPARISON_TOLERANCE_FACTOR = 0.05  # 5% of bond length for position comparison
SMART_FILTERING_PERTURBATION_SCALE = (
    0.3  # Reduced perturbation scale for smart facet filtering
)

# Relaxation and scaling
PLACEMENT_RELAXATION_FACTOR = 0.25  # Relaxation factor for placement attempts
MIN_DISTANCE_THRESHOLD_LOW = 0.4  # Lower threshold for min_distance_factor
MIN_DISTANCE_THRESHOLD_HIGH = 0.8  # Upper threshold for min_distance_factor
CONNECTIVITY_SUGGESTION_BUFFER = 1.05  # Buffer for connectivity factor suggestions

# Physical and computational
PACKING_EFFICIENCY_FCC_HCP = 0.74  # FCC/HCP packing efficiency
KDTREE_THRESHOLD = 50  # Use KDTree for clusters with >= 50 atoms
