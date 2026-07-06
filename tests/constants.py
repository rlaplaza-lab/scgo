"""Shared test constants and configurations.

This module centralizes test configuration values to avoid duplication
across test files and make it easier to adjust test parameters globally.
"""

import os

# Random seed configurations for reproducibility testing
REPRODUCIBILITY_SEEDS = [42, 123, 456, 789, 1001, 2022, 3033, 4044, 5055, 6066]

# Strict floating-point tolerances for deterministic reproducibility assertions
REPRODUCIBILITY_RTOL = 1e-10
REPRODUCIBILITY_ATOL = 1e-12

# Looser tolerances for smoke tests with real EMT relaxations (float noise)
SMOKE_RTOL = 1e-5
SMOKE_ATOL = 1e-8

# Cluster size configurations
SMALL_SIZES = [4, 6, 8, 10]
MEDIUM_SIZES = [15, 20, 25, 30]
LARGE_SIZES = [40, 50, 60]

# Composition configurations
MIXED_COMPOSITIONS = {
    "PtAu": lambda n: ["Pt", "Au"] * (n // 2) + ["Pt"] * (n % 2),
    "PtPd": lambda n: ["Pt", "Pd"] * (n // 2) + ["Pt"] * (n % 2),
    "AuPdPt": lambda n: (["Au", "Pd", "Pt"] * ((n // 3) + 1))[:n],
}

# Initialization modes
INITIALIZATION_MODES = ["random_spherical", "seed+growth", "template", "smart"]

# Batch testing configurations
BATCH_TEST_SAMPLES = int(os.environ.get("SCGO_BATCH_TEST_SAMPLES", "100"))
BATCH_TEST_SAMPLES_SLOW = 15  # For slow batch tests
UNIQUENESS_THRESHOLD = 0.8  # Minimum uniqueness ratio (80%)

# Diversity testing thresholds
DIVERSITY_THRESHOLD_MIN = 0.6  # Minimum diversity threshold
DIVERSITY_THRESHOLD_DEFAULT = 0.7  # Default diversity threshold
DIVERSITY_TEST_SAMPLES_SMALL = 10
DIVERSITY_TEST_SAMPLES_MEDIUM = 15
DIVERSITY_TEST_SAMPLES_LARGE = 20

# RNG seed range for random sampling
RNG_SEED_RANGE = (0, 100000)

# Geometry parameters
MIN_DISTANCE_FACTOR_DEFAULT = 0.4

# EMT reference physics (shared across physics assertion helpers)
EMT_PT2_BOND_ANG = 2.26
EMT_PT2_BOND_TOL_ANG = 0.02
EMT_H2_BARRIER_EV = (2.0, 5.0)
NN_DISTANCE_BAND = (0.9, 1.3)
TS_FMAX_CONVERGED = 0.15
ADSORPTION_HEIGHT_TOLERANCE_ANG = 0.15
PT_O_DISTANCE_ANG = (1.8, 2.4)
# EMT barrier for Pt4 tetrahedron <-> planar isomerization (eV).
PT4_EMT_BARRIER_EV = (0.05, 8.0)
