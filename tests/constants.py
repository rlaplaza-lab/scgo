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

# Cluster size configurations
_FULL_INIT_MATRIX = os.environ.get("SCGO_FULL_INIT_MATRIX", "0") == "1"
SMALL_SIZES = [4, 6, 8, 10] if _FULL_INIT_MATRIX else [4]
MEDIUM_SIZES = [15, 20, 25, 30] if _FULL_INIT_MATRIX else [15]
LARGE_SIZES = [40, 50, 60] if _FULL_INIT_MATRIX else [40]

# Composition configurations
MIXED_COMPOSITIONS = {
    "PtAu": lambda n: ["Pt", "Au"] * (n // 2) + ["Pt"] * (n % 2),
    "PtPd": lambda n: ["Pt", "Pd"] * (n // 2) + ["Pt"] * (n % 2),
    "AuPdPt": lambda n: (["Au", "Pd", "Pt"] * ((n // 3) + 1))[:n],
}

# Initialization modes
INITIALIZATION_MODES = ["random_spherical", "seed+growth", "template", "smart"]

# Batch testing configurations
BATCH_TEST_SAMPLES = int(os.environ.get("SCGO_BATCH_TEST_SAMPLES", "25"))
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
# Nearest-neighbour distance band as a fraction of the sum of covalent radii.
# Pre-flight (tests/physics/test_reference_emt.py, fixture seed 42) yields a Pt3
# NN scaled ratio of 1.20; the upper bound is tightened from 1.3 to 1.25 to keep
# a small safety margin above that real value. NOTE: the plan's suggested
# (0.95, 1.15) was rejected -- it breaks the sacred Pt3 NN check (1.20 > 1.15).
NN_DISTANCE_BAND = (0.9, 1.25)

# Per-atom force tolerance a converged TS must satisfy. Pin the literal so a
# production loosen of ``_TS_NEB_FMAX`` fails loudly (see drift test in
# tests/param_presets/test_param_presets.py). 0.20 eV/A is the soft-MEP floor;
# tighter values collapse interior saddles to endpoints.
TS_FMAX_CONVERGED = 0.20
ADSORPTION_HEIGHT_TOLERANCE_ANG = 0.1
PT_O_DISTANCE_ANG = (1.8, 2.4)
# NOTE: the plan proposed tightening hi to 2.2, but Pre-flight shows relaxed
# EMT Pt-O distances reach ~2.3-2.5 A (test_oh_relax_reports_connected_structure_emt
# fails at 2.2). 2.4 is the tightest band real chemistry satisfies; kept as-is.
# EMT barrier for Pt4 tetrahedron <-> planar isomerization (eV). Pre-flight
# captures ~0.763 eV (NEB endpoint-max, status=failed). Tightened from
# (0.05, 8.0): the 8 eV ceiling admitted physically-trivial barriers while the
# 0.05 floor admitted near-zero (no real transition) barriers. Measured value
# leaves margin at both ends.
PT4_EMT_BARRIER_EV = (0.5, 3.0)
# Generic MLIP barrier band (eV). Not an EMT accuracy reference: it only guards
# against physically-implausible / negative barriers. Used by the GPU example
# matrix (MACE + UPET), which runs a different PES than EMT. Tightened from
# (0.0, 10.0): the 10 eV ceiling admitted absurd barriers. GPU-only (skipped in
# local CPU runs) -- requires Kaggle GPU CI re-verification.
MLIP_BARRIER_EV = (0.0, 6.0)
