"""Global optimization algorithms for atomic clusters.

This package contains implementations of various global optimization algorithms
adapted for atomic cluster structure search:

- Simple: Single optimization for 1-2 atom clusters
- Basin Hopping: Random perturbations with Metropolis acceptance
- Genetic Algorithm: Population-based evolution with batched relaxations

.. warning::
    These functions are primarily for internal use. Most users should use the
    high-level API in :mod:`scgo.runner_api` (e.g., :func:`~scgo.runner_api.run_go`) instead
    of calling these algorithm functions directly.
"""

from __future__ import annotations

from .basinhopping_go import bh_go
from .geneticalgorithm_go_torchsim import ga_go
from .simple_go import simple_go

__all__ = [
    "bh_go",
    "ga_go",
    "simple_go",
]
