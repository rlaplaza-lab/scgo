"""Global optimization to find minima.

This package contains the core workflow for global optimization of atomic
clusters: single-run execution, deduplication across datetime-tagged runs,
Hessian validation, and result persistence.
"""

from __future__ import annotations

from scgo.minima_search.core import run_trials, scgo

__all__ = ["run_trials", "scgo"]
