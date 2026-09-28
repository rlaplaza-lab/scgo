"""CUDA / GPU runtime helpers used across calculators and TS search."""

from __future__ import annotations


def is_cuda_oom_error(exc: BaseException) -> bool:
    """True if ``exc`` is a CUDA OOM (type name or message); no torch import."""
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    return "out of memory" in str(exc).lower()
