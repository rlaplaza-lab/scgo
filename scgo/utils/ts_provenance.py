"""Shared provenance fields for TS / NEB JSON outputs.

``schema_version`` tracks the provenance header; it is **3** for current SCGO
releases. On-disk layouts are documented in ``docs/source/quickstart.rst``
(*On-disk layout*).
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from scgo.utils.logging import get_logger

_logger = get_logger(__name__)
_version_warned: set[str] = set()

TS_OUTPUT_SCHEMA_VERSION = 3
CLUSTER_ADSORBATE_OUTPUT_SCHEMA_VERSION = 1


def ts_output_provenance(*, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return flat metadata merged into TS/NEB JSON and GO ``results_summary.json``."""
    meta: dict[str, Any] = {
        "schema_version": TS_OUTPUT_SCHEMA_VERSION,
        "scgo_version": package_version("scgo"),
        "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python_version": sys.version.split()[0],
    }
    if extra:
        meta.update(extra)
    return meta


def package_version(dist_name: str) -> str:
    try:
        return version(dist_name)
    except PackageNotFoundError:
        if dist_name not in _version_warned:
            _logger.warning(
                "Could not resolve package version for %r; provenance will record "
                "'unknown'",
                dist_name,
            )
            _version_warned.add(dist_name)
        return "unknown"


def is_cuda_oom_error(exc: BaseException) -> bool:
    """True if ``exc`` is a CUDA OOM error (exception type or message pattern)."""
    import torch.cuda

    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True

    # Fallback to message pattern matching
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg
