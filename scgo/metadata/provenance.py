"""Shared provenance header for on-disk JSON artifacts.

``schema_version`` tracks the single output-JSON provenance header (currently 4).
This is distinct from the SQLite DB stamp ``schema_version`` in
:mod:`scgo.metadata.db_stamp`. All output-JSON artifacts (GO/TS/NEB metadata,
timing, and cluster-adsorbate provenance) share this one version; there are no
per-artifact schema-version keys.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from scgo._version import __version__ as SCGO_VERSION
from scgo.utils.logging import get_logger

logger = get_logger(__name__)
_version_warned: set[str] = set()

OUTPUT_JSON_SCHEMA_VERSION = 4


def output_json_provenance(*, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return flat provenance fields merged into GO/TS/NEB JSON artifacts."""
    meta: dict[str, Any] = {
        "schema_version": OUTPUT_JSON_SCHEMA_VERSION,
        "scgo_version": package_version("scgo"),
        "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python_version": sys.version.split()[0],
    }
    if extra:
        meta.update(extra)
    return meta


def package_version(dist_name: str) -> str:
    """Resolve a distribution version for provenance JSON.

    For ``scgo``, prefer the in-tree :data:`~scgo.__version__` so editable
    checkouts are not stuck on a stale ``dist-info`` after a bump.
    """
    if dist_name == "scgo":
        return str(SCGO_VERSION)
    try:
        return version(dist_name)
    except PackageNotFoundError:
        if dist_name not in _version_warned:
            logger.warning(
                "Could not resolve package version for %r; provenance will record "
                "'unknown'",
                dist_name,
            )
            _version_warned.add(dist_name)
        return "unknown"


def is_cuda_oom_error(exc: BaseException) -> bool:
    """True if ``exc`` is a CUDA OOM error (exception type or message pattern)."""
    # Avoid importing torch here (keeps provenance import-light for CPU paths).
    # ``torch.cuda.OutOfMemoryError`` is a ``RuntimeError`` subclass whose type
    # name is "OutOfMemoryError"; matching on the type name covers it without a
    # torch import. "cuda error: out of memory" is a subset of "out of memory".
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    return "out of memory" in str(exc).lower()
