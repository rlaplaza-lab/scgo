"""Shared batch site-type counting helper.

Both cluster-adsorbate hierarchical assembly and surface deposition keep a
running tally of how many accepted structures came from each site type. This
helper unifies the previously duplicated ``_record_batch_site_type`` logic
from :mod:`scgo.cluster_adsorbate.hierarchical` and
:mod:`scgo.surface.deposition`.
"""

from __future__ import annotations

from threading import Lock


def increment_site_type_count(
    counts: dict[str, int] | None,
    site_type: str,
    lock: Lock | None = None,
) -> None:
    """Increment the running count for ``site_type`` in ``counts``.

    Both callers resolve their own representative site type before calling
    this helper, so the count key is always a concrete string. The call is a
    no-op when ``counts`` is ``None`` or ``site_type`` is not a string (e.g.
    a missing tag resolves to ``None``).

    When ``lock`` is provided, the increment is guarded so the counter is safe
    to update from parallel worker threads.

    Args:
        counts: Mutable ``site_type -> count`` map, or ``None`` to skip.
        site_type: The resolved site type key to increment.
        lock: Optional lock guarding ``counts`` under parallelism.
    """
    if counts is None or not isinstance(site_type, str):
        return
    if lock is not None:
        with lock:
            if site_type in counts:
                counts[site_type] += 1
    elif site_type in counts:
        counts[site_type] += 1
