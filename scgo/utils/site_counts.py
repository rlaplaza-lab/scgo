"""Shared helper for batch site-type tallies."""

from __future__ import annotations

from threading import Lock


def increment_site_type_count(
    counts: dict[str, int] | None,
    site_type: object,
    lock: Lock | None = None,
) -> None:
    """Increment ``counts[site_type]`` when ``counts`` is set and the key exists.

    No-op when ``counts`` is ``None`` or ``site_type`` is not a ``str``.
    """
    if counts is None or not isinstance(site_type, str):
        return
    if lock is not None:
        with lock:
            if site_type in counts:
                counts[site_type] += 1
    elif site_type in counts:
        counts[site_type] += 1
