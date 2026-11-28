"""Shared end-of-run minima collection for BH / GA / simple GO."""

from __future__ import annotations

from typing import Any

from ase import Atoms

from scgo.database.sync import PRESET_HPC, database_retry
from scgo.metadata.atoms import filter_by_tags, get_tag
from scgo.utils.helpers import extract_minima_from_database


def collect_run_minima(
    da: Any,
    *,
    run_id: str | None = None,
    ga_eligible_default: bool = True,
    operation_name: str | None = None,
    exception_types: tuple[type[BaseException], ...] | None = None,
) -> list[tuple[float, Atoms]]:
    """Load relaxed candidates, filter by ``run_id`` / ``ga_eligible``, return minima.

    GA uses ``ga_eligible_default=False``; BH/simple use ``True`` for legacy rows.
    """
    kwargs: dict[str, Any] = {"config": PRESET_HPC}
    if operation_name is not None:
        kwargs["operation_name"] = operation_name
    if exception_types is not None:
        kwargs["exception_types"] = exception_types

    candidates = database_retry(da.get_all_relaxed_candidates, **kwargs)
    if run_id is not None:
        candidates = filter_by_tags(candidates, run_id=run_id)
    candidates = [
        c
        for c in candidates
        if bool(get_tag(c, "ga_eligible", default=ga_eligible_default))
    ]
    return extract_minima_from_database(candidates)
