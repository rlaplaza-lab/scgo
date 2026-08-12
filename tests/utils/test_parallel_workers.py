"""Tests for :mod:`scgo.utils.parallel_workers` CPU-parallelism helpers."""

from __future__ import annotations

import pytest

from scgo.utils.parallel_workers import (
    DEFAULT_N_JOBS,
    inherit_n_jobs,
    resolve_n_jobs_for_tasks,
    resolve_n_jobs_to_workers,
)


@pytest.mark.parametrize(
    "stage, top, expected",
    [
        (None, None, None),
        (None, -2, -2),
        (None, 1, 1),
        (1, -2, 1),
        (-2, -2, -2),
        (3, None, 3),
    ],
)
def test_inherit_n_jobs(
    stage: int | None, top: int | None, expected: int | None
) -> None:
    assert inherit_n_jobs(stage, top) == expected


@pytest.mark.parametrize(
    "n_jobs, n_tasks, expected",
    [
        (None, 1, 1),  # None -> default, floor at 1
        (None, 8, DEFAULT_N_JOBS),  # default is 1
        (1, 8, 1),
        (3, 8, 3),
        (3, 2, 2),  # capped at n_tasks
        (-1, 2, 2),  # all CPUs capped at n_tasks
        (-2, 2, 2),
        (-1, 1000, resolve_n_jobs_to_workers(-1)),
        (-2, 1000, resolve_n_jobs_to_workers(-2)),
    ],
)
def test_resolve_n_jobs_for_tasks(
    n_jobs: int | None, n_tasks: int, expected: int
) -> None:
    assert resolve_n_jobs_for_tasks(n_jobs, n_tasks) == expected
    # Contract: always >= 1 and never more than the number of tasks.
    assert resolve_n_jobs_for_tasks(n_jobs, n_tasks) >= 1
    assert resolve_n_jobs_for_tasks(n_jobs, n_tasks) <= max(1, n_tasks)


def test_resolve_n_jobs_for_tasks_rejects_bad_value() -> None:
    from scgo.exceptions import SCGOValidationError

    with pytest.raises(SCGOValidationError):
        resolve_n_jobs_for_tasks(0, 4)
