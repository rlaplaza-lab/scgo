"""Central knob resolving ``n_jobs`` style parallelism (-1 / -2 / positive).

This module owns the single project-wide parallelism default. Library entry
points take ``n_jobs: int | None = None`` and weave the default in through
:func:`resolve_n_jobs`, instead of repeating a literal default in every
signature, so the default can be changed in exactly one place.

**One knob, several CPU stages.** Runner ``params`` carry a single top-level
``n_jobs`` that every CPU-bound stage inherits: GA population initialization,
GA offspring construction, and post-GO Hessian/force validation. The per-stage
keys (``optimizer_params["ga"]["n_jobs_population_init"]``,
``optimizer_params["ga"]["n_jobs_offspring"]``, and top-level
``validation_n_jobs``) stay available as overrides: ``None`` means "inherit the
top-level ``n_jobs``", any explicit value wins for that stage only. That
cascade is applied with :func:`inherit_n_jobs`. GPU/TorchSim NEB batching
(``use_parallel_neb`` / ``parallel_neb_max_*``) is a separate, unrelated knob.

Call sites that own a pool convert the resolved setting into a concrete worker
count with :func:`resolve_n_jobs_for_tasks`, which also caps the pool at the
number of tasks so small batches never spawn idle workers.
"""

from __future__ import annotations

import os

from scgo.exceptions import (
    SCGOValidationError,
)

DEFAULT_N_JOBS = 1
"""Project-wide parallelism default.

Single-threaded unless the user opts in. This mirrors the convention used by
scikit-learn / joblib (``n_jobs=None`` means one worker), so the library never
silently oversubscribes the host by spawning one worker per logical CPU on top of
the internal BLAS / MACE / TorchSIM thread pools. Callers that want parallelism
pass ``-1`` (all CPUs) or ``-2`` (all CPUs except one) explicitly, or set their
own worker count.
"""


def validate_n_jobs(n_jobs: int, name: str = "n_jobs") -> None:
    """Reject ``n_jobs`` values outside ``-1``, ``-2``, or ``>= 1``.

    Raises:
        SCGOValidationError: If ``n_jobs`` is not a supported value.
    """
    if n_jobs not in (-1, -2) and n_jobs < 1:
        raise SCGOValidationError(f"{name} must be -1, -2, or >= 1, got {n_jobs}")


def resolve_n_jobs(n_jobs: int | None, name: str = "n_jobs") -> int:
    """Resolve an optional ``n_jobs`` to a concrete, validated setting.

    ``None`` means "use the project default" (``DEFAULT_N_JOBS``); any
    other value is validated and returned unchanged.

    Raises:
        SCGOValidationError: If ``n_jobs`` is not ``None``, ``-1``, ``-2``, or
            ``>= 1``.
    """
    if n_jobs is None:
        return DEFAULT_N_JOBS
    validate_n_jobs(n_jobs, name)
    return n_jobs


def resolve_n_jobs_to_workers(n_jobs: int | None) -> int:
    """Map batch-parallel ``n_jobs`` to a concrete worker count (``>= 1``).

    Semantics match :func:`~scgo.initialization.create_initial_cluster_batch`:

    - ``None``: the project default (``DEFAULT_N_JOBS``).
    - ``1``: sequential (callers using ``max_workers == 1`` stay single-threaded).
    - ``> 1``: use that many workers.
    - ``-1``: all logical CPUs.
    - ``-2``: all logical CPUs except one.

    Raises:
        SCGOValidationError: If ``n_jobs`` is not ``None``, ``-1``, ``-2``, or
            ``>= 1``.
    """
    resolved = resolve_n_jobs(n_jobs)
    cpu = os.cpu_count() or 1
    if resolved == -1:
        return cpu
    if resolved == -2:
        return max(1, cpu - 1)
    return resolved


def inherit_n_jobs(stage_value: int | None, top_level_value: int | None) -> int | None:
    """Apply the per-stage override / top-level ``n_jobs`` inheritance rule.

    Args:
        stage_value: Per-stage setting (e.g. ``n_jobs_population_init``).
            ``None`` means "inherit".
        top_level_value: Top-level ``params["n_jobs"]``, itself possibly ``None``.

    Returns:
        ``stage_value`` when it is set, otherwise ``top_level_value``. The result
        may be ``None``, which downstream :func:`resolve_n_jobs` turns into
        ``DEFAULT_N_JOBS``.
    """
    return stage_value if stage_value is not None else top_level_value


def resolve_n_jobs_for_tasks(n_jobs: int | None, n_tasks: int) -> int:
    """Resolve ``n_jobs`` to a worker count for ``n_tasks`` queued tasks.

    Wraps :func:`resolve_n_jobs_to_workers` with the cap every pool owner needs:
    never start more workers than there are tasks, and never fewer than one.

    Args:
        n_jobs: Parallelism setting (``None`` uses ``DEFAULT_N_JOBS``).
        n_tasks: Number of tasks about to be dispatched.

    Returns:
        Worker count in ``[1, max(1, n_tasks)]``.

    Raises:
        SCGOValidationError: If ``n_jobs`` is not ``None``, ``-1``, ``-2``, or
            ``>= 1``.
    """
    return max(1, min(resolve_n_jobs_to_workers(n_jobs), max(1, n_tasks)))
