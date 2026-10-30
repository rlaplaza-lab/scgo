"""Intra-generation CPU/GPU overlap for TorchSim GA.

Double-buffers offspring construction with batched relaxation on a dedicated
GPU-owning thread. SQLite / population updates stay on the caller thread.

The GA enables this path only when ``batch_size`` is set (micro-batches that
must be relaxed separately). With ``batch_size=None`` the generation keeps a
single full ``relax_batch`` so TorchSim InFlight stays packed.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import torch
from ase import Atoms

from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer
from scgo.exceptions import SCGORuntimeError


def resolve_pipeline_chunk(n_offspring: int, batch_size: int | None) -> int:
    """Wave size for double-buffering: ``batch_size`` if set, else ``n // 2``."""
    n = max(1, int(n_offspring))
    if batch_size is not None:
        return max(1, min(n, int(batch_size)))
    return max(1, n // 2)


def relaxer_supports_cpu_gpu_pipeline(relaxer: Any) -> bool:
    """True for CUDA :class:`TorchSimBatchRelaxer` (not ASE / CPU)."""
    if not isinstance(relaxer, TorchSimBatchRelaxer):
        return False
    device = getattr(relaxer, "device", None)
    if device is None:
        return False
    return torch.device(device).type == "cuda"


@dataclass
class GpuRelaxJob:
    originals: list[Atoms]
    prepared: list[Atoms]


@dataclass
class GpuRelaxResult:
    originals: list[Atoms]
    relaxed_results: list[tuple[float, Atoms]]
    error: BaseException | None = None
    relax_s: float = 0.0


class GpuRelaxWorker:
    """Thread that owns ``relaxer.relax_batch`` calls for one GA run."""

    _STOP = object()

    def __init__(self, relaxer: Any) -> None:
        self._relaxer = relaxer
        self._jobs: queue.Queue[Any] = queue.Queue()
        self._results: queue.Queue[GpuRelaxResult] = queue.Queue()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="scgo-ga-gpu-relax", daemon=True
        )
        self._thread.start()

    def submit(self, job: GpuRelaxJob) -> None:
        if self._thread is None:
            raise SCGORuntimeError("GpuRelaxWorker.submit called before start()")
        self._jobs.put(job)

    def wait_result(self) -> GpuRelaxResult:
        return self._results.get()

    def shutdown(self) -> None:
        if self._thread is None:
            return
        self._jobs.put(self._STOP)
        self._thread.join()
        self._thread = None

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            if job is self._STOP:
                break
            assert isinstance(job, GpuRelaxJob)
            t0 = perf_counter()
            try:
                relaxed = self._relaxer.relax_batch(job.prepared)
                if len(relaxed) != len(job.originals):
                    raise SCGORuntimeError(
                        "TorchSim relaxer returned mismatched batch size"
                    )
                self._results.put(
                    GpuRelaxResult(
                        originals=job.originals,
                        relaxed_results=list(relaxed),
                        relax_s=perf_counter() - t0,
                    )
                )
            except BaseException as exc:
                self._results.put(
                    GpuRelaxResult(
                        originals=job.originals,
                        relaxed_results=[],
                        error=exc,
                        relax_s=perf_counter() - t0,
                    )
                )


def run_double_buffered_with_worker(
    *,
    worker: GpuRelaxWorker,
    produce_wave: Callable[[], list[Atoms] | None],
    prepare_batch: Callable[[list[Atoms]], list[Atoms]],
    commit_result: Callable[[GpuRelaxResult], None],
) -> dict[str, float]:
    """Consume wave N on the GPU while producing wave N+1 on the caller thread.

    ``produce_wave`` returns inserted unrelaxed candidates, or ``None`` when
    production is finished. An empty list means no successes this wave.
    """
    timings = {
        "produce_s": 0.0,
        "relax_s": 0.0,
        "commit_s": 0.0,
        "wall_s": 0.0,
        "overlapped_s": 0.0,
    }
    wall0 = perf_counter()
    worker.start()
    try:
        t0 = perf_counter()
        current = produce_wave()
        timings["produce_s"] += perf_counter() - t0

        while current is not None:
            if not current:
                t0 = perf_counter()
                current = produce_wave()
                timings["produce_s"] += perf_counter() - t0
                continue

            worker.submit(
                GpuRelaxJob(
                    originals=list(current),
                    prepared=prepare_batch(list(current)),
                )
            )
            t0 = perf_counter()
            nxt = produce_wave()
            produce_s = perf_counter() - t0
            timings["produce_s"] += produce_s

            result = worker.wait_result()
            timings["relax_s"] += float(result.relax_s)
            timings["overlapped_s"] += min(float(result.relax_s), produce_s)
            if result.error is not None:
                raise result.error
            t0 = perf_counter()
            commit_result(result)
            timings["commit_s"] += perf_counter() - t0
            current = nxt
    finally:
        worker.shutdown()

    timings["wall_s"] = perf_counter() - wall0
    return timings
