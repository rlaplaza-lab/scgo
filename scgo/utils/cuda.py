"""CUDA / GPU runtime helpers used across calculators and TS search."""

from __future__ import annotations

import contextlib
import gc
import sys
from typing import Any

try:
    import torch
except ImportError:  # Torch is optional; the OOM classifier does not need it.
    torch = None  # type: ignore[assignment]


def is_cuda_oom_error(exc: BaseException) -> bool:
    """True if ``exc`` is a CUDA OOM (message); no torch import.

    Warp's CUDA memory pool reports a failed ``cudaMallocAsync`` as
    ``Failed to allocate N bytes on device 'cuda:0'``. The words "out of
    memory" are printed by the native runtime and are not part of that
    exception. A CPU allocation failure uses ``device 'cpu'`` and is ignored.
    The message must name CUDA or a GPU: CPU and CUDA share ``OutOfMemoryError``.

    The walk follows ``__cause__`` when set, otherwise ``__context__`` when
    that context is not suppressed.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if _exception_text_is_cuda_oom(current):
            return True
        current = _next_exception(current)
    return False


def _next_exception(exc: BaseException) -> BaseException | None:
    if exc.__cause__ is not None:
        return exc.__cause__
    if exc.__context__ is not None and not exc.__suppress_context__:
        return exc.__context__
    return None


def _exception_text_is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    if ("cuda" in text or "gpu" in text) and (
        "out of memory" in text or "out of gpu memory" in text
    ):
        return True
    return "failed to allocate" in text and "device 'cuda" in text


def _cuda_memory_snapshot() -> str:
    """One line of PyTorch and Warp allocator stats, or a short fallback."""
    parts: list[str] = []
    torch_line = _torch_memory_line()
    if torch_line is not None:
        parts.append(torch_line)
    warp_line = _warp_memory_line()
    if warp_line is not None:
        parts.append(warp_line)
    if not parts:
        return "cuda memory stats unavailable"
    return "; ".join(parts)


def _torch_memory_line() -> str | None:
    if torch is None:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
    except Exception:  # noqa: BLE001 - a failed query must not abort the release
        return None
    return f"torch allocated={allocated} reserved={reserved} free={free} total={total}"


def _warp_memory_line() -> str | None:
    warp = sys.modules.get("warp")
    if warp is None:
        return None
    try:
        device = warp.get_device()
        if not device.is_cuda or not warp.is_mempool_enabled(device):
            return None
        return (
            "warp pool used="
            f"{warp.get_mempool_used_mem_current(device)} "
            f"high={warp.get_mempool_used_mem_high(device)} "
            f"release_threshold={warp.get_mempool_release_threshold(device)}"
        )
    except Exception:  # noqa: BLE001 - a failed query must not abort the release
        return None


def _release_warp_mempool() -> None:
    """Return unused Warp CUDA-pool blocks to the driver.

    PyTorch's caching allocator and Warp's pool take memory from the same
    device. A later Warp allocation of a few dozen bytes fails when the
    driver has only a small gap left, even if PyTorch's reserved cache is
    idle. Synchronizing with a zero release threshold gives that gap back.
    The caller's threshold is restored, including when ``synchronize`` raises.
    """
    warp = sys.modules.get("warp")
    if warp is None:
        return
    try:
        device = warp.get_device()
        if not device.is_cuda or not warp.is_mempool_enabled(device):
            return
        previous = warp.get_mempool_release_threshold(device)
    except (RuntimeError, ValueError):
        return
    try:
        if previous != 0:
            warp.set_mempool_release_threshold(device, 0)
        warp.synchronize()
    except (RuntimeError, ValueError):
        return
    finally:
        if previous != 0:
            with contextlib.suppress(RuntimeError, ValueError):
                warp.set_mempool_release_threshold(device, previous)


def _release_torch_cache(logger: Any | None) -> None:
    """Drop PyTorch's caching allocator, then collect Python garbage."""
    if torch is not None and torch.cuda.is_available():
        with contextlib.suppress(RuntimeError):
            torch.cuda.synchronize()
        with contextlib.suppress(RuntimeError):
            torch.cuda.empty_cache()
        if logger is not None:
            with contextlib.suppress(RuntimeError):
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                logger.debug(
                    "Post-cleanup CUDA memory: allocated=%s reserved=%s",
                    allocated,
                    reserved,
                )
                if reserved - allocated > 100_000_000:
                    logger.debug(
                        "CUDA fragmentation detected: reserved-allocated=%s bytes",
                        reserved - allocated,
                    )
    gc.collect()


def release_gpu_memory(logger: Any | None = None) -> str:
    """Release the Warp pool and the PyTorch cache. Return ``before -> after``.

    The snapshot cannot abort this call. Warp's release threshold is restored
    even when the pool sync fails.
    """
    before = _cuda_memory_snapshot()
    _release_warp_mempool()
    _release_torch_cache(logger)
    after = _cuda_memory_snapshot()
    return f"{before} -> {after}"
