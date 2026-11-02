"""CUDA out-of-memory classification and the single GPU-release helper."""

import sys

import pytest

from scgo.exceptions import SCGORuntimeError
from scgo.utils.cuda import is_cuda_oom_error, release_gpu_memory
from scgo.utils.run_helpers import cleanup_torch_cuda


class OutOfMemoryError(RuntimeError):
    """Stand-in for a CUDA out-of-memory message."""


def test_is_cuda_oom_error_matches_torch_and_warp_messages():
    assert is_cuda_oom_error(OutOfMemoryError("CUDA out of memory"))
    assert is_cuda_oom_error(
        RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")
    )
    assert is_cuda_oom_error(
        RuntimeError("Failed to allocate 108 bytes on device 'cuda:0'")
    )
    wrapped = SCGORuntimeError(
        "TorchSim relaxation ran out of GPU memory during optimization."
    )
    assert is_cuda_oom_error(wrapped)
    chained = SCGORuntimeError("relax failed")
    chained.__cause__ = RuntimeError("Failed to allocate 108 bytes on device 'cuda:0'")
    assert is_cuda_oom_error(chained)


def test_is_cuda_oom_error_ignores_cpu_and_unrelated_failures():
    assert not is_cuda_oom_error(
        RuntimeError("Failed to allocate 108 bytes on device 'cpu'")
    )
    assert not is_cuda_oom_error(
        OutOfMemoryError("DefaultCPUAllocator: not enough memory")
    )
    assert not is_cuda_oom_error(RuntimeError("MPS backend out of memory"))
    assert not is_cuda_oom_error(OutOfMemoryError("out of memory"))
    assert not is_cuda_oom_error(RuntimeError("neighbor list failed"))
    assert not is_cuda_oom_error(ValueError("max_metric 10"))


def test_is_cuda_oom_error_follows_implicit_context_not_suppressed_context():
    try:
        raise RuntimeError("Failed to allocate 108 bytes on device 'cuda:0'")
    except RuntimeError:
        try:
            raise RuntimeError("relax failed")
        except RuntimeError as wrapped:
            implicit = wrapped
    assert is_cuda_oom_error(implicit)

    try:
        raise RuntimeError("Failed to allocate 108 bytes on device 'cuda:0'")
    except RuntimeError:
        try:
            raise RuntimeError("relax failed") from None
        except RuntimeError as wrapped:
            suppressed = wrapped
    assert not is_cuda_oom_error(suppressed)

    explicit_other = RuntimeError("neighbor list failed")
    explicit_other.__cause__ = RuntimeError("unrelated")
    explicit_other.__context__ = RuntimeError(
        "Failed to allocate 108 bytes on device 'cuda:0'"
    )
    explicit_other.__suppress_context__ = True
    assert not is_cuda_oom_error(explicit_other)

    cycle = RuntimeError("neighbor list failed")
    cycle.__cause__ = cycle
    assert not is_cuda_oom_error(cycle)


class _Device:
    def __init__(self, *, is_cuda: bool) -> None:
        self.is_cuda = is_cuda


class _Warp:
    def __init__(self, *, is_cuda: bool = True, enabled: bool = True) -> None:
        self.device = _Device(is_cuda=is_cuda)
        self.enabled = enabled
        self.threshold = 4096
        self.sets: list[int] = []
        self.synced = 0
        self.fail_sync = False
        self.seen_during_sync: list[int] = []
        self.used = 3
        self.high = 4
        self.fail_stats = False

    def get_device(self):
        return self.device

    def is_mempool_enabled(self, device):
        return self.enabled

    def get_mempool_release_threshold(self, device):
        return self.threshold

    def set_mempool_release_threshold(self, device, value):
        self.sets.append(value)
        self.threshold = value

    def synchronize(self):
        self.synced += 1
        self.seen_during_sync.append(self.threshold)
        if self.fail_sync:
            raise RuntimeError("sync failed")

    def get_mempool_used_mem_current(self, device):
        if self.fail_stats:
            raise RuntimeError("stat failed")
        return self.used

    def get_mempool_used_mem_high(self, device):
        if self.fail_stats:
            raise RuntimeError("stat failed")
        return self.high


def _install_warp(monkeypatch, warp: _Warp | None) -> None:
    if warp is None:
        monkeypatch.delitem(sys.modules, "warp", raising=False)
        return
    monkeypatch.setitem(sys.modules, "warp", warp)


def _patch_torch_cuda(monkeypatch, *, queries_raise: bool = False) -> dict[str, int]:
    import torch

    calls = {"synchronize": 0, "empty_cache": 0}

    def _sync():
        calls["synchronize"] += 1

    def _empty():
        calls["empty_cache"] += 1

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    if queries_raise:

        def _raise(*_a, **_k):
            raise RuntimeError("cuda query failed")

        monkeypatch.setattr(torch.cuda, "mem_get_info", _raise)
        monkeypatch.setattr(torch.cuda, "memory_allocated", _raise)
        monkeypatch.setattr(torch.cuda, "memory_reserved", _raise)
    else:
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (7, 9))
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 1)
        monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 2)
    monkeypatch.setattr(torch.cuda, "synchronize", _sync)
    monkeypatch.setattr(torch.cuda, "empty_cache", _empty)
    return calls


def test_release_gpu_memory_restores_threshold_after_sync(monkeypatch):
    warp = _Warp()
    _install_warp(monkeypatch, warp)
    calls = _patch_torch_cuda(monkeypatch)

    report = release_gpu_memory()

    assert warp.seen_during_sync == [0]
    assert warp.threshold == 4096
    assert warp.sets == [0, 4096]
    assert calls == {"synchronize": 1, "empty_cache": 1}
    assert "torch allocated=1 reserved=2 free=7 total=9" in report
    assert "warp pool used=3 high=4 release_threshold=" in report
    assert " -> " in report


def test_release_gpu_memory_restores_threshold_when_sync_raises(monkeypatch):
    warp = _Warp()
    warp.fail_sync = True
    _install_warp(monkeypatch, warp)
    _patch_torch_cuda(monkeypatch)

    report = release_gpu_memory()

    assert warp.seen_during_sync == [0]
    assert warp.threshold == 4096
    assert isinstance(report, str)
    assert " -> " in report


def test_release_gpu_memory_threshold_zero_only_syncs(monkeypatch):
    warp = _Warp()
    warp.threshold = 0
    _install_warp(monkeypatch, warp)
    _patch_torch_cuda(monkeypatch)

    release_gpu_memory()

    assert warp.sets == []
    assert warp.synced == 1
    assert warp.threshold == 0


@pytest.mark.parametrize(
    ("warp", "synced"),
    [
        (None, 0),
        (_Warp(is_cuda=False), 0),
        (_Warp(enabled=False), 0),
    ],
)
def test_release_gpu_memory_skips_warp_when_pool_is_unavailable(
    monkeypatch, warp, synced
):
    _install_warp(monkeypatch, warp)
    _patch_torch_cuda(monkeypatch)

    report = release_gpu_memory()

    assert " -> " in report
    if warp is not None:
        assert warp.synced == synced
        assert warp.sets == []


def test_release_gpu_memory_survives_a_failing_snapshot(monkeypatch):
    warp = _Warp()
    warp.fail_stats = True
    _install_warp(monkeypatch, warp)
    calls = _patch_torch_cuda(monkeypatch, queries_raise=True)

    report = release_gpu_memory()

    assert calls == {"synchronize": 1, "empty_cache": 1}
    assert warp.synced == 1
    assert warp.threshold == 4096
    assert report == "cuda memory stats unavailable -> cuda memory stats unavailable"


def test_cleanup_torch_cuda_delegates_to_release_gpu_memory(monkeypatch):
    seen: list[object] = []

    def _release(logger=None):
        seen.append(logger)
        return "before -> after"

    monkeypatch.setattr("scgo.utils.run_helpers.release_gpu_memory", _release)
    sentinel = object()
    cleanup_torch_cuda(logger=sentinel)
    assert seen == [sentinel]
