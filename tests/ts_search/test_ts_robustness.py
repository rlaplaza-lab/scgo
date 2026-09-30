"""Robustness tests for TS search: OOM handling and malformed metadata."""

from __future__ import annotations

import json
import tempfile

import pytest
import torch

from scgo.ts_search.transition_state_run import run_transition_state_search
from scgo.ts_search.ts_network import save_ts_network_metadata
from tests.helpers import create_cu2_ts_searches_dir


def test_save_ts_network_metadata_skips_malformed_success():
    """A `status=='success'` entry missing numeric fields should be skipped."""
    ts_results = [
        {
            "pair_id": "0_1",
            "status": "success",
            "reactant_energy": -5.0,
            "product_energy": -4.8,
            "ts_energy": None,  # malformed
            "barrier_height": None,
            "neb_converged": True,
            "n_images": 5,
        },
        {
            "pair_id": "1_2",
            "status": "success",
            "reactant_energy": -4.8,
            "product_energy": -4.6,
            "ts_energy": -4.3,
            "barrier_height": 0.5,
            "neb_converged": True,
            "n_images": 5,
        },
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_ts_network_metadata(
            ts_results, tmpdir, composition=["Cu", "Cu", "Cu"], minima_count=3
        )

        with open(path) as f:
            meta = json.load(f)

        # The malformed successful entry should be skipped
        assert meta["statistics"]["successful_ts"] == 1
        assert len(meta["ts_connections"]) == 1


def test_run_transition_state_search_handles_cuda_oom(monkeypatch):
    """Simulate a per-pair CUDA OOM and ensure the campaign continues and
    GPU cleanup is attempted. This test sets up a minimal mock DB locally.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        create_cu2_ts_searches_dir(tmpdir, n_minima=2)

        def fake_find_transition_state(*args, **kwargs):
            raise RuntimeError(
                "CUDA out of memory [scgo-simulated-failure]. Tried to allocate ..."
            )

        monkeypatch.setattr(
            "scgo.ts_search.transition_state_run.find_transition_state",
            fake_find_transition_state,
        )

        cleaned = {"called": False}

        def fake_cleanup(logger=None):
            cleaned["called"] = True

        monkeypatch.setattr(
            "scgo.ts_search.transition_state_run.cleanup_torch_cuda", fake_cleanup
        )

        results = run_transition_state_search(
            composition=["Cu", "Cu"],
            system_type="gas_cluster",
            output_dir=tmpdir,
            params={"calculator": "EMT", "calculator_kwargs": {}},
            verbosity=0,
            max_pairs=1,
            neb_n_images=3,
            neb_fmax=0.5,
            neb_steps=10,
        )

        assert isinstance(results, list)
        assert any(r.get("status") == "failed" for r in results)
        assert cleaned["called"] is True
        for r in results:
            ts = r.get("transition_state")
            if ts is not None:
                assert ts.calc is None


def test_pairwise_cleanup_even_without_errors(monkeypatch):
    """GPU cleanup should be attempted after every pair, not only on OOMs.

    We simulate a minimal two-pair search and verify our patched
    `cleanup_torch_cuda` hook is invoked once per pair.  This regression test
    guards against future edits that accidentally remove the unconditional
    cleanup added after prior regression investigations.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        create_cu2_ts_searches_dir(tmpdir, n_minima=2)

        calls = {"count": 0}

        def fake_cleanup(logger=None):
            calls["count"] += 1

        monkeypatch.setattr(
            "scgo.ts_search.transition_state_run.cleanup_torch_cuda", fake_cleanup
        )

        results = run_transition_state_search(
            composition=["Cu", "Cu"],
            system_type="gas_cluster",
            output_dir=tmpdir,
            params={"calculator": "EMT", "calculator_kwargs": {}},
            verbosity=0,
            max_pairs=2,
            neb_n_images=3,
            neb_fmax=0.5,
            neb_steps=10,
        )

        assert isinstance(results, list)
        assert calls["count"] >= 2


def test_transition_state_results_do_not_retain_calculators(tmp_path):
    """Returned ts_results must not carry an attached calculator object."""
    params = {"calculator": "EMT", "calculator_kwargs": {}}
    results = run_transition_state_search(
        composition=["H", "H"],
        system_type="gas_cluster",
        output_dir=tmp_path,
        params=params,
        verbosity=0,
        max_pairs=1,
        neb_n_images=3,
        neb_fmax=0.5,
        neb_steps=10,
    )
    assert isinstance(results, list)
    for r in results:
        ts = r.get("transition_state")
        if ts is not None:
            assert ts.calc is None


@pytest.mark.requires_cuda
@pytest.mark.requires_mace
def test_gpu_memory_does_not_grow(tmp_path, monkeypatch):
    """Repeated campaigns with a GPU-backed dummy calculator should not leak.

    This test monkeypatches ``get_calculator_class`` to return a simple
    ASE calculator that allocates a small CUDA tensor in its constructor.  We
    verify that memory usage immediately after two successive runs remains
    bounded (within a few MB) to catch regressions where calculators are
    retained by result structures.
    """
    from ase.calculators.emt import EMT

    from scgo.utils.run_helpers import get_calculator_class as _orig_get

    class GpuDummy(EMT):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            if torch.cuda.is_available():
                # allocate a tiny tensor to tag this instance
                self._buf = torch.zeros(1, device="cuda")

    def fake_get(name):
        if name == "GPUDUMMY":
            return GpuDummy
        return _orig_get(name)

    monkeypatch.setattr("scgo.utils.run_helpers.get_calculator_class", fake_get)
    # transition_state_run imports the function directly, so patch its copy too
    monkeypatch.setattr(
        "scgo.ts_search.transition_state_run.get_calculator_class", fake_get
    )

    params = {"calculator": "GPUDUMMY", "calculator_kwargs": {}}

    # baseline memory
    before = torch.cuda.memory_allocated()
    run_transition_state_search(
        composition=["Cu", "Cu"],
        system_type="gas_cluster",
        output_dir=str(tmp_path),
        params=params,
        verbosity=0,
        max_pairs=1,
        neb_n_images=3,
        neb_fmax=0.5,
        neb_steps=10,
    )
    import gc

    gc.collect()
    mid = torch.cuda.memory_allocated()
    run_transition_state_search(
        composition=["Cu", "Cu"],
        system_type="gas_cluster",
        output_dir=str(tmp_path),
        params=params,
        verbosity=0,
        max_pairs=1,
        neb_n_images=3,
        neb_fmax=0.5,
        neb_steps=10,
    )
    gc.collect()
    after = torch.cuda.memory_allocated()

    # allow a small tolerance for driver bookkeeping
    assert mid <= before + 10_000_000
    assert after <= before + 10_000_000
