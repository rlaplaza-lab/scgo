"""Tests for GA CPU/GPU double-buffer pipeline helpers."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ase import Atoms
from ase.calculators.emt import EMT

from scgo.algorithms import ga_go
from scgo.algorithms.ga_cpu_gpu_pipeline import (
    GpuRelaxJob,
    GpuRelaxWorker,
    relaxer_supports_cpu_gpu_pipeline,
    resolve_pipeline_chunk,
    run_double_buffered_with_worker,
)
from scgo.calculators.ase_batch_relaxer import AseBatchRelaxer
from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer
from tests.helpers import MockRelaxer


def test_resolve_pipeline_chunk():
    assert resolve_pipeline_chunk(10, None) == 5
    assert resolve_pipeline_chunk(1, None) == 1
    assert resolve_pipeline_chunk(10, 3) == 3
    assert resolve_pipeline_chunk(2, 10) == 2


def test_relaxer_supports_pipeline_gate():
    assert relaxer_supports_cpu_gpu_pipeline(MockRelaxer()) is False
    assert (
        relaxer_supports_cpu_gpu_pipeline(AseBatchRelaxer(calculator=None, max_steps=1))
        is False
    )
    assert relaxer_supports_cpu_gpu_pipeline(SimpleNamespace(device="cuda")) is False

    class _CudaTs(TorchSimBatchRelaxer):
        pass

    obj = object.__new__(_CudaTs)
    obj.device = "cuda"
    assert relaxer_supports_cpu_gpu_pipeline(obj) is True
    obj.device = "cpu"
    assert relaxer_supports_cpu_gpu_pipeline(obj) is False


def test_double_buffer_overlaps_produce_with_relax():
    class _SleepRelaxer:
        def __init__(self) -> None:
            self.intervals: list[tuple[float, float]] = []

        def relax_batch(self, prepared):
            t0 = time.perf_counter()
            time.sleep(0.08)
            self.intervals.append((t0, time.perf_counter()))
            return [(float(i), a.copy()) for i, a in enumerate(prepared)]

    relaxer = _SleepRelaxer()
    waves = [
        [Atoms("H", positions=[[0, 0, 0]]), Atoms("H", positions=[[1, 0, 0]])],
        [Atoms("H", positions=[[2, 0, 0]]), Atoms("H", positions=[[3, 0, 0]])],
    ]
    idx = {"i": 0}
    produce_intervals: list[tuple[float, float]] = []
    committed: list[int] = []

    def produce():
        i = idx["i"]
        if i >= len(waves):
            return None
        idx["i"] = i + 1
        t0 = time.perf_counter()
        time.sleep(0.08)
        produce_intervals.append((t0, time.perf_counter()))
        return waves[i]

    worker = GpuRelaxWorker(relaxer)
    try:
        timings = run_double_buffered_with_worker(
            worker=worker,
            produce_wave=produce,
            prepare_batch=lambda batch: [a.copy() for a in batch],
            commit_result=lambda result: committed.append(len(result.originals)),
        )
    finally:
        worker.shutdown()
    assert committed == [2, 2]
    assert timings["overlapped_s"] > 0.04
    p1, r0 = produce_intervals[1], relaxer.intervals[0]
    assert min(p1[1], r0[1]) - max(p1[0], r0[0]) > 0.03


def test_gpu_relax_worker_propagates_errors():
    class _Boom:
        def relax_batch(self, prepared):
            raise RuntimeError("gpu boom")

    worker = GpuRelaxWorker(_Boom())
    worker.start()
    try:
        atoms = Atoms("H", positions=[[0, 0, 0]])
        worker.submit(GpuRelaxJob(originals=[atoms], prepared=[atoms.copy()]))
        result = worker.wait_result()
        assert result.error is not None
        assert "gpu boom" in str(result.error)
    finally:
        worker.shutdown()


def test_ga_go_mock_relaxer_stays_serial(tmp_path, rng):
    minima = ga_go(
        composition=["Pt", "Pt", "Pt"],
        output_dir=str(tmp_path / "ga_no_pipe"),
        calculator=EMT(),
        relaxer=MockRelaxer(max_steps=1),
        niter=1,
        population_size=4,
        offspring_fraction=0.5,
        niter_local_relaxation=1,
        batch_size=2,
        n_jobs_offspring=1,
        write_timing_json=True,
        detailed_timing=True,
        rng=rng,
    )
    assert isinstance(minima, list)


def test_ga_go_batch_size_none_reports_no_pipeline_overlap(tmp_path, rng):
    """Scaler default (batch_size=None) must stay on the serial full-batch path."""
    out = Path(tmp_path) / "ga_scaler_serial"
    minima = ga_go(
        composition=["Pt", "Pt", "Pt"],
        output_dir=str(out),
        calculator=EMT(),
        relaxer=MockRelaxer(max_steps=1),
        niter=1,
        population_size=8,
        offspring_fraction=0.5,
        niter_local_relaxation=1,
        batch_size=None,
        n_jobs_offspring=1,
        write_timing_json=True,
        detailed_timing=True,
        rng=rng,
    )
    assert isinstance(minima, list)
    timing = json.loads(next(out.rglob("timing.json")).read_text())
    for gen in timing["per_generation"]:
        assert float(gen["timings_s"].get("pipeline_overlapped_s", 0.0)) == 0.0


def test_ga_go_logs_pipeline_enabled_when_gated_on(tmp_path, rng, caplog):
    """With batch_size set and CUDA gate true, log the pipeline enable once."""
    with (
        patch(
            "scgo.algorithms.geneticalgorithm_go_torchsim.relaxer_supports_cpu_gpu_pipeline",
            return_value=True,
        ),
        caplog.at_level(
            logging.INFO,
            logger="scgo.algorithms.geneticalgorithm_go_torchsim",
        ),
    ):
        ga_go(
            composition=["Pt", "Pt", "Pt"],
            output_dir=str(tmp_path / "ga_pipe_log"),
            calculator=EMT(),
            relaxer=MockRelaxer(max_steps=1),
            niter=1,
            population_size=4,
            offspring_fraction=0.5,
            niter_local_relaxation=1,
            batch_size=2,
            n_jobs_offspring=1,
            verbosity=1,
            rng=rng,
        )
    assert caplog.text.count("CPU/GPU offspring pipeline enabled (batch_size=2)") == 1
