"""GA relaxation batching: accumulate-and-flush behavior (``relax_batch_target``)."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT

from scgo.algorithms import ga_go
from scgo.algorithms.geneticalgorithm_go_torchsim import _resolve_relax_batch_target
from scgo.exceptions import SCGOValidationError


class RecordingRelaxer:
    """MockRelaxer that records the size of every relax_batch call."""

    def __init__(self, max_steps: int | None = None):
        self.max_steps = max_steps
        self.batch_sizes: list[int] = []

    def relax_batch(self, batch: list[Atoms]):
        self.batch_sizes.append(len(batch))
        return [(float(i) * 0.1, a.copy()) for i, a in enumerate(batch)]


def test_resolve_relax_batch_target_auto_uses_population_size():
    assert (
        _resolve_relax_batch_target(
            "auto", population_size=16, n_offspring=4, batch_size=None
        )
        == 16
    )


def test_resolve_relax_batch_target_legacy_none_uses_offspring():
    assert (
        _resolve_relax_batch_target(
            None, population_size=16, n_offspring=4, batch_size=None
        )
        == 4
    )
    assert (
        _resolve_relax_batch_target(
            0, population_size=16, n_offspring=4, batch_size=None
        )
        == 4
    )


def test_resolve_relax_batch_target_respects_batch_size_cap():
    assert (
        _resolve_relax_batch_target(
            "auto", population_size=16, n_offspring=4, batch_size=6
        )
        == 6
    )
    assert (
        _resolve_relax_batch_target(
            32, population_size=16, n_offspring=4, batch_size=None
        )
        == 32
    )


def test_resolve_relax_batch_target_rejects_bad_string():
    with pytest.raises(SCGOValidationError, match="relax_batch_target"):
        _resolve_relax_batch_target(
            "everything", population_size=8, n_offspring=2, batch_size=None
        )


def _run_ga(tmp_path, rng, relaxer, *, relax_batch_target, niter=4):
    return ga_go(
        composition=["Pt"] * 4,
        output_dir=str(tmp_path),
        calculator=EMT(),
        relaxer=relaxer,
        niter=niter,
        population_size=8,
        offspring_fraction=0.5,
        niter_local_relaxation=1,
        early_stopping_niter=0,
        relax_batch_target=relax_batch_target,
        rng=rng,
    )


def test_relax_batch_target_reduces_number_of_relax_calls(tmp_path, rng):
    """Accumulating offspring yields fewer, larger relax calls than one per gen."""
    legacy_relaxer = RecordingRelaxer(max_steps=1)
    _run_ga(
        tmp_path / "legacy",
        rng,
        legacy_relaxer,
        relax_batch_target=None,
    )

    batched_relaxer = RecordingRelaxer(max_steps=1)
    _run_ga(
        tmp_path / "batched",
        np.random.default_rng(1234),
        batched_relaxer,
        relax_batch_target="auto",
    )

    # First call in each run is the initial population; compare the generation calls.
    legacy_gen_calls = legacy_relaxer.batch_sizes[1:]
    batched_gen_calls = batched_relaxer.batch_sizes[1:]

    assert len(legacy_gen_calls) > len(batched_gen_calls)
    assert max(batched_gen_calls) > max(legacy_gen_calls)
    # Nothing is dropped: every offspring is relaxed exactly once.
    assert sum(batched_gen_calls) == sum(legacy_gen_calls)


def test_relax_batch_target_flushes_remaining_backlog(tmp_path, rng):
    """The final drain leaves no unrelaxed candidate behind."""
    from scgo.database import get_connection

    outdir = tmp_path / "drain"
    relaxer = RecordingRelaxer(max_steps=1)
    _run_ga(outdir, rng, relaxer, relax_batch_target=1000, niter=3)

    with get_connection(str(outdir / "ga_go.db")) as da:
        assert da.get_number_of_unrelaxed_candidates() == 0
