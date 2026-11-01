"""GPU smoke tests for UPET + TorchSim (Kaggle ``requires_upet`` suite)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.integration.gpu_smoke_common import (
    assert_cuda_autobatcher,
    assert_gpu_run_go,
    assert_relax_batch,
    build_cuda_relaxer,
)


@pytest.fixture(autouse=True)
def _require_upet_stack() -> None:
    pytest.importorskip("upet")
    pytest.importorskip("metatomic_torchsim")
    pytest.importorskip("torch_sim")


_UPET_RELAXER = {
    "model_kind": "upet",
    "upet_model_name": "pet-mad-s",
    "upet_version": "1.5.0",
}


@pytest.mark.gpu_smoke
@pytest.mark.requires_cuda
@pytest.mark.requires_upet
def test_upet_torchsim_batch_relaxer_uses_cuda_and_autobatcher():
    """UPET TorchSim relaxer must land on CUDA with InFlight autobatching."""
    relaxer = build_cuda_relaxer(
        **_UPET_RELAXER,
        force_tol=0.1,
        max_steps=5,
        expected_max_atoms=40,
        max_atoms_to_try=40,
    )
    assert_cuda_autobatcher(relaxer)


@pytest.mark.gpu_smoke
@pytest.mark.requires_cuda
@pytest.mark.requires_upet
def test_upet_torchsim_relax_batch_gpu():
    """Batched single-point evaluation on GPU for a tiny Pt cluster population."""
    relaxer = build_cuda_relaxer(
        **_UPET_RELAXER,
        force_tol=0.2,
        max_steps=20,
        expected_max_atoms=20,
        max_atoms_to_try=20,
    )
    assert_relax_batch(relaxer, steps=20, check_restored_cell=True)


@pytest.mark.gpu_smoke
@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.requires_cuda
@pytest.mark.requires_upet
def test_upet_run_go_gpu_smoke(tmp_path: Path):
    """Short UPET GO campaign on GPU (gas Pt4) for Kaggle CI."""
    from scgo.param_presets import get_default_upet_params

    assert_gpu_run_go(
        get_default_upet_params(),
        output_dir=tmp_path / "upet_go",
        model_kind="upet",
        require_autobatcher=True,
    )
