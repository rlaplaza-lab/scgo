"""GPU smoke tests for MACE + TorchSim (Kaggle ``gpu_smoke`` / MACE suite)."""

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
def _require_mace_stack() -> None:
    pytest.importorskip("mace")
    pytest.importorskip("torch_sim")


_MACE_RELAXER = {
    "model_kind": "mace",
    "mace_model_name": "mace_matpes_0",
}


@pytest.mark.gpu_smoke
@pytest.mark.requires_cuda
@pytest.mark.requires_mace
def test_mace_torchsim_batch_relaxer_uses_cuda_and_autobatcher():
    """MACE TorchSim relaxer must land on CUDA with InFlight autobatching."""
    relaxer = build_cuda_relaxer(
        **_MACE_RELAXER,
        force_tol=0.1,
        max_steps=5,
        expected_max_atoms=40,
        max_atoms_to_try=40,
    )
    assert_cuda_autobatcher(relaxer)


@pytest.mark.gpu_smoke
@pytest.mark.requires_cuda
@pytest.mark.requires_mace
def test_mace_torchsim_relax_batch_gpu():
    """Batched single-point evaluation on GPU for a tiny Pt cluster population."""
    relaxer = build_cuda_relaxer(
        **_MACE_RELAXER,
        force_tol=0.2,
        max_steps=20,
        expected_max_atoms=20,
        max_atoms_to_try=20,
    )
    assert_relax_batch(relaxer, steps=20)


@pytest.mark.gpu_smoke
@pytest.mark.requires_cuda
@pytest.mark.requires_mace
def test_mace_torchsim_relax_batch_fixatoms_gpu():
    """Constrained batched MACE GO on GPU (FixAtoms + autobatcher split/pop)."""
    relaxer = build_cuda_relaxer(
        **_MACE_RELAXER,
        force_tol=0.2,
        max_steps=10,
        expected_max_atoms=20,
        max_atoms_to_try=20,
    )
    assert_relax_batch(relaxer, steps=10, fix_first=True)


@pytest.mark.gpu_smoke
@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.requires_cuda
@pytest.mark.requires_mace
def test_mace_run_go_gpu_smoke(tmp_path: Path):
    """Short MACE GO campaign on GPU (gas Pt4) for Kaggle smoke CI."""
    from scgo.param_presets import get_low_effort_torchsim_ga_params

    params = get_low_effort_torchsim_ga_params(system_type="gas_cluster", seed=42)
    assert_gpu_run_go(
        params,
        output_dir=tmp_path / "mace_go",
        model_kind="mace",
    )
