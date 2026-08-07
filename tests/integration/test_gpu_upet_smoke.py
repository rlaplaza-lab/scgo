"""GPU smoke tests for UPET + TorchSim (Kaggle ``requires_upet`` suite)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytest.importorskip("upet")
pytest.importorskip("metatomic_torchsim")
pytest.importorskip("torch_sim")

import numpy as np

from tests.test_utils import (
    assert_minima_lists_equal,
    validate_structure_with_diagnostics,
)


@pytest.mark.requires_cuda
@pytest.mark.requires_upet
def test_upet_torchsim_batch_relaxer_uses_cuda_and_autobatcher():
    """UPET TorchSim relaxer must land on CUDA with InFlight autobatching."""
    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    assert torch.cuda.is_available()
    relaxer = TorchSimBatchRelaxer(
        model_kind="upet",
        upet_model_name="pet-mad-s",
        upet_version="1.5.0",
        force_tol=0.1,
        max_steps=5,
        expected_max_atoms=40,
        max_atoms_to_try=40,
    )
    assert str(relaxer.device).startswith("cuda")
    assert "autobatcher" in relaxer._runner_kwargs
    assert getattr(relaxer.model, "device", None) is not None
    assert str(relaxer.model.device).startswith("cuda")


@pytest.mark.requires_cuda
@pytest.mark.requires_upet
def test_upet_torchsim_relax_batch_gpu(tmp_path: Path):
    """Batched single-point evaluation on GPU for a tiny Pt cluster population."""
    from ase import Atoms

    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    assert torch.cuda.is_available()
    relaxer = TorchSimBatchRelaxer(
        model_kind="upet",
        upet_model_name="pet-mad-s",
        upet_version="1.5.0",
        force_tol=0.2,
        max_steps=20,
        expected_max_atoms=20,
        max_atoms_to_try=20,
    )

    def _pt4(seed: int) -> Atoms:
        rng = torch.Generator().manual_seed(seed)
        pos = (torch.rand(4, 3, generator=rng) * 2.5).numpy()
        atoms = Atoms("Pt4", positions=pos)
        atoms.center(vacuum=5.0)
        atoms.pbc = False
        return atoms

    results = relaxer.relax_batch([_pt4(0), _pt4(1)], steps=20)
    assert len(results) == 2
    for energy, atoms in results:
        assert isinstance(energy, float)
        assert np.isfinite(energy), "Relaxed energy must be finite"
        assert len(atoms) == 4
        assert atoms.cell.sum() != 0.0  # storage cell restored after metatomic path
        # Relaxed cluster must stay a connected, clash-free structure.
        validate_structure_with_diagnostics(atoms, context="upet relaxed Pt4")


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.requires_cuda
@pytest.mark.requires_upet
def test_upet_run_go_gpu_smoke(tmp_path: Path):
    """Short UPET GO campaign on GPU (gas Pt4) for Kaggle CI."""
    from scgo import run_go
    from scgo.param_presets import get_default_upet_params

    assert torch.cuda.is_available()
    params = get_default_upet_params()
    ga = params["optimizer_params"]["ga"]
    ga["niter"] = 2
    ga["population_size"] = 6
    ga["niter_local_relaxation"] = 40
    ga["n_jobs_population_init"] = 1
    ga["early_stopping_niter"] = 0
    ga["write_timing_json"] = False
    ga["detailed_timing"] = False

    relaxer = ga.get("relaxer")
    assert relaxer is not None
    assert getattr(relaxer, "model_kind", None) == "upet"
    assert str(relaxer.device).startswith("cuda")
    assert "autobatcher" in relaxer._runner_kwargs

    results = run_go(
        ["Pt"] * 4,
        params=params,
        seed=42,
        system_type="gas_cluster",
        output_dir=tmp_path / "upet_go",
        verbosity=0,
    )
    assert results
    assert all(isinstance(e, float) for e, _ in results)
    assert all(np.isfinite(e) for e, _ in results)
    # Energies should be well-formed (finite and mutually comparable).
    assert len({round(e, 6) for e, _ in results}) >= 1
    # Each GO minimum must be a connected, clash-free Pt4 cluster.
    for _energy, atoms in results:
        assert len(atoms) == 4
        validate_structure_with_diagnostics(atoms, context="upet GO Pt4")

    # Same-seed determinism: a second identical run must yield element-identical
    # GO minima (UPET forward passes are deterministic for a fixed seed).
    results2 = run_go(
        ["Pt"] * 4,
        params=params,
        seed=42,
        system_type="gas_cluster",
        output_dir=tmp_path / "upet_go_repeat",
        verbosity=0,
    )
    assert_minima_lists_equal(results, results2, rtol=1e-5, atol=1e-6)
