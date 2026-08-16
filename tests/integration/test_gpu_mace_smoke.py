"""GPU smoke tests for MACE + TorchSim (Kaggle ``gpu_smoke`` / MACE suite)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tests.helpers import assert_e2e_minima_list

pytest.importorskip("mace")
pytest.importorskip("torch_sim")


@pytest.mark.gpu_smoke
@pytest.mark.requires_cuda
@pytest.mark.requires_mace
def test_mace_torchsim_batch_relaxer_uses_cuda_and_autobatcher():
    """MACE TorchSim relaxer must land on CUDA with InFlight autobatching."""
    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    assert torch.cuda.is_available()
    relaxer = TorchSimBatchRelaxer(
        model_kind="mace",
        mace_model_name="mace_matpes_0",
        force_tol=0.1,
        max_steps=5,
        expected_max_atoms=40,
        max_atoms_to_try=40,
    )
    assert str(relaxer.device).startswith("cuda")
    assert "autobatcher" in relaxer._runner_kwargs
    assert getattr(relaxer.model, "device", None) is not None
    assert str(relaxer.model.device).startswith("cuda")


@pytest.mark.gpu_smoke
@pytest.mark.requires_cuda
@pytest.mark.requires_mace
def test_mace_torchsim_relax_batch_gpu():
    """Batched single-point evaluation on GPU for a tiny Pt cluster population."""
    from ase import Atoms

    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    assert torch.cuda.is_available()
    relaxer = TorchSimBatchRelaxer(
        model_kind="mace",
        mace_model_name="mace_matpes_0",
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
        assert len(atoms) == 4


@pytest.mark.gpu_smoke
@pytest.mark.requires_cuda
@pytest.mark.requires_mace
def test_mace_torchsim_relax_batch_fixatoms_gpu():
    """Constrained batched MACE GO on GPU (FixAtoms + autobatcher split/pop)."""
    from ase import Atoms
    from ase.constraints import FixAtoms

    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    assert torch.cuda.is_available()
    relaxer = TorchSimBatchRelaxer(
        model_kind="mace",
        mace_model_name="mace_matpes_0",
        force_tol=0.2,
        max_steps=10,
        expected_max_atoms=20,
        max_atoms_to_try=20,
    )

    def _pt4(seed: int) -> Atoms:
        rng = torch.Generator().manual_seed(seed)
        pos = (torch.rand(4, 3, generator=rng) * 2.5).numpy()
        atoms = Atoms("Pt4", positions=pos)
        atoms.center(vacuum=5.0)
        atoms.pbc = False
        atoms.set_constraint(FixAtoms(indices=[0]))
        return atoms

    results = relaxer.relax_batch([_pt4(0), _pt4(1)], steps=10)
    assert len(results) == 2
    for energy, atoms in results:
        assert isinstance(energy, float)
        assert len(atoms) == 4


@pytest.mark.gpu_smoke
@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.requires_cuda
@pytest.mark.requires_mace
def test_mace_run_go_gpu_smoke(tmp_path: Path):
    """Short MACE GO campaign on GPU (gas Pt4) for Kaggle smoke CI."""
    from scgo import run_go
    from scgo.param_presets import get_low_effort_torchsim_ga_params

    assert torch.cuda.is_available()
    params = get_low_effort_torchsim_ga_params(system_type="gas_cluster", seed=42)
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
    assert getattr(relaxer, "model_kind", None) == "mace"
    assert str(relaxer.device).startswith("cuda")

    results = run_go(
        ["Pt"] * 4,
        params=params,
        seed=42,
        system_type="gas_cluster",
        output_dir=tmp_path / "mace_go",
        verbosity=0,
    )
    assert_e2e_minima_list(
        results,
        expected_n_atoms=4,
        output_dir=tmp_path / "mace_go",
    )
