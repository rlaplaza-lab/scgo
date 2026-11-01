"""Shared bodies for the MACE and UPET GPU smoke modules.

Each suite keeps its own test module so ``requires_mace`` / ``requires_upet``
stay file-local. The sequences below are the parts that do not differ.
"""

from __future__ import annotations

from pathlib import Path

import torch
from ase import Atoms
from ase.constraints import FixAtoms

from scgo import run_go
from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer
from tests.helpers import assert_e2e_minima_list


def build_cuda_relaxer(**relaxer_kwargs) -> TorchSimBatchRelaxer:
    assert torch.cuda.is_available()
    return TorchSimBatchRelaxer(**relaxer_kwargs)


def assert_cuda_autobatcher(relaxer: TorchSimBatchRelaxer) -> None:
    """Relaxer landed on CUDA with InFlight autobatching."""
    assert str(relaxer.device).startswith("cuda")
    assert "autobatcher" in relaxer._runner_kwargs
    assert getattr(relaxer.model, "device", None) is not None
    assert str(relaxer.model.device).startswith("cuda")


def pt4_cluster(seed: int, *, fix_first: bool = False) -> Atoms:
    rng = torch.Generator().manual_seed(seed)
    pos = (torch.rand(4, 3, generator=rng) * 2.5).numpy()
    atoms = Atoms("Pt4", positions=pos)
    atoms.center(vacuum=5.0)
    atoms.pbc = False
    if fix_first:
        atoms.set_constraint(FixAtoms(indices=[0]))
    return atoms


def assert_relax_batch(
    relaxer: TorchSimBatchRelaxer,
    *,
    steps: int,
    fix_first: bool = False,
    check_restored_cell: bool = False,
) -> None:
    results = relaxer.relax_batch(
        [pt4_cluster(0, fix_first=fix_first), pt4_cluster(1, fix_first=fix_first)],
        steps=steps,
    )
    assert len(results) == 2
    for energy, atoms in results:
        assert isinstance(energy, float)
        assert len(atoms) == 4
        if check_restored_cell:
            assert atoms.cell.sum() != 0.0


def apply_smoke_ga_budget(ga: dict) -> None:
    ga["niter"] = 2
    ga["population_size"] = 6
    ga["niter_local_relaxation"] = 40
    ga["n_jobs_population_init"] = 1
    ga["early_stopping_niter"] = 0
    ga["write_timing_json"] = False
    ga["detailed_timing"] = False


def assert_gpu_run_go(
    params: dict,
    *,
    output_dir: Path,
    model_kind: str,
    require_autobatcher: bool = False,
) -> None:
    assert torch.cuda.is_available()
    ga = params["optimizer_params"]["ga"]
    apply_smoke_ga_budget(ga)

    relaxer = ga.get("relaxer")
    assert relaxer is not None
    assert getattr(relaxer, "model_kind", None) == model_kind
    assert str(relaxer.device).startswith("cuda")
    if require_autobatcher:
        assert "autobatcher" in relaxer._runner_kwargs

    results = run_go(
        ["Pt"] * 4,
        params=params,
        seed=42,
        system_type="gas_cluster",
        output_dir=output_dir,
        verbosity=0,
    )
    assert_e2e_minima_list(
        results,
        expected_n_atoms=4,
        output_dir=output_dir,
    )
