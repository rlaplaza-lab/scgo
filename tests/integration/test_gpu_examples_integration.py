"""GPU integration tests mirroring examples/ at minimal GA/TS scale."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from scgo import (
    get_torchsim_ga_params,
    get_ts_search_params,
    make_graphite_surface_config,
    run_go_ts,
)
from scgo.surface.config import SurfaceSystemConfig
from scgo.system_types import SystemType
from tests.test_utils import assert_supported_cluster_binding

SEED = 42
COMPOSITION = "Pt5"

MINIMAL_GA = {
    "niter": 2,
    "population_size": 4,
    "offspring_fraction": 0.5,
    "niter_local_relaxation": 50,
    "n_jobs_population_init": 1,
    "early_stopping_niter": 0,
    "write_timing_json": False,
    "detailed_timing": False,
}

MINIMAL_TS = {
    "max_pairs": 1,
    "neb_n_images": 3,
    "neb_steps": 50,
    "write_timing_json": False,
}

CONNECTIVITY = 1.8
SLAB_LAYERS = 2


def _adsorbates_oh(*, n: int = 1) -> list[Atoms]:
    out: list[Atoms] = []
    for i in range(n):
        shift = float(2.2 * i)
        out.append(
            Atoms(
                symbols=["O", "H"],
                positions=[[shift, 0.0, 0.0], [shift, 0.0, 0.96]],
            )
        )
    return out


@dataclass(frozen=True)
class GpuExampleCase:
    system_type: SystemType
    surface_config: SurfaceSystemConfig | None = None
    adsorbates: list[Atoms] | None = None
    connectivity_factor: float | None = None
    freeze_adsorbate_internal_geometry: bool = False
    extra_ts: dict = field(default_factory=dict)
    expected_mobile_atoms: int = 5
    adsorbate_fragment_lengths: list[int] | None = None


def _graphite_config() -> SurfaceSystemConfig:
    return make_graphite_surface_config(slab_layers=SLAB_LAYERS)


GPU_EXAMPLE_CASES = [
    GpuExampleCase(system_type="gas_cluster"),
    GpuExampleCase(
        system_type="surface_cluster",
        surface_config=_graphite_config(),
        connectivity_factor=CONNECTIVITY,
    ),
    GpuExampleCase(
        system_type="gas_cluster_adsorbate",
        adsorbates=_adsorbates_oh(n=1),
        connectivity_factor=CONNECTIVITY,
        freeze_adsorbate_internal_geometry=True,
        expected_mobile_atoms=7,
        adsorbate_fragment_lengths=[2],
    ),
    GpuExampleCase(
        system_type="surface_cluster_adsorbate",
        surface_config=_graphite_config(),
        adsorbates=_adsorbates_oh(n=2),
        connectivity_factor=CONNECTIVITY,
        freeze_adsorbate_internal_geometry=True,
        extra_ts={"energy_gap_threshold": 1.0},
        expected_mobile_atoms=9,
        adsorbate_fragment_lengths=[2, 2],
    ),
]


def _build_go_params(case: GpuExampleCase) -> dict:
    go_params = get_torchsim_ga_params(
        system_type=case.system_type,
        surface_config=case.surface_config,
        seed=SEED,
    )
    if case.connectivity_factor is not None:
        go_params["connectivity_factor"] = case.connectivity_factor
    go_params["optimizer_params"]["ga"].update(MINIMAL_GA)
    if case.freeze_adsorbate_internal_geometry:
        go_params["freeze_adsorbate_internal_geometry"] = True
    return go_params


def _build_ts_params(case: GpuExampleCase) -> dict:
    ts_params = get_ts_search_params(
        system_type=case.system_type,
        surface_config=case.surface_config,
        seed=SEED,
    )
    ts_params.update(MINIMAL_TS)
    if case.connectivity_factor is not None:
        ts_params["connectivity_factor"] = case.connectivity_factor
    ts_params.update(case.extra_ts)
    return ts_params


@pytest.mark.parametrize("case", GPU_EXAMPLE_CASES, ids=lambda c: c.system_type)
@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.requires_cuda
@pytest.mark.requires_mace
def test_run_go_ts_gpu_example_smoke(tmp_path: Path, case: GpuExampleCase) -> None:
    """End-to-end GO+TS with MACE/TorchSim for each example system type."""
    output_dir = tmp_path / f"gpu_{case.system_type}"
    summary = run_go_ts(
        COMPOSITION,
        go_params=_build_go_params(case),
        ts_params=_build_ts_params(case),
        seed=SEED,
        verbosity=0,
        output_dir=output_dir,
        system_type=case.system_type,
        surface_config=case.surface_config,
        adsorbates=case.adsorbates,
        log_summary=False,
    )

    assert isinstance(summary, dict)
    for key in (
        "minima_by_formula",
        "ts_results",
        "ts_total_count",
        "ts_success_count",
    ):
        assert key in summary

    minima = summary["minima_by_formula"]["Pt5"]
    assert len(minima) >= 1
    assert all(np.isfinite(energy) for energy, _atoms in minima)

    db_files = list(output_dir.glob("**/*.db"))
    assert db_files, "No database files found after run_go_ts"

    _energy, best = minima[0]
    n_slab = len(case.surface_config.slab) if case.surface_config is not None else 0
    assert len(best) == n_slab + case.expected_mobile_atoms

    if case.surface_config is not None:
        assert_supported_cluster_binding(
            best,
            case.surface_config,
            n_core_mobile=5,
            adsorbate_fragment_lengths=case.adsorbate_fragment_lengths,
        )

    assert summary["ts_total_count"] >= 0
    for result in summary["ts_results"]:
        assert isinstance(result, dict)
        assert "pair_id" in result
        assert "status" in result
        if result.get("status") == "success":
            barrier = result.get("barrier_height")
            assert barrier is not None
            assert np.isfinite(float(barrier))
