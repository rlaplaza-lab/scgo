"""GPU integration tests mirroring examples/ at reduced GA/TS scale for Kaggle CI.

Per-case knobs are derived from the matching ``examples/example_pt5_*.py`` scripts
(~20–25% of their niter / population_size / max_pairs, with heavier surface NEB
budgets preserved). Targets ~30 min for the full ``requires_cuda and requires_mace``
Kaggle suite (vs ~10 min with the prior minimal settings).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from scgo import (
    get_cluster_formula,
    get_torchsim_ga_params,
    get_ts_search_params,
    make_graphite_surface_config,
    parse_composition_arg,
    run_go_ts,
)
from scgo.surface.config import SurfaceSystemConfig
from scgo.system_types import SystemType, build_adsorbate_definition_from_inputs
from tests.test_utils import assert_supported_cluster_binding

SEED = 42
COMPOSITION = "Pt5"

# Shared GA/TS base; per-case overrides mirror example_pt5_*.py ratios.
CI_EXAMPLE_GA_BASE = {
    "offspring_fraction": 0.5,
    "niter_local_relaxation": 70,
    "n_jobs_population_init": 1,
    "early_stopping_niter": 0,
    "write_timing_json": False,
    "detailed_timing": False,
}

CI_EXAMPLE_TS_BASE = {
    "neb_n_images": 5,
    "write_timing_json": False,
}

CONNECTIVITY = 1.8
SLAB_LAYERS = 3


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
    ga_overrides: dict = field(default_factory=dict)
    ts_overrides: dict = field(default_factory=dict)
    extra_ts: dict = field(default_factory=dict)
    expected_mobile_atoms: int = 5
    adsorbate_fragment_lengths: list[int] | None = None


def _graphite_config() -> SurfaceSystemConfig:
    return make_graphite_surface_config(slab_layers=SLAB_LAYERS)


GPU_EXAMPLE_CASES = [
    # example_pt5_gas.py: NITER=10, POPULATION_SIZE=50, MAX_PAIRS=15
    GpuExampleCase(
        system_type="gas_cluster",
        ga_overrides={"niter": 3, "population_size": 7},
        ts_overrides={"max_pairs": 2, "neb_steps": 70},
    ),
    # example_pt5_graphite.py: NITER=6, POPULATION_SIZE=24, MAX_PAIRS=10
    GpuExampleCase(
        system_type="surface_cluster",
        surface_config=_graphite_config(),
        connectivity_factor=CONNECTIVITY,
        ga_overrides={"niter": 2, "population_size": 6},
        ts_overrides={"max_pairs": 2, "neb_steps": 90},
    ),
    # example_pt5_oh_gas.py: NITER=8, POPULATION_SIZE=40, MAX_PAIRS=12
    GpuExampleCase(
        system_type="gas_cluster_adsorbate",
        adsorbates=_adsorbates_oh(n=1),
        connectivity_factor=CONNECTIVITY,
        freeze_adsorbate_internal_geometry=True,
        ga_overrides={"niter": 3, "population_size": 6},
        ts_overrides={"max_pairs": 2, "neb_steps": 70},
        expected_mobile_atoms=7,
        adsorbate_fragment_lengths=[2],
    ),
    # example_pt5_2oh_graphite.py: NITER=6, POP=24, MAX_PAIRS=10, neb 7/800
    GpuExampleCase(
        system_type="surface_cluster_adsorbate",
        surface_config=_graphite_config(),
        adsorbates=_adsorbates_oh(n=2),
        connectivity_factor=CONNECTIVITY,
        freeze_adsorbate_internal_geometry=True,
        ga_overrides={"niter": 2, "population_size": 6},
        ts_overrides={"max_pairs": 2, "neb_n_images": 5, "neb_steps": 120},
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
    ga_params = dict(CI_EXAMPLE_GA_BASE)
    ga_params.update(case.ga_overrides)
    go_params["optimizer_params"]["ga"].update(ga_params)
    if case.freeze_adsorbate_internal_geometry:
        go_params["freeze_adsorbate_internal_geometry"] = True
    return go_params


def _expected_formula(case: GpuExampleCase) -> str:
    """Match run_go_ts: core composition plus adsorbate symbols when present."""
    core = parse_composition_arg(COMPOSITION)
    if case.adsorbates is None:
        return get_cluster_formula(core)
    _ads_def, _fragments, full_mobile = build_adsorbate_definition_from_inputs(
        system_type=case.system_type,
        composition=core,
        adsorbates=case.adsorbates,
        context="test_run_go_ts_gpu_example_smoke",
    )
    return get_cluster_formula(full_mobile)


def _build_ts_params(case: GpuExampleCase) -> dict:
    ts_params = get_ts_search_params(
        system_type=case.system_type,
        surface_config=case.surface_config,
        seed=SEED,
    )
    ts_params.update(CI_EXAMPLE_TS_BASE)
    ts_params.update(case.ts_overrides)
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
    go_params = _build_go_params(case)
    summary = run_go_ts(
        COMPOSITION,
        go_params=go_params,
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
        "formula",
        "minima_by_formula",
        "ts_results",
        "ts_total_count",
        "ts_success_count",
    ):
        assert key in summary

    expected_formula = _expected_formula(case)
    assert summary["formula"] == expected_formula
    minima = summary["minima_by_formula"][expected_formula]
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
            connectivity_factor=go_params["connectivity_factor"],
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
