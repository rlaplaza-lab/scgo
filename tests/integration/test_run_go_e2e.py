"""Unmocked end-to-end tests for public ``run_go`` and ``run_go_ts`` APIs."""

from __future__ import annotations

import numpy as np
import pytest

from scgo.param_presets import get_testing_params, get_ts_search_params
from scgo.runner_api import run_go, run_go_ts
from tests.constants import PT4_EMT_BARRIER_EV
from tests.test_utils import (
    assert_db_final_row,
    assert_supported_cluster_binding,
    assert_ts_result_valid,
)


def _emt_ts_params(**overrides) -> dict:
    params = {
        **get_ts_search_params(
            system_type="gas_cluster",
            calculator="EMT",
            calculator_kwargs={},
        ),
        "use_torchsim": False,
        "use_parallel_neb": False,
        "max_pairs": 1,
        "n_images": 7,
        "neb_steps": 150,
        "climb": True,
    }
    params.update(overrides)
    return params


@pytest.mark.slow
@pytest.mark.integration
def test_run_go_pt2_produces_tagged_minima(tmp_path) -> None:
    params = get_testing_params()
    params["tag_final_minima"] = True
    output_dir = tmp_path / "pt2_go"
    minima = run_go(
        ["Pt", "Pt"],
        params=params,
        seed=42,
        verbosity=0,
        output_dir=str(output_dir),
        system_type="gas_cluster",
    )
    assert len(minima) >= 1
    for energy, _atoms in minima:
        assert np.isfinite(energy)

    db_files = list(output_dir.glob("**/*.db"))
    assert db_files, "No database files found after run_go"
    assert_db_final_row(str(db_files[0]), None, expect_final_id=True)

    xyz_dir = output_dir / "final_unique_minima"
    xyz_files = list(xyz_dir.glob("*.xyz"))
    assert xyz_files, "No XYZ minima exported"


@pytest.mark.slow
@pytest.mark.integration
def test_run_go_ts_pt4_finds_ts_candidate_pairs(tmp_path) -> None:
    """End-to-end: Pt4 GO+TS workflow discovers at least one candidate pair."""
    go_params = get_testing_params()
    go_params["optimizer_params"]["ga"].update(
        {
            "niter": 3,
            "population_size": 8,
            "niter_local_relaxation": 5,
        }
    )
    ts_params = _emt_ts_params(neb_steps=200, max_pairs=1)

    summary = run_go_ts(
        ["Pt", "Pt", "Pt", "Pt"],
        go_params=go_params,
        ts_params=ts_params,
        seed=42,
        verbosity=0,
        output_dir=str(tmp_path / "pt4_go_ts"),
        system_type="gas_cluster",
    )
    assert isinstance(summary, dict)
    assert summary.get("ts_total_count", 0) >= 1
    ts_results = summary.get("ts_results") or []
    assert ts_results, "Expected TS result dicts"
    for result in ts_results:
        assert isinstance(result, dict)
        assert "pair_id" in result
        assert "status" in result
        if result.get("status") == "success":
            assert_ts_result_valid(
                result,
                barrier_range=PT4_EMT_BARRIER_EV,
                require_interior_ts=True,
            )


@pytest.mark.slow
@pytest.mark.integration
def test_run_go_ts_h2_has_no_ts_pairs(tmp_path) -> None:
    go_params = get_testing_params()
    go_params["optimizer_params"]["simple"].update(
        {
            "niter": 2,
            "niter_local_relaxation": 8,
        }
    )
    ts_params = _emt_ts_params(neb_steps=200, max_pairs=3)

    summary = run_go_ts(
        ["H", "H"],
        go_params=go_params,
        ts_params=ts_params,
        seed=42,
        verbosity=0,
        output_dir=str(tmp_path / "h2_go_ts"),
        system_type="gas_cluster",
    )
    assert isinstance(summary, dict)
    assert summary.get("ts_total_count", -1) == 0
    assert summary.get("ts_success_count", -1) == 0
    assert summary.get("ts_results") == []


@pytest.mark.slow
@pytest.mark.integration
def test_run_go_surface_cluster_remains_chemisorbed(
    tmp_path, surface_config_pt111
) -> None:
    """Public run_go on a supported Pt cluster should stay bound to the slab."""
    params = get_testing_params()
    params["optimizer_params"]["ga"].update(
        {
            "niter": 1,
            "population_size": 4,
            "niter_local_relaxation": 30,
            "batch_size": 2,
            "surface_config": surface_config_pt111,
        }
    )
    slab = surface_config_pt111.slab
    n_slab = len(slab)

    minima = run_go(
        ["Pt", "Pt", "Pt", "Pt"],
        params=params,
        seed=42,
        verbosity=0,
        output_dir=str(tmp_path / "surf_go"),
        system_type="surface_cluster",
        surface_config=surface_config_pt111,
    )
    assert minima, "run_go returned no minima for surface_cluster"
    _energy, best = minima[0]
    assert len(best) == n_slab + 4
    assert_supported_cluster_binding(best, surface_config_pt111)
