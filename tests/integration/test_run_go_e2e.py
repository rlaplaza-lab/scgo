"""Unmocked end-to-end tests for public ``run_go`` and ``run_go_ts`` APIs."""

from __future__ import annotations

import numpy as np
import pytest

from scgo.param_presets import get_testing_params, get_ts_search_params
from scgo.runner_api import run_go, run_go_ts
from tests.test_utils import assert_db_final_row


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
def test_run_go_ts_pt4_produces_valid_barrier(tmp_path) -> None:
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
    if summary.get("ts_total_count", 0) == 0:
        pytest.skip("TS search found no pairs (insufficient distinct minima)")
    ts_results = summary.get("ts_results") or []
    valid = [
        r
        for r in ts_results
        if isinstance(r, dict) and r.get("barrier_height") is not None
    ]
    assert valid, "No TS dicts with barrier_height"
    barrier = float(valid[0]["barrier_height"])
    assert 0.0 <= barrier <= 10.0
    assert valid[0].get("ts_energy") is not None


@pytest.mark.slow
@pytest.mark.integration
def test_run_go_surface_adsorbate_height_bounds(tmp_path, surface_config_pt111) -> None:
    from ase import Atoms

    params = get_testing_params()
    params["optimizer_params"]["ga"].update(
        {
            "niter": 1,
            "population_size": 3,
            "niter_local_relaxation": 30,
            "surface_config": surface_config_pt111,
        }
    )
    slab = surface_config_pt111.slab
    n_slab = len(slab)
    oh = Atoms("OH", positions=[[0, 0, 0], [0, 0, 0.96]])

    minima = run_go(
        ["Pt", "Pt", "Pt"],
        params=params,
        seed=7,
        verbosity=0,
        output_dir=str(tmp_path / "surf_go"),
        system_type="surface_cluster_adsorbate",
        surface_config=surface_config_pt111,
        adsorbates=[oh],
    )
    if not minima:
        pytest.skip("No minima returned from surface adsorbate GO")
    _energy, best = minima[0]
    assert len(best) > n_slab
    symbols = best.get_chemical_symbols()
    assert "O" in symbols[n_slab:]
    assert "H" in symbols[n_slab:]
