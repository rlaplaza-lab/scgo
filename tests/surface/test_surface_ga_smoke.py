"""End-to-end smoke tests for adsorbate-on-slab GA (`ga_go` + `SurfaceSystemConfig`)."""

from __future__ import annotations

import pytest
from ase.calculators.emt import EMT
from numpy.random import default_rng

from scgo.algorithms import ga_go
from tests.test_utils import (
    MockRelaxer,
    assert_deposition_height_in_bounds,
    assert_supported_cluster_binding,
)


def _assert_surface_ga_result(
    minima,
    slab,
    surface_config,
    n_adsorbate: int,
    *,
    post_relaxation: bool,
) -> None:
    assert len(minima) >= 1
    _e, best = minima[0]
    n_slab = len(slab)
    assert len(best) == n_slab + n_adsorbate
    if post_relaxation:
        assert_supported_cluster_binding(best, surface_config)
    else:
        assert_deposition_height_in_bounds(
            best,
            slab,
            surface_config.adsorption_height_min,
            surface_config.adsorption_height_max,
            n_slab=n_slab,
            axis=surface_config.surface_normal_axis,
        )


@pytest.mark.slow
def test_ga_go_surface_config_smoke_emt(
    surface_config_pt111,
    minimal_ga_kwargs,
    tmp_path,
):
    """Minimal GA on a slab with real EMT relaxation."""
    slab = surface_config_pt111.slab
    rng = default_rng(42)
    out = tmp_path / "surface_ga_smoke_emt"
    out.mkdir(parents=True, exist_ok=True)

    minima = ga_go(
        composition=["Pt", "Pt"],
        output_dir=str(out),
        calculator=EMT(),
        verbosity=0,
        rng=rng,
        system_type="surface_cluster",
        surface_config=surface_config_pt111,
        **minimal_ga_kwargs,
    )
    _assert_surface_ga_result(
        minima,
        slab,
        surface_config_pt111,
        n_adsorbate=2,
        post_relaxation=True,
    )


def test_ga_go_surface_config_smoke_mock_relaxer(
    surface_config_pt111,
    minimal_ga_kwargs,
    tmp_path,
):
    """Minimal GA on a slab with MockRelaxer (placement geometry preserved)."""
    slab = surface_config_pt111.slab
    rng = default_rng(42)
    out = tmp_path / "surface_ga_smoke_mock"
    out.mkdir(parents=True, exist_ok=True)

    minima = ga_go(
        composition=["Pt", "Pt"],
        output_dir=str(out),
        calculator=EMT(),
        relaxer=MockRelaxer(max_steps=1),
        verbosity=0,
        rng=rng,
        system_type="surface_cluster",
        surface_config=surface_config_pt111,
        niter=1,
        population_size=3,
        niter_local_relaxation=20,
        batch_size=2,
        offspring_fraction=minimal_ga_kwargs["offspring_fraction"],
        early_stopping_niter=minimal_ga_kwargs.get("early_stopping_niter", 0),
        n_jobs_population_init=minimal_ga_kwargs.get("n_jobs_population_init", 1),
    )
    _assert_surface_ga_result(
        minima,
        slab,
        surface_config_pt111,
        n_adsorbate=2,
        post_relaxation=False,
    )
