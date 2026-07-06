"""End-to-end smoke tests for adsorbate-on-slab GA (`ga_go` + `SurfaceSystemConfig`)."""

from __future__ import annotations

import numpy as np
import pytest
from ase.calculators.emt import EMT
from numpy.random import default_rng

from scgo.algorithms import ga_go
from scgo.surface.deposition import slab_surface_extreme
from tests.test_utils import MockRelaxer


def _assert_surface_ga_result(minima, slab, n_adsorbate: int) -> None:
    assert len(minima) >= 1
    _e, best = minima[0]
    n_slab = len(slab)
    assert len(best) == n_slab + n_adsorbate
    z_top = slab_surface_extreme(slab, 2, upper=True)
    ads_z = best.get_positions()[n_slab:, 2]
    assert np.min(ads_z) > z_top - 0.2


@pytest.mark.parametrize(
    ("relaxer_factory", "ga_overrides"),
    [
        pytest.param(None, {}, id="emt"),
        pytest.param(
            lambda: MockRelaxer(max_steps=1),
            {
                "niter": 1,
                "population_size": 3,
                "niter_local_relaxation": 20,
                "batch_size": 2,
            },
            id="mock_relaxer",
        ),
    ],
)
def test_ga_go_surface_config_smoke(
    surface_config_pt111,
    minimal_ga_kwargs,
    tmp_path,
    relaxer_factory,
    ga_overrides,
):
    """Minimal GA on a slab via direct ``ga_go`` (EMT or MockRelaxer)."""
    slab = surface_config_pt111.slab
    rng = default_rng(42)
    out = tmp_path / "surface_ga_smoke"
    out.mkdir(parents=True, exist_ok=True)

    kwargs = {**minimal_ga_kwargs, **ga_overrides}
    relaxer = relaxer_factory() if relaxer_factory is not None else None

    minima = ga_go(
        composition=["Pt", "Pt"],
        output_dir=str(out),
        calculator=EMT(),
        relaxer=relaxer,
        verbosity=0,
        rng=rng,
        system_type="surface_cluster",
        surface_config=surface_config_pt111,
        **kwargs,
    )

    _assert_surface_ga_result(minima, slab, n_adsorbate=2)
