"""`_run_go_trials` with adsorbate-on-slab GA via `optimizer_params['ga']`.

`_run_go_trials` selects the algorithm from len(composition) only. For slab GA,
``surface_config`` must live under ``optimizer_params['ga']``, which is only read
when the chosen algorithm is ``ga`` — so use **at least four** adsorbate atoms.
"""

from __future__ import annotations

from scgo.param_presets import get_testing_params
from scgo.runner_api import _run_go_trials
from scgo.utils.helpers import deep_merge_dicts
from tests.test_utils import assert_adsorption_height_in_bounds


def test__run_go_trials_passes_surface_config_when_ga_selected(
    surface_config_pt111, minimal_ga_kwargs, tmp_path
):
    base = get_testing_params()
    params = deep_merge_dicts(
        base,
        {
            "optimizer_params": {
                "ga": {
                    **minimal_ga_kwargs,
                    "niter_local_relaxation": 400,
                    "surface_config": surface_config_pt111,
                }
            }
        },
    )

    # Four adsorbate Pt atoms => chosen_go == "ga" (see select_scgo_minima_algorithm).
    composition = ["Pt", "Pt", "Pt", "Pt"]

    minima = _run_go_trials(
        composition=composition,
        system_type="surface_cluster",
        params=params,
        seed=42,
        verbosity=0,
        output_dir=str(tmp_path / "surf_go"),
    )

    slab = surface_config_pt111.slab
    assert len(minima) >= 1
    _e, best = minima[0]
    n_slab = len(slab)
    assert len(best) == n_slab + 4
    assert_adsorption_height_in_bounds(
        best,
        slab,
        surface_config_pt111.adsorption_height_min,
        surface_config_pt111.adsorption_height_max,
        n_slab=n_slab,
        axis=surface_config_pt111.surface_normal_axis,
    )
