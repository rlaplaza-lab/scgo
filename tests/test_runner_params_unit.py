"""Unit tests for ``scgo.runner_params`` prepare/merge helpers.

No GO/TS pipeline. Complements ``tests/integration/test_run_api.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from ase import Atoms
from ase.build import fcc111

from scgo.exceptions import SCGOValidationError
from scgo.param_presets import get_testing_params, get_ts_search_params
from scgo.runner_params import (
    _as_int_seed,
    _copy_params,
    _optimizer_write_timing_json_enabled,
    _prepare_run_go_context,
    _prepare_run_go_ts_context,
    _prepare_run_ts_search_context,
    _resolve_go_params,
    _with_surface_on_params,
    format_completion_details,
    resolve_workflow_seed,
)
from scgo.surface.config import SurfaceSystemConfig
from scgo.system_types import AdsorbateDefinition


def _pt111() -> SurfaceSystemConfig:
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    slab.pbc = True
    return SurfaceSystemConfig(slab=slab, fix_all_slab_atoms=True)


def _emt_ts_params(*, system_type: str, surface_config=None) -> dict:
    return {
        **get_ts_search_params(
            system_type=system_type,
            surface_config=surface_config,
            calculator="EMT",
            calculator_kwargs={},
        ),
        "use_parallel_neb": False,
        "use_torchsim": False,
    }


def test_copy_params_isolates_optimizer_slots() -> None:
    class _NonCopyable:
        def __deepcopy__(self, memo):
            raise RuntimeError("deepcopy should not clone the relaxer")

    relaxer = _NonCopyable()
    src = {
        "calculator": "EMT",
        "optimizer_params": {
            "ga": {"niter": 2, "relaxer": relaxer},
            "bh": {"niter": 5},
        },
    }
    copied = _copy_params(src)
    assert copied is not src
    assert copied["optimizer_params"] is not src["optimizer_params"]
    assert copied["optimizer_params"]["ga"] is not src["optimizer_params"]["ga"]
    assert copied["optimizer_params"]["ga"]["relaxer"] is relaxer
    copied["optimizer_params"]["ga"]["niter"] = 99
    assert src["optimizer_params"]["ga"]["niter"] == 2
    assert _copy_params(None) == {}

    # surface_config inject must share the relaxer (no deepcopy).
    cfg = _pt111()
    merged = _resolve_go_params(
        {
            "calculator": "EMT",
            "calculator_kwargs": {},
            "optimizer_params": {"ga": {"relaxer": relaxer}},
        },
        surface_config=cfg,
    )
    assert merged["surface_config"] is cfg
    assert merged["optimizer_params"]["ga"]["relaxer"] is relaxer


def test_as_int_seed_and_workflow_seed() -> None:
    assert _as_int_seed("seed", 7) == 7
    assert _as_int_seed("seed", "7") == 7
    assert _as_int_seed("seed", 1.5) == 1
    with pytest.raises(SCGOValidationError, match="must be int-like"):
        _as_int_seed("seed", "not_a_seed")
    with pytest.raises(SCGOValidationError, match="must be int-like"):
        _as_int_seed("seed", {})
    assert resolve_workflow_seed(seed_kw="7") == 7  # type: ignore[arg-type]
    assert resolve_workflow_seed(go_params={"seed": "3"}) == 3


def test_format_completion_details_order_and_omission() -> None:
    assert format_completion_details() == ""
    assert (
        format_completion_details(compositions=2, minima=5, output_dir="/tmp/out")
        == "compositions=2 minima=5 output_dir=/tmp/out"
    )
    assert (
        format_completion_details(successful_nebs=(3, 8), minima=1)
        == "minima=1 successful_nebs=3/8"
    )


def test_optimizer_write_timing_json_enabled() -> None:
    assert not _optimizer_write_timing_json_enabled({})
    assert not _optimizer_write_timing_json_enabled(
        {"optimizer_params": {"ga": {"niter": 1}}}
    )
    assert _optimizer_write_timing_json_enabled(
        {"optimizer_params": {"ga": {"write_timing_json": True}}}
    )
    assert _optimizer_write_timing_json_enabled(
        {"optimizer_params": {"bh": {"write_timing_json": True}}}
    )


def test_with_surface_on_params() -> None:
    cfg = _pt111()
    out = _with_surface_on_params({"optimizer_params": {"ga": {}}}, surface_config=cfg)
    assert out["surface_config"] is cfg
    other = SurfaceSystemConfig(
        slab=cfg.slab.copy(), fix_all_slab_atoms=True, name="other_surface"
    )
    with pytest.raises(SCGOValidationError, match="must match"):
        _with_surface_on_params(
            {"surface_config": cfg, "optimizer_params": {"ga": {}}},
            surface_config=other,
        )


def test_prepare_run_go_context(tmp_path: Path) -> None:
    params = get_testing_params()
    gas = _prepare_run_go_context(
        ["Pt", "Pt", "Pt"],
        params=params,
        seed=42,
        verbosity=0,
        run_id=None,
        clean=False,
        output_dir=tmp_path,
        calculator_for_global_optimization=None,
        surface_config=None,
        system_type="gas_cluster",
        adsorbates=None,
    )
    assert gas.composition == ["Pt", "Pt", "Pt"]
    assert gas.system_type == "gas_cluster"
    assert gas.seed == 42
    assert gas.output_dir == tmp_path.resolve()
    assert gas.output_summary_dir == str(tmp_path.resolve())

    oh = Atoms("OH", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.96]])
    ads_ctx = _prepare_run_go_context(
        ["Pt", "Pt"],
        params=get_testing_params(),
        seed=1,
        verbosity=0,
        run_id=None,
        clean=False,
        output_dir=tmp_path,
        calculator_for_global_optimization=None,
        surface_config=None,
        system_type="gas_cluster_adsorbate",
        adsorbates=oh,
    )
    assert ads_ctx.composition == ["Pt", "Pt", "O", "H"]
    ads = ads_ctx.params.get("adsorbate_definition")
    assert isinstance(ads, AdsorbateDefinition)
    assert ads.core_symbols == ["Pt", "Pt"]
    assert ads.adsorbate_symbols == ["O", "H"]

    # Top-level params['surface_config'] is enough when the run arg is omitted.
    cfg = _pt111()
    surface_params = get_testing_params()
    surface_params["surface_config"] = cfg
    surface_ctx = _prepare_run_go_context(
        ["Pt", "Pt"],
        params=surface_params,
        seed=1,
        verbosity=0,
        run_id=None,
        clean=False,
        output_dir=tmp_path,
        calculator_for_global_optimization=None,
        surface_config=None,
        system_type="surface_cluster",
        adsorbates=None,
    )
    assert surface_ctx.params.get("surface_config") is cfg


def test_prepare_run_go_ts_and_ts_search_context(tmp_path: Path) -> None:
    cfg = _pt111()
    go_ts = _prepare_run_go_ts_context(
        ["Pt", "Pt"],
        go_params=get_testing_params(),
        ts_params=_emt_ts_params(system_type="surface_cluster", surface_config=cfg),
        seed=7,
        verbosity=0,
        output_dir=tmp_path,
        output_root=None,
        output_stem=None,
        surface_config=cfg,
        system_type="surface_cluster",
        adsorbates=None,
    )
    assert go_ts.system_type == "surface_cluster"
    assert go_ts.composition == ["Pt", "Pt"]
    assert go_ts.seed == 7
    assert go_ts.go_params.get("surface_config") is cfg
    assert go_ts.output_dir == tmp_path.resolve()

    go_only = get_testing_params()
    go_only["surface_config"] = cfg
    go_ts_from_params = _prepare_run_go_ts_context(
        ["Pt", "Pt"],
        go_params=go_only,
        ts_params=_emt_ts_params(system_type="surface_cluster", surface_config=cfg),
        seed=7,
        verbosity=0,
        output_dir=tmp_path,
        output_root=None,
        output_stem=None,
        surface_config=None,
        system_type="surface_cluster",
        adsorbates=None,
    )
    assert go_ts_from_params.go_params.get("surface_config") is cfg

    ts = _prepare_run_ts_search_context(
        ["Pt", "Pt"],
        ts_params=_emt_ts_params(system_type="gas_cluster"),
        output_dir=tmp_path,
        searches_dir=None,
        seed=3,
        verbosity=0,
        surface_config=None,
        system_type="gas_cluster",
        adsorbates=None,
    )
    assert ts.composition == ["Pt", "Pt"]
    assert ts.system_type == "gas_cluster"
    assert ts.seed == 3
    assert ts.adsorbate_definition is None
    assert ts.output_dir == tmp_path.resolve()
