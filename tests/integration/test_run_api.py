"""Thinned public runner API wiring and validation tests (mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from ase import Atoms
from ase.build import fcc111
from ase.calculators.emt import EMT

from scgo import parse_composition_arg
from scgo.exceptions import SCGOValidationError
from scgo.minima_search import run_trials
from scgo.param_presets import (
    get_default_params,
    get_testing_params,
    get_ts_search_params,
)
from scgo.runner_api import (
    build_one_element_compositions,
    build_two_element_compositions,
    resolve_workflow_seed,
    run_go,
    run_go_campaign,
    run_go_ts,
    run_go_ts_campaign,
    run_ts_campaign,
    run_ts_search,
)
from scgo.runner_go import _run_go_campaign_compositions, _run_go_trials
from scgo.runner_params import _prepare_run_go_campaign_context
from scgo.surface.config import SurfaceSystemConfig
from scgo.system_types import get_system_policy
from scgo.utils.helpers import get_composition_counts


def _emt_ts_gasc() -> dict:
    return {
        **get_ts_search_params(
            system_type="gas_cluster",
            calculator="EMT",
            calculator_kwargs={},
        ),
        "use_parallel_neb": False,
        "use_torchsim": False,
    }


def _emt_ts_surf_ads(surface_config: SurfaceSystemConfig) -> dict:
    return {
        **get_ts_search_params(
            system_type="surface_cluster_adsorbate",
            surface_config=surface_config,
            calculator="EMT",
            calculator_kwargs={},
        ),
        "use_parallel_neb": False,
        "use_torchsim": False,
    }


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


def _surface_cfg() -> SurfaceSystemConfig:
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    slab.pbc = [True, True, True]
    return SurfaceSystemConfig(slab=slab, fix_all_slab_atoms=True)


def _slab_search_cfg() -> SurfaceSystemConfig:
    slab = fcc111("Pt", size=(2, 2, 2), vacuum=6.0, orthogonal=True)
    slab.pbc = [True, True, True]
    return SurfaceSystemConfig(
        slab=slab,
        fix_all_slab_atoms=False,
        n_relax_top_slab_layers=1,
        name="pt_slab",
    )


@pytest.mark.parametrize(
    "fn,args",
    [
        pytest.param(
            build_one_element_compositions,
            ("", 2, 4),
            id="one_element_empty_symbol",
        ),
        pytest.param(
            build_one_element_compositions,
            ("Pt", 0, 3),
            id="one_element_min_atoms_zero",
        ),
        pytest.param(
            build_one_element_compositions,
            ("Pt", 5, 3),
            id="one_element_min_gt_max",
        ),
        pytest.param(
            build_two_element_compositions,
            ("", "Pt", 2, 4),
            id="two_elements_empty_first_symbol",
        ),
        pytest.param(
            build_two_element_compositions,
            ("Pt", "", 2, 4),
            id="two_elements_empty_second_symbol",
        ),
        pytest.param(
            build_two_element_compositions,
            ("Pt", "Au", 0, 3),
            id="two_elements_min_atoms_zero",
        ),
        pytest.param(
            build_two_element_compositions,
            ("Pt", "Au", 5, 3),
            id="two_elements_min_gt_max",
        ),
        pytest.param(
            _run_go_campaign_compositions,
            ([], "gas_cluster"),
            id="arbitrary_compositions_empty",
        ),
    ],
)
def test_run_campaign_invalid_inputs(fn, args):
    with pytest.raises(SCGOValidationError):
        fn(*args)


def test_rng_in_optimizer_params_raises():
    params = get_default_params()
    params["optimizer_params"]["ga"] = params["optimizer_params"].get("ga", {})
    params["optimizer_params"]["ga"]["rng"] = "not-allowed"
    with pytest.raises(SCGOValidationError):
        _run_go_trials(["Pt"] * 4, "gas_cluster", params=params)


def test_scgo_validations(rng):
    from scgo.minima_search import scgo

    with pytest.raises(SCGOValidationError):
        scgo(["Pt"], "ga", {}, "out_dir", None)

    with pytest.raises(SCGOValidationError, match="Unknown global_optimizer"):
        scgo(
            ["Pt"],
            "invalid_optimizer",
            {"system_type": "gas_cluster"},
            "out_dir",
            rng,
            calculator_for_global_optimization=EMT(),
        )

    with pytest.raises(SCGOValidationError, match="system_type must be set"):
        scgo(
            ["Pt"],
            "ga",
            {},
            "out_dir",
            rng,
            calculator_for_global_optimization=EMT(),
        )


def test_run_trials_validations(rng):
    with pytest.raises(SCGOValidationError):
        run_trials([], "ga", {}, "out", rng)
    with pytest.raises(SCGOValidationError):
        run_trials(["Pt"], "ga", {}, "", rng)
    with pytest.raises(SCGOValidationError):
        run_trials(["Pt"], "ga", {}, "out", None)
    with pytest.raises(SCGOValidationError):
        run_trials(["Pt"], "ga", {}, "out", rng, verbosity=5)
    with pytest.raises(SCGOValidationError, match="system_type must be set"):
        run_trials(["Pt"], "ga", {}, "out", rng)


@pytest.mark.parametrize(
    "system_type",
    [
        "gas_cluster",
        "surface_cluster",
        "gas_cluster_adsorbate",
        "surface_cluster_adsorbate",
        "surface",
        "surface_adsorbate",
    ],
)
def test_run_go_system_type_matrix(monkeypatch, system_type):
    captured: dict[str, object] = {}

    def _fake_trials(composition, *args, **kwargs):
        captured["params"] = kwargs["params"]
        captured["composition"] = composition
        return []

    monkeypatch.setattr("scgo.runner_go._run_go_trials", _fake_trials)
    policy = get_system_policy(system_type)
    if policy.slab_is_search_target:
        composition: str | list[str] = []
        surface = _slab_search_cfg()
    elif "adsorbate" in system_type:
        composition = ["Pt", "Pt", "Pt"]
        surface = _surface_cfg() if "surface" in system_type else None
    else:
        composition = "Pt3"
        surface = _surface_cfg() if "surface" in system_type else None
    kwargs = {}
    if surface is not None:
        kwargs["surface_config"] = surface
    run_go(
        composition,
        params={"optimizer_params": {"simple": {}, "ga": {}, "bh": {}}},
        verbosity=0,
        system_type=system_type,
        adsorbates=(_adsorbates_oh(n=1) if "adsorbate" in system_type else None),
        **kwargs,
    )
    params = captured["params"]
    assert params["optimizer_params"]["simple"]["system_type"] == system_type
    assert params["optimizer_params"]["ga"]["system_type"] == system_type
    assert params["optimizer_params"]["bh"]["system_type"] == system_type
    if policy.slab_is_search_target and not policy.has_adsorbate:
        assert captured["composition"] == []


def test_system_policy_surface_neb_defaults():
    gas = get_system_policy("gas_cluster")
    bare = get_system_policy("surface_cluster")
    ads = get_system_policy("surface_cluster_adsorbate")
    slab = get_system_policy("surface")
    slab_ads = get_system_policy("surface_adsorbate")
    assert gas.neb_force_mic is False
    assert gas.neb_surface_cell_remap is False
    assert gas.neb_surface_lattice_rotation is False
    assert bare.neb_force_mic is True
    assert bare.neb_surface_cell_remap is True
    assert bare.neb_surface_lattice_rotation is True
    assert ads.neb_force_mic is True
    assert ads.neb_surface_cell_remap is True
    assert ads.neb_surface_lattice_rotation is False
    assert slab.slab_is_search_target is True
    assert slab.neb_force_mic is True
    assert slab.neb_surface_lattice_rotation is True
    assert slab_ads.slab_is_search_target is True
    assert slab_ads.has_adsorbate is True
    assert slab_ads.neb_surface_lattice_rotation is False
    assert slab_ads.needs_supported_deposit_validation is True


def test_run_go_requires_system_type():
    with pytest.raises(SCGOValidationError, match="system_type is required"):
        run_go("Pt3", params=None, verbosity=0)


def test_run_go_requires_adsorbates_for_adsorbate_system_types():
    with pytest.raises(SCGOValidationError, match="adsorbates is required"):
        run_go("Pt5", params=None, verbosity=0, system_type="gas_cluster_adsorbate")


def test_run_go_accepts_valid_adsorbates_input(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_trials(composition, *args, **kwargs):
        captured["composition"] = composition
        return []

    monkeypatch.setattr("scgo.runner_go._run_go_trials", _fake_trials)
    run_go(
        ["Pt", "Pt", "Pt", "Pt", "Pt"],
        params=None,
        verbosity=0,
        system_type="gas_cluster_adsorbate",
        adsorbates=_adsorbates_oh(n=1),
    )
    assert captured["composition"] == ["Pt", "Pt", "Pt", "Pt", "Pt", "O", "H"]


def test_run_go_campaign_normalizes_items(monkeypatch):
    captured: list[list[str]] = []

    def _fake_campaign(compositions, *args, **kwargs):
        captured.extend(compositions)
        return {}

    monkeypatch.setattr(
        "scgo.runner_go._run_go_campaign_compositions",
        _fake_campaign,
    )
    run_go_campaign(
        [Atoms("Au2"), "Pt", ["Cu", "Cu"]],
        params=None,
        verbosity=0,
        system_type="gas_cluster",
    )
    assert captured == [["Au", "Au"], ["Pt"], ["Cu", "Cu"]]


def test_run_go_campaign_reconciles_wrong_preset_core() -> None:
    wrong_core = ["Ru"] * 10 + ["W", "O"]
    params = {
        "adsorbate_definition": {
            "core_symbols": wrong_core,
            "adsorbate_symbols": ["O", "H"],
            "adsorbate_fragment_lengths": [2],
        },
        "adsorbate_fragment_template": _adsorbates_oh(n=1)[0],
    }
    context = _prepare_run_go_campaign_context(
        ["HO2Ru9W2"],
        params=params,
        seed=42,
        verbosity=0,
        run_id=None,
        clean=False,
        output_dir=None,
        surface_config=None,
        system_type="gas_cluster_adsorbate",
        adsorbates=None,
    )
    parsed = parse_composition_arg("HO2Ru9W2")
    assert get_composition_counts(context.compositions[0]) == get_composition_counts(
        parsed
    )


def test_run_go_campaign_reconciles_wrong_preset_core_surface() -> None:
    params = {
        "adsorbate_definition": {
            "core_symbols": ["Pt"] * 4 + ["O"],
            "adsorbate_symbols": ["O", "H"],
            "adsorbate_fragment_lengths": [2],
        },
        "adsorbate_fragment_template": _adsorbates_oh(n=1)[0],
    }
    context = _prepare_run_go_campaign_context(
        [parse_composition_arg("Pt5O2H")],
        params=params,
        seed=42,
        verbosity=0,
        run_id=None,
        clean=False,
        output_dir=None,
        surface_config=_surface_cfg(),
        system_type="surface_cluster_adsorbate",
        adsorbates=None,
    )
    parsed = parse_composition_arg("Pt5O2H")
    assert get_composition_counts(context.compositions[0]) == get_composition_counts(
        parsed
    )


def test_run_go_campaign_empty_raises():
    with pytest.raises(SCGOValidationError, match="empty"):
        run_go_campaign([], params=None, verbosity=0, system_type="gas_cluster")


def test_run_go_campaign_requires_system_type():
    with pytest.raises(SCGOValidationError, match="system_type is required"):
        run_go_campaign(["Pt2"], params=None, verbosity=0)


def test_run_go_campaign_skips_failed_composition(monkeypatch, tmp_path):
    called: list[list[str]] = []

    def fake_trials(composition, system_type, params, **kwargs):
        called.append(list(composition))
        if composition == ["Pt", "Pt"]:
            raise ValueError("init failed")
        return []

    monkeypatch.setattr("scgo.runner_go._run_go_trials", fake_trials)
    monkeypatch.setattr(
        "scgo.runner_go.get_calculator_class",
        lambda name: lambda **kwargs: MagicMock(),
    )
    results = run_go_campaign(
        [["Pt", "Pt"], ["Au", "Au"]],
        params=get_testing_params(),
        seed=0,
        verbosity=0,
        system_type="gas_cluster",
        output_dir=tmp_path,
        clean=True,
    )
    assert called == [["Pt", "Pt"], ["Au", "Au"]]
    assert results["Pt2"] == []
    assert "Au2" in results


def test_run_go_campaign_skips_failed_composition_uses_path_key(monkeypatch, tmp_path):
    called: list[list[str]] = []

    def fake_trials(composition, system_type, params, **kwargs):
        called.append(list(composition))
        if composition == ["Pt", "Pt"]:
            raise ValueError("init failed")
        return [(0.0, Atoms("Au2"))]

    monkeypatch.setattr("scgo.runner_go._run_go_trials", fake_trials)
    monkeypatch.setattr(
        "scgo.runner_go.get_calculator_class",
        lambda name: lambda **kwargs: MagicMock(),
    )
    cfg = _slab_search_cfg()
    results = run_go_campaign(
        [["Pt", "Pt"], ["Au", "Au"]],
        params=get_testing_params(),
        seed=0,
        verbosity=0,
        system_type="surface_cluster",
        surface_config=cfg,
        output_dir=tmp_path,
        clean=True,
    )
    assert called == [["Pt", "Pt"], ["Au", "Au"]]
    assert "Pt2" not in results
    assert results["Pt2_pt_slab"] == []
    assert "Au2_pt_slab" in results
    assert results["Au2_pt_slab"]


def test_run_ts_search_requires_system_type():
    with pytest.raises(SCGOValidationError, match="system_type is required"):
        run_ts_search("Pt2", ts_params=_emt_ts_gasc(), verbosity=0)


def test_run_ts_search_requires_adsorbates_for_adsorbate_system_types():
    with pytest.raises(SCGOValidationError, match="adsorbates is required"):
        run_ts_search(
            "Pt5",
            ts_params={**_emt_ts_gasc()},
            verbosity=0,
            system_type="gas_cluster_adsorbate",
        )


def test_run_ts_search_passes_system_type(monkeypatch):
    captured: dict[str, object] = {}

    def _fake(composition, **kwargs):
        captured["kwargs"] = kwargs
        return []

    monkeypatch.setattr("scgo.runner_ts._ts_search", _fake)
    cfg = _surface_cfg()
    run_ts_search(
        ["Pt", "Pt", "Pt", "Pt", "Pt"],
        ts_params=_emt_ts_surf_ads(cfg),
        verbosity=0,
        surface_config=cfg,
        system_type="surface_cluster_adsorbate",
        adsorbates=_adsorbates_oh(n=2),
    )
    assert captured["kwargs"]["system_type"] == "surface_cluster_adsorbate"


def test_run_ts_campaign_requires_system_type():
    with pytest.raises(SCGOValidationError, match="system_type is required"):
        run_ts_campaign(
            [Atoms("Au2"), "Pt"],
            ts_params=_emt_ts_gasc(),
            verbosity=0,
        )


def test_run_go_ts_campaign_paths(monkeypatch, tmp_path):
    calls: list[tuple[list[str], object]] = []

    def _fake_pipeline(composition, system_type, **kwargs):
        calls.append((list(composition), kwargs.get("output_dir")))
        return {"formula": "x", "ts_total_count": 0}

    monkeypatch.setattr("scgo.runner_ts._run_go_ts_pipeline", _fake_pipeline)
    root = tmp_path / "camp"
    run_go_ts_campaign(
        ["Pt2", ["Au", "Au"]],
        go_params={},
        ts_params=_emt_ts_gasc(),
        verbosity=0,
        output_dir=root,
        system_type="gas_cluster",
    )
    assert len(calls) == 2
    # Unified sibling layout: each composition runs against the shared campaign
    # root (no ``{path_key}_campaign/`` wrapper).
    assert calls[0][0] == ["Pt", "Pt"]
    assert calls[0][1] == root
    assert calls[1][0] == ["Au", "Au"]
    assert calls[1][1] == root


def test_run_go_ts_campaign_requires_system_type():
    with pytest.raises(SCGOValidationError, match="system_type is required"):
        run_go_ts_campaign(
            ["H2"],
            go_params={},
            ts_params=_emt_ts_gasc(),
            verbosity=0,
        )


def test_run_go_ts_uses_default_go_and_ts_presets_when_missing(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_pipeline(composition, system_type, **kwargs):
        captured["go_params"] = kwargs["go_params"]
        captured["ts_kwargs"] = kwargs["ts_kwargs"]
        return {"ts_results": []}

    monkeypatch.setattr("scgo.runner_ts._run_go_ts_pipeline", _fake_pipeline)
    run_go_ts(
        "Pt2",
        go_params=None,
        ts_params=None,
        verbosity=0,
        system_type="gas_cluster",
    )
    go_params = captured["go_params"]
    ts_kwargs = captured["ts_kwargs"]
    assert go_params["calculator"] == "MACE"
    assert ts_kwargs["params"]["calculator"] == "MACE"
    assert ts_kwargs["system_type"] == "gas_cluster"


def test_run_go_ts_rejects_ts_system_type_mismatch():
    with pytest.raises(SCGOValidationError, match="ts_params\\['system_type'\\]"):
        run_go_ts(
            "H2",
            go_params={"optimizer_params": {"ga": {}}},
            ts_params={**_emt_ts_gasc(), "system_type": "surface_cluster_adsorbate"},
            verbosity=0,
            system_type="gas_cluster",
        )


def test_run_ts_search_rejects_ts_system_type_mismatch():
    with pytest.raises(SCGOValidationError, match="ts_params\\['system_type'\\]"):
        run_ts_search(
            "Pt2",
            ts_params={
                **_emt_ts_gasc(),
                "system_type": "surface_cluster_adsorbate",
                "surface_config": _surface_cfg(),
            },
            verbosity=0,
            system_type="gas_cluster",
        )


def test_run_go_ts_rejects_go_optimizer_system_type_mismatch():
    with pytest.raises(SCGOValidationError, match="coherence error"):
        run_go_ts(
            "Pt2",
            go_params={
                "optimizer_params": {
                    "ga": {"system_type": "surface_cluster"},
                    "bh": {},
                    "simple": {},
                }
            },
            ts_params=_emt_ts_gasc(),
            verbosity=0,
            system_type="gas_cluster",
        )


def test_run_go_ts_rejects_ts_surface_config_for_gas_system():
    with pytest.raises(SCGOValidationError, match="coherence error"):
        run_go_ts(
            "Pt2",
            go_params={"optimizer_params": {"ga": {}}},
            ts_params={**_emt_ts_gasc(), "surface_config": _surface_cfg()},
            verbosity=0,
            system_type="gas_cluster",
        )


def test_get_ts_search_params_requires_surface_config_for_surface_systems():
    with pytest.raises(SCGOValidationError, match="requires surface_config"):
        get_ts_search_params(system_type="surface_cluster", calculator="EMT")


def test_resolve_workflow_seed_unifies():
    assert (
        resolve_workflow_seed(seed_kw=1, go_params={"seed": 1}, ts_params={"seed": 1})
        == 1
    )
    assert resolve_workflow_seed(seed_kw=None, go_params={"seed": 2}) == 2


def test_resolve_workflow_seed_rejects_mismatch():
    with pytest.raises(SCGOValidationError, match="Inconsistent random seeds"):
        resolve_workflow_seed(seed_kw=1, go_params={"seed": 2})


def test_run_go_rejects_top_level_go_system_type():
    with pytest.raises(SCGOValidationError, match="does not allow top-level go_params"):
        run_go(
            "Pt3",
            params={"system_type": "gas_cluster", "optimizer_params": {"ga": {}}},
            verbosity=0,
            system_type="gas_cluster",
        )


def test_run_go_ts_rejects_top_level_go_system_type():
    with pytest.raises(SCGOValidationError, match="does not allow top-level go_params"):
        run_go_ts(
            "H2",
            go_params={"system_type": "gas_cluster", "optimizer_params": {"ga": {}}},
            ts_params=_emt_ts_gasc(),
            verbosity=0,
            system_type="gas_cluster",
        )


def test_run_go_ts_rejects_mismatched_seeds():
    with pytest.raises(SCGOValidationError, match="Inconsistent random seeds"):
        run_go_ts(
            "H2",
            go_params={"seed": 1, "optimizer_params": {"ga": {}}},
            ts_params={**_emt_ts_gasc(), "seed": 2},
            seed=1,
            verbosity=0,
            system_type="gas_cluster",
        )


def test_run_go_ts_rejects_mismatched_go_run_surface_config():
    cfg = _surface_cfg()
    slab_b = fcc111("Pt", size=(4, 4, 1), vacuum=6.0, orthogonal=True)
    slab_b.pbc = [True, True, True]
    cfg_b = SurfaceSystemConfig(slab=slab_b, fix_all_slab_atoms=True)
    with pytest.raises(SCGOValidationError, match="surface_config"):
        run_go_ts(
            ["Pt", "Pt", "Pt", "Pt", "Pt"],
            go_params={"surface_config": cfg, "optimizer_params": {"ga": {}}},
            ts_params=_emt_ts_surf_ads(cfg),
            verbosity=0,
            surface_config=cfg_b,
            system_type="surface_cluster_adsorbate",
            adsorbates=_adsorbates_oh(n=2),
        )


def test_run_go_ts_rejects_mismatched_ts_run_surface_config():
    cfg = _surface_cfg()
    slab_b = fcc111("Pt", size=(4, 4, 1), vacuum=6.0, orthogonal=True)
    slab_b.pbc = [True, True, True]
    cfg_b = SurfaceSystemConfig(slab=slab_b, fix_all_slab_atoms=True)
    with pytest.raises(SCGOValidationError, match="surface_config"):
        run_go_ts(
            ["Pt", "Pt", "Pt", "Pt", "Pt"],
            go_params={"surface_config": cfg, "optimizer_params": {"ga": {}}},
            ts_params=_emt_ts_surf_ads(cfg_b),
            verbosity=0,
            surface_config=cfg,
            system_type="surface_cluster_adsorbate",
            adsorbates=_adsorbates_oh(n=2),
        )


def test_run_go_ts_accepts_run_surface_config_without_go_top_level(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_pipeline(composition, system_type, **kwargs):
        captured["go_params"] = kwargs["go_params"]
        return {"ts_results": []}

    monkeypatch.setattr("scgo.runner_ts._run_go_ts_pipeline", _fake_pipeline)
    cfg = _surface_cfg()
    run_go_ts(
        ["Pt", "Pt", "Pt", "Pt", "Pt"],
        go_params={"optimizer_params": {"ga": {}}},
        ts_params=_emt_ts_surf_ads(cfg),
        verbosity=0,
        surface_config=cfg,
        system_type="surface_cluster_adsorbate",
        adsorbates=_adsorbates_oh(n=2),
    )
    go_params = captured["go_params"]
    assert go_params["surface_config"] == cfg
    assert go_params["optimizer_params"]["ga"]["surface_config"] == cfg
