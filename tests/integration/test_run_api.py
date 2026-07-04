import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111

from scgo import parse_composition_arg
from scgo.minima_search import run_trials
from scgo.param_presets import (
    get_default_params,
    get_testing_params,
    get_ts_search_params,
)
from scgo.runner_api import (
    _run_go_campaign_compositions,
    _run_go_trials,
    _run_go_ts_pipeline,
    _run_one_element_go_ts_pipeline,
    build_one_element_compositions,
    build_two_element_compositions,
    log_go_ts_summary,
    resolve_workflow_seed,
    run_go,
    run_go_campaign,
    run_go_ts,
    run_go_ts_campaign,
    run_ts_campaign,
    run_ts_search,
)
from scgo.surface.config import SurfaceSystemConfig
from scgo.system_types import get_system_policy
from scgo.utils.ts_runner_kwargs import coerce_ts_params_to_runner_kwargs
from tests.test_utils import isolated_workflow_cwd


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


def test_parse_composition_arg_formats():
    assert parse_composition_arg("Pt,Pt,Au") == ["Pt", "Pt", "Au"]
    assert parse_composition_arg("Pt3Au") == ["Pt", "Pt", "Pt", "Au"]
    assert parse_composition_arg("Pt10") == ["Pt"] * 10


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
    with pytest.raises(ValueError):
        fn(*args)


def test_rng_in_optimizer_params_raises():
    params = get_default_params()
    # inject forbidden 'rng' into optimizer params
    params["optimizer_params"]["ga"] = params["optimizer_params"].get("ga", {})
    params["optimizer_params"]["ga"]["rng"] = "not-allowed"

    with pytest.raises(ValueError):
        _run_go_trials(["Pt"] * 4, "gas_cluster", params=params)


def test_scgo_validations(rng):
    # Use deterministic rng fixture from conftest

    # Invalid RNG
    with pytest.raises(ValueError):
        from scgo.minima_search import scgo

        scgo(["Pt"], "ga", {}, "out_dir", None)

    # Invalid optimizer name
    with pytest.raises(ValueError):
        scgo(["Pt"], "invalid_optimizer", {}, "out_dir", rng)

    # Invalid system_type in optimizer kwargs
    with pytest.raises(ValueError, match="system_type must be set"):
        from scgo.minima_search import scgo

        scgo(["Pt"], "ga", {}, "out_dir", rng)


def test_run_trials_validations(rng):
    with pytest.raises(ValueError):
        run_trials([], "ga", {}, "out", rng)

    with pytest.raises(ValueError):
        run_trials(["Pt"], 123, {}, "out", rng)

    with pytest.raises(ValueError):
        run_trials(["Pt"], "ga", {}, "", rng)

    with pytest.raises(ValueError):
        run_trials(["Pt"], "ga", {}, "out", None)

    with pytest.raises(ValueError):
        run_trials(["Pt"], "ga", {}, "out", rng, verbosity=5)

    with pytest.raises(ValueError, match="system_type must be set"):
        run_trials(["Pt"], "ga", {}, "out", rng)


def test_parse_composition_arg_case_insensitive():
    assert parse_composition_arg("pt3") == ["Pt", "Pt", "Pt"]
    assert parse_composition_arg("pt3au") == ["Pt", "Pt", "Pt", "Au"]
    assert parse_composition_arg("pt,pt,au") == ["Pt", "Pt", "Au"]


def test_parse_composition_arg_zero_count():
    # Zero counts in compact formula should be rejected
    with pytest.raises(ValueError):
        parse_composition_arg("Pt0")
    with pytest.raises(ValueError):
        parse_composition_arg("pt0au")
    with pytest.raises(ValueError):
        parse_composition_arg("AuPt0")


def test_seed_in_params_respected(tmp_path):
    params = get_testing_params()
    params["seed"] = 12345

    comp = ["Pt", "Pt"]  # Pt2 small test

    with isolated_workflow_cwd(tmp_path / "run_a"):
        res1 = _run_go_trials(
            comp,
            "gas_cluster",
            params=params,
            verbosity=0,
            output_dir=str(tmp_path / "run_a"),
            clean=True,
        )
    with isolated_workflow_cwd(tmp_path / "run_b"):
        res2 = _run_go_trials(
            comp,
            "gas_cluster",
            params=params,
            verbosity=0,
            output_dir=str(tmp_path / "run_b"),
            clean=True,
        )

    # Compare basic properties - energies should be very close and compositions identical
    assert len(res1) == len(res2)
    for (e1, a1), (e2, a2) in zip(res1, res2, strict=True):
        assert abs(e1 - e2) < 1e-6  # Energy tolerance
        # Check that compositions are identical
        assert a1.get_chemical_symbols() == a2.get_chemical_symbols()
        # Check that cell and PBC are identical
        assert np.allclose(a1.get_cell(), a2.get_cell())
        assert np.array_equal(a1.get_pbc(), a2.get_pbc())


def test_campaign_respects_params_seed(tmp_path):
    params = get_testing_params()
    params["seed"] = 54321

    comps = build_one_element_compositions("Pt", 2, 2)
    with isolated_workflow_cwd(tmp_path / "campaign_a"):
        res_a = _run_go_campaign_compositions(
            comps,
            "gas_cluster",
            params=params,
            verbosity=0,
            output_dir=str(tmp_path / "campaign_a"),
            clean=True,
        )
    with isolated_workflow_cwd(tmp_path / "campaign_b"):
        res_b = _run_go_campaign_compositions(
            comps,
            "gas_cluster",
            params=params,
            verbosity=0,
            output_dir=str(tmp_path / "campaign_b"),
            clean=True,
        )

    assert res_a == res_b


def test__run_one_element_go_ts_pipeline_smoke(monkeypatch, tmp_path):
    import scgo.runner_api as runner_api_module
    import scgo.ts_search as ts_search_module

    def _fake_trials(*args, **kwargs):
        return []

    def _fake_ts(*args, **kwargs):
        return [{"status": "success"}, {"status": "failed"}]

    monkeypatch.setattr(
        runner_api_module,
        "_run_go_trials",
        _fake_trials,
    )
    monkeypatch.setattr(
        ts_search_module,
        "run_transition_state_search",
        _fake_ts,
    )

    flat_ts = {
        **get_ts_search_params(
            system_type="gas_cluster",
            calculator="EMT",
            calculator_kwargs={},
        ),
        "use_torchsim": False,
        "use_parallel_neb": False,
        "max_pairs": 2,
    }
    summary = _run_one_element_go_ts_pipeline(
        "Pt",
        5,
        "gas_cluster",
        go_params=get_testing_params(),
        ts_kwargs=coerce_ts_params_to_runner_kwargs(flat_ts, system_type="gas_cluster"),
        seed=42,
        verbosity=0,
        output_dir=tmp_path / "pt5_gas",
    )
    assert summary["formula"] == "Pt5"
    assert summary["ts_success_count"] == 1
    assert summary["ts_total_count"] == 2
    assert summary["output_dir"] == (tmp_path / "pt5_gas").resolve()


def test_run_go_atoms_matches_explicit_list(monkeypatch):
    captured: dict[str, list] = {}

    def _fake_trials(composition, *args, **kwargs):
        captured["composition"] = composition
        return []

    monkeypatch.setattr("scgo.runner_api._run_go_trials", _fake_trials)

    run_go(Atoms("Pt3"), params=None, verbosity=0, system_type="gas_cluster")
    assert captured["composition"] == ["Pt", "Pt", "Pt"]

    run_go(["Pt", "Pt", "Pt"], params=None, verbosity=0, system_type="gas_cluster")
    assert captured["composition"] == ["Pt", "Pt", "Pt"]

    run_go("Pt3", params=None, verbosity=0, system_type="gas_cluster")
    assert captured["composition"] == ["Pt", "Pt", "Pt"]


def test_run_go_system_type_wires_optimizer_params(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_trials(composition, *args, **kwargs):
        captured["params"] = kwargs["params"]
        return []

    monkeypatch.setattr("scgo.runner_api._run_go_trials", _fake_trials)
    run_go(
        ["Pt", "Pt", "Pt", "Pt", "Pt"],
        params={"optimizer_params": {"ga": {}, "bh": {}}},
        verbosity=0,
        surface_config=_surface_cfg(),
        system_type="surface_cluster_adsorbate",
        adsorbates=_adsorbates_oh(n=2),
    )
    params = captured["params"]
    assert (
        params["optimizer_params"]["ga"]["system_type"] == "surface_cluster_adsorbate"
    )
    assert (
        params["optimizer_params"]["bh"]["system_type"] == "surface_cluster_adsorbate"
    )
    assert (
        params["optimizer_params"]["simple"]["system_type"]
        == "surface_cluster_adsorbate"
    )


def test_run_go_timing_from_params(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_trials(composition, *args, **kwargs):
        captured["params"] = kwargs["params"]
        return []

    monkeypatch.setattr("scgo.runner_api._run_go_trials", _fake_trials)
    run_go(
        "Pt3",
        params={
            "optimizer_params": {
                "ga": {"write_timing_json": True, "detailed_timing": True},
            }
        },
        verbosity=0,
        system_type="gas_cluster",
    )
    ga = captured["params"]["optimizer_params"]["ga"]
    assert ga["write_timing_json"] is True
    assert ga["detailed_timing"] is True


@pytest.mark.parametrize(
    "system_type",
    [
        "gas_cluster",
        "surface_cluster",
        "gas_cluster_adsorbate",
        "surface_cluster_adsorbate",
    ],
)
def test_run_go_system_type_matrix(monkeypatch, system_type):
    captured: dict[str, object] = {}

    def _fake_trials(composition, *args, **kwargs):
        captured["params"] = kwargs["params"]
        return []

    monkeypatch.setattr("scgo.runner_api._run_go_trials", _fake_trials)
    composition = ["Pt", "Pt", "Pt"] if "adsorbate" in system_type else "Pt3"
    kwargs = {}
    if "surface" in system_type:
        kwargs["surface_config"] = _surface_cfg()
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


def test_system_policy_surface_neb_defaults():
    gas = get_system_policy("gas_cluster")
    surf = get_system_policy("surface_cluster_adsorbate")
    assert gas.neb_force_mic is False
    assert gas.neb_disable_alignment is False
    assert gas.neb_surface_cell_remap is False
    assert gas.neb_surface_lattice_rotation is False
    assert surf.neb_force_mic is True
    assert surf.neb_disable_alignment is False
    assert surf.neb_surface_cell_remap is True
    assert surf.neb_surface_lattice_rotation is True


def test_run_go_requires_system_type():
    with pytest.raises(ValueError, match="system_type is required"):
        run_go("Pt3", params=None, verbosity=0)


def test_run_go_requires_adsorbates_for_adsorbate_system_types():
    with pytest.raises(ValueError, match="adsorbates is required"):
        run_go(
            "Pt5",
            params=None,
            verbosity=0,
            system_type="gas_cluster_adsorbate",
        )


def test_run_go_accepts_valid_adsorbates_input(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_trials(composition, *args, **kwargs):
        captured["composition"] = composition
        return []

    monkeypatch.setattr("scgo.runner_api._run_go_trials", _fake_trials)
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
        "scgo.runner_api._run_go_campaign_compositions",
        _fake_campaign,
    )

    run_go_campaign(
        [Atoms("Au2"), "Pt", ["Cu", "Cu"]],
        params=None,
        verbosity=0,
        system_type="gas_cluster",
    )
    assert captured == [["Au", "Au"], ["Pt"], ["Cu", "Cu"]]


def test_run_go_campaign_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        run_go_campaign([], params=None, verbosity=0, system_type="gas_cluster")


def test_run_go_campaign_requires_system_type():
    with pytest.raises(ValueError, match="system_type is required"):
        run_go_campaign(["Pt2"], params=None, verbosity=0)


def test_run_go_campaign_skips_failed_composition(monkeypatch, tmp_path):
    """A single composition failure should not abort the whole campaign."""
    from unittest.mock import MagicMock

    called: list[list[str]] = []

    def fake_trials(composition, system_type, params, **kwargs):
        called.append(list(composition))
        if composition == ["Pt", "Pt"]:
            raise ValueError("init failed")
        return []

    monkeypatch.setattr("scgo.runner_api._run_go_trials", fake_trials)
    monkeypatch.setattr(
        "scgo.runner_api.get_calculator_class",
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


def test_run_ts_search_normalizes_composition(monkeypatch):
    captured: dict[str, list] = {}

    def _fake(composition, **kwargs):
        captured["composition"] = composition
        return []

    monkeypatch.setattr(
        "scgo.runner_api._ts_search",
        _fake,
    )

    run_ts_search(
        "Cu2",
        ts_params=_emt_ts_gasc(),
        verbosity=0,
        system_type="gas_cluster",
    )
    assert captured["composition"] == ["Cu", "Cu"]

    run_ts_search(
        Atoms("Cu2"),
        ts_params=_emt_ts_gasc(),
        verbosity=0,
        system_type="gas_cluster",
    )
    assert captured["composition"] == ["Cu", "Cu"]

    run_ts_search(
        ["Cu", "Cu"],
        ts_params=_emt_ts_gasc(),
        verbosity=0,
        system_type="gas_cluster",
    )
    assert captured["composition"] == ["Cu", "Cu"]


def test_run_ts_search_passes_system_type(monkeypatch):
    captured: dict[str, object] = {}

    def _fake(composition, **kwargs):
        captured["kwargs"] = kwargs
        return []

    monkeypatch.setattr("scgo.runner_api._ts_search", _fake)
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


def test_run_ts_search_requires_system_type():
    with pytest.raises(ValueError, match="system_type is required"):
        run_ts_search("Pt2", ts_params=_emt_ts_gasc(), verbosity=0)


def test_run_ts_search_requires_adsorbates_for_adsorbate_system_types():
    with pytest.raises(ValueError, match="adsorbates is required"):
        run_ts_search(
            "Pt5",
            ts_params={
                **_emt_ts_gasc(),
            },
            verbosity=0,
            system_type="gas_cluster_adsorbate",
        )


def test_run_ts_search_uses_default_ts_preset_when_missing(monkeypatch):
    captured: dict[str, object] = {}

    def _fake(composition, **kwargs):
        captured["kwargs"] = kwargs
        return []

    monkeypatch.setattr("scgo.runner_api._ts_search", _fake)
    run_ts_search(
        "Pt2",
        ts_params=None,
        verbosity=0,
        system_type="gas_cluster",
    )
    assert captured["kwargs"]["params"]["calculator"] == "MACE"
    assert captured["kwargs"]["system_type"] == "gas_cluster"


def test_run_ts_search_empty_ts_params_uses_defaults(monkeypatch):
    captured: dict[str, object] = {}

    def _fake(composition, **kwargs):
        captured["kwargs"] = kwargs
        return []

    monkeypatch.setattr("scgo.runner_api._ts_search", _fake)
    run_ts_search(
        "Pt2",
        ts_params={},
        verbosity=0,
        system_type="gas_cluster",
    )
    assert captured["kwargs"]["params"]["calculator"] == "MACE"
    assert captured["kwargs"]["system_type"] == "gas_cluster"


def test_run_ts_campaign_normalizes_items(monkeypatch):
    captured: list[list[str]] = []

    def _fake(compositions, system_type, **kwargs):
        captured.extend(compositions)
        return {}

    monkeypatch.setattr(
        "scgo.runner_api._ts_campaign",
        _fake,
    )

    run_ts_campaign(
        [Atoms("Au2"), "Pt"],
        ts_params=_emt_ts_gasc(),
        verbosity=0,
        system_type="gas_cluster",
    )
    assert captured == [["Au", "Au"], ["Pt"]]


def test_run_ts_campaign_empty_ts_params_uses_defaults(monkeypatch):
    captured: list[list[str]] = []

    def _fake(compositions, system_type, **kwargs):
        captured.extend(compositions)
        return {}

    monkeypatch.setattr("scgo.runner_api._ts_campaign", _fake)
    run_ts_campaign(
        [Atoms("Au2"), "Pt"],
        ts_params={},
        verbosity=0,
        system_type="gas_cluster",
    )
    assert captured == [["Au", "Au"], ["Pt"]]


def test_run_ts_campaign_requires_system_type():
    with pytest.raises(ValueError, match="system_type is required"):
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

    monkeypatch.setattr("scgo.runner_api._run_go_ts_pipeline", _fake_pipeline)

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
    assert calls[0][0] == ["Pt", "Pt"]
    assert calls[0][1] == root / "Pt2_campaign"
    assert calls[1][0] == ["Au", "Au"]
    assert calls[1][1] == root / "Au2_campaign"


def test_run_go_ts_passes_timing_from_go_params(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_pipeline(composition, system_type, **kwargs):
        captured["go_params"] = kwargs["go_params"]
        return {"ts_results": []}

    monkeypatch.setattr("scgo.runner_api._run_go_ts_pipeline", _fake_pipeline)
    run_go_ts(
        "Pt2",
        go_params={
            "optimizer_params": {
                "ga": {"write_timing_json": True, "detailed_timing": True},
            }
        },
        ts_params=_emt_ts_gasc(),
        verbosity=0,
        system_type="gas_cluster",
    )
    ga = captured["go_params"]["optimizer_params"]["ga"]
    assert ga["write_timing_json"] is True
    assert ga["detailed_timing"] is True


def test_run_go_ts_passes_adsorbate_definition_to_pipeline(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_pipeline(composition, system_type, **kwargs):
        captured["composition"] = composition
        captured["system_type"] = system_type
        captured["adsorbate_definition"] = kwargs.get("adsorbate_definition")
        return {"ts_results": []}

    monkeypatch.setattr("scgo.runner_api._run_go_ts_pipeline", _fake_pipeline)
    run_go_ts(
        ["Pt", "Pt", "Pt", "Pt", "Pt"],
        go_params={"optimizer_params": {"ga": {}}},
        ts_params=_emt_ts_gasc(),
        verbosity=0,
        system_type="gas_cluster_adsorbate",
        adsorbates=_adsorbates_oh(n=1),
    )
    ads_def = captured["adsorbate_definition"]
    assert captured["system_type"] == "gas_cluster_adsorbate"
    assert ads_def is not None
    assert ads_def["core_symbols"] == ["Pt", "Pt", "Pt", "Pt", "Pt"]
    assert ads_def["adsorbate_symbols"] == ["O", "H"]
    assert ads_def["adsorbate_fragment_lengths"] == [2]
    assert True


def test_go_ts_pipeline_forwards_adsorbate_definition_to_ts(monkeypatch, tmp_path):
    import scgo.runner_api as runner_api_module
    import scgo.ts_search as ts_search_module

    captured: dict[str, object] = {}

    def _fake_trials(*args, **kwargs):
        return []

    def _fake_ts(*args, **kwargs):
        captured["adsorbate_definition"] = kwargs.get("adsorbate_definition")
        captured["system_type"] = kwargs.get("system_type")
        return []

    monkeypatch.setattr(runner_api_module, "_run_go_trials", _fake_trials)
    monkeypatch.setattr(ts_search_module, "run_transition_state_search", _fake_ts)

    ads_def = {
        "core_symbols": ["Pt", "Pt", "Pt", "Pt", "Pt"],
        "adsorbate_symbols": ["O", "H"],
        "adsorbate_fragment_lengths": [2],
    }
    _run_go_ts_pipeline(
        ["Pt", "Pt", "Pt", "Pt", "Pt", "O", "H"],
        "gas_cluster_adsorbate",
        go_params=get_testing_params(),
        ts_kwargs=coerce_ts_params_to_runner_kwargs(
            _emt_ts_gasc(), system_type="gas_cluster_adsorbate"
        ),
        adsorbate_definition=ads_def,
        seed=42,
        verbosity=0,
        output_dir=tmp_path / "pt5oh_gas",
    )
    assert captured["system_type"] == "gas_cluster_adsorbate"
    assert captured["adsorbate_definition"] == ads_def


def test_run_go_ts_accepts_top_level_go_surface_config(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_pipeline(composition, system_type, **kwargs):
        captured["go_params"] = kwargs["go_params"]
        return {"ts_results": []}

    monkeypatch.setattr("scgo.runner_api._run_go_ts_pipeline", _fake_pipeline)
    cfg = _surface_cfg()
    run_go_ts(
        ["Pt", "Pt", "Pt", "Pt", "Pt"],
        go_params={"surface_config": cfg, "optimizer_params": {"ga": {}}},
        ts_params=_emt_ts_surf_ads(cfg),
        verbosity=0,
        surface_config=cfg,
        system_type="surface_cluster_adsorbate",
        adsorbates=_adsorbates_oh(n=2),
    )
    go_params = captured["go_params"]
    assert go_params["surface_config"] == cfg
    assert go_params["optimizer_params"]["ga"]["surface_config"] == cfg


def test_run_go_ts_campaign_no_output_dir(
    monkeypatch,
):
    calls: list[object] = []

    def _fake_pipeline(composition, system_type, **kwargs):
        calls.append(kwargs.get("output_dir"))
        return {}

    monkeypatch.setattr("scgo.runner_api._run_go_ts_pipeline", _fake_pipeline)

    run_go_ts_campaign(
        ["H2"],
        go_params={},
        ts_params=_emt_ts_gasc(),
        verbosity=0,
        output_dir=None,
        system_type="gas_cluster",
    )
    assert len(calls) == 1
    assert calls[0] is not None
    assert "H2_campaign" in str(calls[0])


def test_run_go_ts_campaign_empty_ts_params_uses_defaults(monkeypatch):
    calls: list[tuple[list[str], object]] = []

    def _fake_pipeline(composition, system_type, **kwargs):
        calls.append((list(composition), kwargs.get("ts_kwargs")))
        return {"formula": "x", "ts_total_count": 0}

    monkeypatch.setattr("scgo.runner_api._run_go_ts_pipeline", _fake_pipeline)
    run_go_ts_campaign(
        ["H2"],
        go_params={},
        ts_params={},
        verbosity=0,
        output_dir=None,
        system_type="gas_cluster",
    )
    assert len(calls) == 1
    assert calls[0][1]["params"]["calculator"] == "MACE"


def test_run_go_ts_campaign_requires_system_type():
    with pytest.raises(ValueError, match="system_type is required"):
        run_go_ts_campaign(
            ["H2"],
            go_params={},
            ts_params=_emt_ts_gasc(),
            verbosity=0,
            output_dir=None,
        )


def test_run_go_ts_empty_ts_params_uses_defaults(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_pipeline(composition, system_type, **kwargs):
        captured["ts_kwargs"] = kwargs["ts_kwargs"]
        return {"ts_results": []}

    monkeypatch.setattr("scgo.runner_api._run_go_ts_pipeline", _fake_pipeline)
    run_go_ts(
        "H2",
        go_params={},
        ts_params={},
        verbosity=0,
        system_type="gas_cluster",
    )
    assert captured["ts_kwargs"]["params"]["calculator"] == "MACE"


def test_run_go_ts_uses_default_go_and_ts_presets_when_missing(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_pipeline(composition, system_type, **kwargs):
        captured["go_params"] = kwargs["go_params"]
        captured["ts_kwargs"] = kwargs["ts_kwargs"]
        return {"ts_results": []}

    monkeypatch.setattr("scgo.runner_api._run_go_ts_pipeline", _fake_pipeline)
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


def test_run_go_ts_default_presets_match_builders_for_key_fields(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_pipeline(composition, system_type, **kwargs):
        captured["go_params"] = kwargs["go_params"]
        captured["ts_kwargs"] = kwargs["ts_kwargs"]
        return {"ts_results": []}

    monkeypatch.setattr("scgo.runner_api._run_go_ts_pipeline", _fake_pipeline)
    run_go_ts(
        "Pt2", go_params=None, ts_params=None, verbosity=0, system_type="gas_cluster"
    )

    used_go_params = captured["go_params"]
    used_ts_kwargs = captured["ts_kwargs"]
    expected_go = get_default_params()
    expected_ts = get_ts_search_params(
        calculator=expected_go["calculator"],
        calculator_kwargs=expected_go.get("calculator_kwargs"),
        system_type="gas_cluster",
    )
    expected_ts_kwargs = coerce_ts_params_to_runner_kwargs(
        expected_ts, system_type="gas_cluster"
    )

    assert used_go_params["calculator"] == expected_go["calculator"]
    assert used_go_params["calculator_kwargs"] == expected_go["calculator_kwargs"]
    assert used_ts_kwargs["params"]["calculator"] == expected_ts["calculator"]
    assert (
        used_ts_kwargs["params"]["calculator_kwargs"]
        == expected_ts["calculator_kwargs"]
    )
    assert used_ts_kwargs["neb_n_images"] == expected_ts_kwargs["neb_n_images"]
    assert used_ts_kwargs["neb_fmax"] == expected_ts_kwargs["neb_fmax"]
    assert used_ts_kwargs["system_type"] == "gas_cluster"


def test_run_ts_search_default_ts_preset_matches_builder(monkeypatch):
    captured: dict[str, object] = {}

    def _fake(composition, **kwargs):
        captured["kwargs"] = kwargs
        return []

    monkeypatch.setattr("scgo.runner_api._ts_search", _fake)
    run_ts_search("Pt2", ts_params=None, verbosity=0, system_type="gas_cluster")

    used_kwargs = captured["kwargs"]
    expected_ts = get_ts_search_params(system_type="gas_cluster")
    expected_kwargs = coerce_ts_params_to_runner_kwargs(
        expected_ts, system_type="gas_cluster"
    )

    assert used_kwargs["params"] == expected_kwargs["params"]
    assert used_kwargs["system_type"] == expected_kwargs["system_type"]


def test_run_go_ts_rejects_ts_system_type_mismatch():
    with pytest.raises(ValueError, match="ts_params\\['system_type'\\]"):
        run_go_ts(
            "H2",
            go_params={"optimizer_params": {"ga": {}}},
            ts_params={**_emt_ts_gasc(), "system_type": "surface_cluster_adsorbate"},
            verbosity=0,
            system_type="gas_cluster",
        )


def test_run_ts_search_rejects_ts_system_type_mismatch():
    with pytest.raises(ValueError, match="ts_params\\['system_type'\\]"):
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
    with pytest.raises(ValueError, match="coherence error"):
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
    with pytest.raises(ValueError, match="coherence error"):
        run_go_ts(
            "Pt2",
            go_params={"optimizer_params": {"ga": {}}},
            ts_params={**_emt_ts_gasc(), "surface_config": _surface_cfg()},
            verbosity=0,
            system_type="gas_cluster",
        )


def test_get_ts_search_params_requires_surface_config_for_surface_systems():
    with pytest.raises(ValueError, match="requires surface_config"):
        get_ts_search_params(system_type="surface_cluster", calculator="EMT")


def test_resolve_workflow_seed_unifies():
    assert (
        resolve_workflow_seed(seed_kw=1, go_params={"seed": 1}, ts_params={"seed": 1})
        == 1
    )
    assert resolve_workflow_seed(seed_kw=None, go_params={"seed": 2}) == 2


def test_resolve_workflow_seed_rejects_mismatch():
    with pytest.raises(ValueError, match="Inconsistent random seeds"):
        resolve_workflow_seed(seed_kw=1, go_params={"seed": 2})


def test_run_go_rejects_top_level_go_system_type():
    with pytest.raises(ValueError, match="does not allow top-level go_params"):
        run_go(
            "Pt3",
            params={"system_type": "gas_cluster", "optimizer_params": {"ga": {}}},
            verbosity=0,
            system_type="gas_cluster",
        )


def test_run_go_ts_rejects_top_level_go_system_type():
    with pytest.raises(ValueError, match="does not allow top-level go_params"):
        run_go_ts(
            "H2",
            go_params={"system_type": "gas_cluster", "optimizer_params": {"ga": {}}},
            ts_params=_emt_ts_gasc(),
            verbosity=0,
            system_type="gas_cluster",
        )


def test_run_go_ts_rejects_mismatched_seeds():
    with pytest.raises(ValueError, match="Inconsistent random seeds"):
        run_go_ts(
            "H2",
            go_params={"seed": 1, "optimizer_params": {"ga": {}}},
            ts_params={**_emt_ts_gasc(), "seed": 2},
            seed=1,
            verbosity=0,
            system_type="gas_cluster",
        )


def test_run_go_ts_rejects_mismatched_go_run_surface_config():
    cfg_a = _surface_cfg()
    slab_b = fcc111("Pt", size=(4, 4, 1), vacuum=6.0, orthogonal=True)
    slab_b.pbc = [True, True, True]
    cfg_b = SurfaceSystemConfig(slab=slab_b, fix_all_slab_atoms=True)
    with pytest.raises(ValueError, match="surface_config"):
        run_go_ts(
            ["Pt", "Pt", "Pt", "Pt", "Pt"],
            go_params={"surface_config": cfg_a, "optimizer_params": {"ga": {}}},
            ts_params=_emt_ts_surf_ads(cfg_a),
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
    with pytest.raises(ValueError, match="surface_config"):
        run_go_ts(
            ["Pt", "Pt", "Pt", "Pt", "Pt"],
            go_params={"surface_config": cfg, "optimizer_params": {"ga": {}}},
            ts_params=_emt_ts_surf_ads(cfg),
            verbosity=0,
            surface_config=cfg_b,
            system_type="surface_cluster_adsorbate",
            adsorbates=_adsorbates_oh(n=2),
        )


def test_run_go_ts_accepts_run_surface_config_without_go_top_level(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_pipeline(composition, system_type, **kwargs):
        captured["go_params"] = kwargs["go_params"]
        return {"ts_results": []}

    monkeypatch.setattr("scgo.runner_api._run_go_ts_pipeline", _fake_pipeline)
    cfg = _surface_cfg()
    run_go_ts(
        ["Pt", "Pt", "Pt", "Pt", "Pt"],
        go_params={"optimizer_params": {"ga": {}}},
        ts_params={
            "calculator": "EMT",
            "calculator_kwargs": {},
            "use_torchsim": False,
            "use_parallel_neb": False,
        },
        verbosity=0,
        surface_config=cfg,
        system_type="surface_cluster_adsorbate",
        adsorbates=_adsorbates_oh(n=2),
    )
    go_params = captured["go_params"]
    assert go_params["surface_config"] == cfg
    assert go_params["optimizer_params"]["ga"]["surface_config"] == cfg


def test_log_go_ts_summary():
    class _Log:
        def __init__(self) -> None:
            self.messages: list[str] = []

        def info(self, fmt: str, *args: object) -> None:
            self.messages.append(fmt % args if args else fmt)

    log = _Log()
    log_go_ts_summary(
        log,
        {"ts_results": [{"status": "success"}, {"status": "failed"}]},
        wall_time_s=3.25,
    )
    assert log.messages[0] == "Successful NEBs: 1/2"
    assert log.messages[1] == "Total wall time: 3.25 s"
