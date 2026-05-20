"""Tests for TS parameter presets and run-kwargs mapping."""

import pytest
from ase.build import fcc111

import scgo.param_presets as param_presets_module
from scgo.constants import DEFAULT_ENERGY_TOLERANCE, DEFAULT_NEB_TANGENT_METHOD
from scgo.param_presets import (
    TS_DEFAULTS_BY_SYSTEM_TYPE,
    get_default_params,
    get_torchsim_ga_params,
    get_ts_defaults,
    get_ts_search_params,
)
from scgo.surface.config import SurfaceSystemConfig
from scgo.system_types import SYSTEM_TYPE_POLICIES, get_system_policy
from scgo.utils.run_helpers import prepare_algorithm_kwargs
from scgo.utils.ts_runner_kwargs import coerce_ts_params_to_runner_kwargs


def _surface_config_for_test() -> SurfaceSystemConfig:
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    slab.pbc = [True, True, True]
    return SurfaceSystemConfig(slab=slab, fix_all_slab_atoms=True)


def _ts_search_params_for(system_type: str) -> dict:
    if get_system_policy(system_type).uses_surface:
        return get_ts_search_params(
            system_type=system_type, surface_config=_surface_config_for_test()
        )
    return get_ts_search_params(system_type=system_type)


@pytest.mark.parametrize("system_type", sorted(TS_DEFAULTS_BY_SYSTEM_TYPE))
def test_ts_defaults_match_system_policy_align_and_mic(system_type):
    """`TS_DEFAULTS_BY_SYSTEM_TYPE` must agree with `SystemPolicy` flags."""
    defaults = get_ts_defaults(system_type)
    policy = SYSTEM_TYPE_POLICIES[system_type]
    assert defaults["neb_align_endpoints"] is (not policy.neb_disable_alignment)
    assert defaults["neb_interpolation_mic"] is policy.neb_force_mic
    assert defaults["neb_surface_cell_remap"] is policy.neb_surface_cell_remap
    assert (
        defaults["neb_surface_lattice_rotation"] is policy.neb_surface_lattice_rotation
    )


@pytest.mark.parametrize("system_type", sorted(TS_DEFAULTS_BY_SYSTEM_TYPE))
def test_get_ts_search_params_seeds_from_per_system_defaults(system_type):
    """Each system type's preset reflects its `get_ts_defaults` block."""
    ts = _ts_search_params_for(system_type)
    defaults = get_ts_defaults(system_type)
    for key, expected in defaults.items():
        assert ts[key] == expected, (
            f"{system_type}: ts_params[{key!r}]={ts[key]!r} != defaults[{key!r}]={expected!r}"
        )


@pytest.mark.parametrize("system_type", sorted(TS_DEFAULTS_BY_SYSTEM_TYPE))
def test_coerce_sparse_ts_params_falls_back_to_per_system_defaults(system_type):
    """A sparse `ts_params` flows through with policy-coherent NEB defaults."""
    sparse: dict = {"calculator": "MACE"}
    if get_system_policy(system_type).uses_surface:
        sparse["surface_config"] = _surface_config_for_test()
    kwargs = coerce_ts_params_to_runner_kwargs(sparse, system_type=system_type)
    defaults = get_ts_defaults(system_type)
    for key in (
        "neb_align_endpoints",
        "neb_interpolation_mic",
        "neb_n_images",
        "neb_spring_constant",
        "neb_fmax",
        "neb_steps",
        "neb_climb",
        "neb_perturb_sigma",
        "neb_interpolation_method",
        "neb_tangent_method",
        "neb_surface_cell_remap",
        "neb_surface_lattice_rotation",
    ):
        expected = defaults[key]
        assert kwargs[key] == expected, (
            f"{system_type}: kwargs[{key!r}]={kwargs[key]!r} != defaults[{key!r}]={expected!r}"
        )
    assert kwargs["torchsim_params"]["force_tol"] == defaults["torchsim_fmax"]
    assert kwargs["torchsim_params"]["max_steps"] == defaults["torchsim_max_steps"]
    assert "torchsim_fmax" not in kwargs
    assert "torchsim_max_steps" not in kwargs


def test_ts_search_params_accepts_seed():
    ts = get_ts_search_params(system_type="gas_cluster", seed=99)
    assert ts["seed"] == 99


def test_ts_search_params_expose_dedupe_and_tolerance_defaults():
    ts = get_ts_search_params(system_type="gas_cluster")

    assert ts.get("dedupe_minima", None) is True
    assert ts.get("minima_energy_tolerance", None) == pytest.approx(
        DEFAULT_ENERGY_TOLERANCE
    )

    kwargs = coerce_ts_params_to_runner_kwargs(ts, system_type="gas_cluster")
    assert kwargs["dedupe_minima"] is True
    assert kwargs["minima_energy_tolerance"] == pytest.approx(DEFAULT_ENERGY_TOLERANCE)
    assert kwargs.get("neb_interpolation_mic") is False
    assert kwargs.get("neb_tangent_method") == DEFAULT_NEB_TANGENT_METHOD
    assert kwargs.get("similarity_pair_cor_max") == pytest.approx(0.1)


def test_ts_search_params_allow_overrides():
    ts = get_ts_search_params(system_type="gas_cluster")
    ts["dedupe_minima"] = False
    ts["minima_energy_tolerance"] = 0.05

    kwargs = coerce_ts_params_to_runner_kwargs(ts, system_type="gas_cluster")
    assert kwargs["dedupe_minima"] is False
    assert kwargs["minima_energy_tolerance"] == pytest.approx(0.05)


def test_ts_search_params_embed_surface_config_for_surface_systems():
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    slab.pbc = [True, True, True]
    cfg = SurfaceSystemConfig(slab=slab, fix_all_slab_atoms=True)
    ts = get_ts_search_params(system_type="surface_cluster", surface_config=cfg)
    assert ts["surface_config"] is cfg
    kwargs = coerce_ts_params_to_runner_kwargs(ts, system_type="surface_cluster")
    assert kwargs.get("surface_config") is cfg


def test_coerce_ts_surface_config_defaults_to_none():
    ts = get_ts_search_params(system_type="gas_cluster")
    kwargs = coerce_ts_params_to_runner_kwargs(ts, system_type="gas_cluster")
    assert kwargs.get("surface_config") is None


def test_coerce_ts_requires_valid_system_type():
    ts = get_ts_search_params(system_type="gas_cluster")
    with pytest.raises(ValueError, match="Unsupported system_type"):
        coerce_ts_params_to_runner_kwargs(ts, system_type="not_a_real_type")


def test_ts_search_surface_regime_mic_and_fmax():
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    slab.pbc = [True, True, True]
    cfg = SurfaceSystemConfig(slab=slab, fix_all_slab_atoms=True)
    ts = get_ts_search_params(system_type="surface_cluster", surface_config=cfg)
    assert ts["neb_interpolation_mic"] is True
    assert ts["neb_n_images"] == 5
    assert ts["neb_fmax"] == pytest.approx(0.1)
    assert ts["torchsim_fmax"] == pytest.approx(0.1)
    assert ts["neb_steps"] == 500
    assert ts["torchsim_max_steps"] == 500
    assert ts["neb_climb"] is False
    assert ts["neb_interpolation_method"] == "idpp"
    assert ts["neb_align_endpoints"] is True
    kwargs = coerce_ts_params_to_runner_kwargs(ts, system_type="surface_cluster")
    assert kwargs["neb_interpolation_mic"] is True
    assert kwargs["neb_n_images"] == 5
    assert kwargs["neb_climb"] is False
    assert kwargs["neb_fmax"] == pytest.approx(0.1)
    assert kwargs["neb_steps"] == 500
    assert kwargs["neb_interpolation_method"] == "idpp"
    assert kwargs["torchsim_params"]["force_tol"] == pytest.approx(0.1)
    assert kwargs["torchsim_params"]["max_steps"] == 500
    assert kwargs["neb_align_endpoints"] is True
    assert kwargs["neb_surface_cell_remap"] is True
    assert kwargs["neb_surface_lattice_rotation"] is True


def test_ts_search_step_defaults_can_be_auto():
    ts = get_ts_search_params(system_type="gas_cluster")

    assert ts.get("neb_steps") == "auto"
    assert ts.get("torchsim_max_steps") == "auto"

    kwargs = coerce_ts_params_to_runner_kwargs(ts, system_type="gas_cluster")
    assert kwargs["neb_steps"] == "auto"
    assert kwargs["torchsim_params"]["max_steps"] == "auto"


def test_default_go_and_default_ts_presets_share_mace_model():
    go_params = get_default_params()
    ts_params = get_ts_search_params(system_type="gas_cluster")

    assert go_params["calculator"] == "MACE"
    assert ts_params["calculator"] == "MACE"
    assert go_params["calculator_kwargs"] == {"model_name": "mace_matpes_0"}
    assert ts_params["calculator_kwargs"] == {"model_name": "mace_matpes_0"}


def test_loaders_default_to_final_unique_minima():
    """Public loaders should default to final_unique_minimum rows only."""
    import inspect

    from scgo.database.helpers import (
        extract_minima_from_database_file,
        extract_transition_states_from_database_file,
        load_previous_run_results,
    )
    from scgo.ts_search.transition_state_io import load_minima_by_composition

    assert (
        inspect.signature(extract_minima_from_database_file)
        .parameters["require_final"]
        .default
        is True
    )
    assert (
        inspect.signature(load_previous_run_results)
        .parameters["prefer_final_unique"]
        .default
        is True
    )
    assert (
        inspect.signature(load_minima_by_composition)
        .parameters["prefer_final_unique"]
        .default
        is True
    )
    assert (
        inspect.signature(extract_transition_states_from_database_file)
        .parameters["require_final_unique_ts"]
        .default
        is True
    )


def _fake_torchsim_go(
    *,
    system_type: str,
    surface_config: SurfaceSystemConfig | None = None,
    seed: int | None = None,
    model_name: str | None = None,
) -> dict:
    from scgo.param_presets import get_default_params

    p = get_default_params()
    for algo in ("simple", "bh", "ga"):
        p["optimizer_params"][algo]["system_type"] = system_type
    if surface_config is not None:
        p["surface_config"] = surface_config
        for algo in ("simple", "bh", "ga"):
            p["optimizer_params"][algo]["surface_config"] = surface_config
    if model_name is not None:
        p["calculator_kwargs"]["model_name"] = model_name
    p["seed"] = seed
    return p


def _build_mace_go_ts_like_runner(
    seed: int,
    *,
    niter: int,
    population_size: int,
    max_pairs: int,
    system_type: str,
    surface_config: SurfaceSystemConfig | None = None,
) -> tuple[dict, dict]:
    go_params = param_presets_module.get_torchsim_ga_params(
        system_type=system_type, seed=seed, surface_config=surface_config
    )
    go_params["calculator"] = "MACE"
    ga = go_params["optimizer_params"]["ga"]
    ga["niter"] = niter
    ga["population_size"] = population_size
    if surface_config is not None:
        go_params["surface_config"] = surface_config
    ts_params = get_ts_search_params(
        system_type=system_type,
        surface_config=surface_config,
    )
    ts_params["max_pairs"] = max_pairs
    return go_params, ts_params


def test_production_style_mace_go_ts_gas(monkeypatch):
    monkeypatch.setattr(
        "scgo.param_presets.get_torchsim_ga_params",
        _fake_torchsim_go,
    )
    go_params, ts_params = _build_mace_go_ts_like_runner(
        7,
        niter=8,
        population_size=18,
        max_pairs=12,
        system_type="gas_cluster",
    )
    ga = go_params["optimizer_params"]["ga"]
    assert ga["niter"] == 8
    assert ga["population_size"] == 18
    assert ts_params["max_pairs"] == 12
    assert "surface_config" not in ts_params
    kw = coerce_ts_params_to_runner_kwargs(ts_params, system_type="gas_cluster")
    assert kw["max_pairs"] == 12


def test_production_style_mace_go_ts_surface_has_surface_config(monkeypatch):
    monkeypatch.setattr(
        "scgo.param_presets.get_torchsim_ga_params",
        _fake_torchsim_go,
    )
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    slab.pbc = [True, True, True]
    cfg = SurfaceSystemConfig(slab=slab, fix_all_slab_atoms=True)
    go_params, ts_params = _build_mace_go_ts_like_runner(
        7,
        niter=8,
        population_size=18,
        max_pairs=12,
        system_type="surface_cluster",
        surface_config=cfg,
    )
    ga = go_params["optimizer_params"]["ga"]
    assert go_params["surface_config"] is cfg
    prepared = prepare_algorithm_kwargs(
        ga,
        {"fitness_strategy": "low_energy"},
        ["Pt"] * 5,
        "ga",
    )
    assert prepared["niter_local_relaxation"] >= 400
    assert ts_params["surface_config"] is cfg
    assert (
        coerce_ts_params_to_runner_kwargs(ts_params, system_type="surface_cluster").get(
            "surface_config"
        )
        is cfg
    )


def test_get_torchsim_ga_params_relaxer_uses_calculator_mace_model_name():
    """TorchSim relaxer must use the same MACE name as ``calculator_kwargs``."""
    pytest.importorskip("torch")
    pytest.importorskip("mace")

    try:
        p = get_torchsim_ga_params(
            system_type="gas_cluster", seed=11, model_name="mace_mp_small"
        )
    except Exception as exc:  # pragma: no cover - environment-dependent torch/mace load
        pytest.skip(f"TorchSim model load unavailable in this env: {exc}")
    assert p["calculator_kwargs"]["model_name"] == "mace_mp_small"
    relaxer = p["optimizer_params"]["ga"]["relaxer"]
    assert relaxer.mace_model_name == "mace_mp_small"


def test_get_torchsim_ga_params_default_relaxer_matches_default_model():
    pytest.importorskip("torch")
    pytest.importorskip("mace")

    try:
        p = get_torchsim_ga_params(system_type="gas_cluster", seed=3)
    except Exception as exc:  # pragma: no cover - environment-dependent torch/mace load
        pytest.skip(f"TorchSim model load unavailable in this env: {exc}")
    assert p["calculator_kwargs"].get("model_name") == "mace_matpes_0"
    assert p["optimizer_params"]["ga"]["relaxer"].mace_model_name == "mace_matpes_0"
