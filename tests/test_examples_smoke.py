"""Smoke tests that example scripts match the current run_go / run_go_ts API."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
from ase import Atoms

from scgo.surface.config import SurfaceSystemConfig

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
EXAMPLE_SCRIPTS = sorted(EXAMPLES_DIR.glob("example_*.py"))

_EXPECTED_OUTPUT_STEMS = {
    "example_pt5_gas.py": "pt5_gas",
    "example_pt5_graphite.py": "pt5_graphite",
    "example_pt5_oh_gas.py": "pt5_oh_gas",
    "example_pt5_2oh_graphite.py": "pt5_2oh_graphite",
    "example_defected_graphite.py": "defected_graphite",
    "example_n_doped_graphite.py": "n_doped_graphite_oh",
    "example_pt5_orr_defected_graphite.py": "pt5_orr_defected_graphite",
}

_GO_ONLY_SCRIPTS = frozenset({"example_pt5_orr_defected_graphite.py"})
_GO_TS_SCRIPTS = [p for p in EXAMPLE_SCRIPTS if p.name not in _GO_ONLY_SCRIPTS]
_GO_ONLY_SCRIPT_PATHS = [p for p in EXAMPLE_SCRIPTS if p.name in _GO_ONLY_SCRIPTS]


def test_every_example_script_is_classified() -> None:
    names = {p.name for p in EXAMPLE_SCRIPTS}
    assert names == set(_EXPECTED_OUTPUT_STEMS)
    assert names >= _GO_ONLY_SCRIPTS
    assert {p.name for p in _GO_TS_SCRIPTS} | _GO_ONLY_SCRIPTS == names


_IDENTITY_KEYS = {
    "system_type",
    "surface_config",
    "adsorbate_definition",
    "adsorbate_fragment_template",
    "cluster_adsorbate_config",
}


def _load_example_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def _fake_go_params(**kwargs: Any) -> dict[str, Any]:
    seed = kwargs.get("seed")
    params: dict[str, Any] = {
        "calculator": "MACE",
        "seed": seed,
        "optimizer_params": {"ga": {}},
    }
    if kwargs.get("surface_config") is not None:
        params["surface_config"] = kwargs["surface_config"]
    return params


def _fake_ts_params(**kwargs: Any) -> dict[str, Any]:
    seed = kwargs.get("seed")
    params: dict[str, Any] = {"calculator": "MACE", "seed": seed}
    if kwargs.get("surface_config") is not None:
        params["surface_config"] = kwargs["surface_config"]
    return params


def _fake_surface_config(**_kwargs):
    from ase.build import graphene

    return SurfaceSystemConfig(
        slab=graphene(size=(2, 2, 1), vacuum=8.0),
        fix_all_slab_atoms=False,
        n_relax_top_slab_layers=1,
        name="fake_surface",
    )


def _patch_surface_makers(module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    for maker in (
        "make_graphite_surface_config",
        "make_defected_graphite_surface_config",
        "make_n_doped_graphite_surface_config",
    ):
        if hasattr(module, maker):
            monkeypatch.setattr(module, maker, _fake_surface_config)


def _assert_ga_timing_and_identity(params: dict[str, Any]) -> None:
    ga = params["optimizer_params"]["ga"]
    assert _IDENTITY_KEYS.isdisjoint(ga)
    assert ga["write_timing_json"] is True
    assert ga["detailed_timing"] is True


def _fragment_symbols(adsorbates: Atoms | list[Atoms]) -> tuple[str, ...]:
    frag = adsorbates[0] if isinstance(adsorbates, list) else adsorbates
    assert isinstance(frag, Atoms)
    return tuple(frag.get_chemical_symbols())


@pytest.mark.parametrize("script_path", _GO_TS_SCRIPTS, ids=lambda p: p.name)
def test_example_main_calls_run_go_ts_with_current_api(
    monkeypatch: pytest.MonkeyPatch, script_path: Path, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def _fake_run_go_ts(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"ts_results": []}

    module = _load_example_module(script_path)
    monkeypatch.setattr(module, "get_low_effort_torchsim_ga_params", _fake_go_params)
    monkeypatch.setattr(module, "get_low_effort_ts_search_params", _fake_ts_params)
    monkeypatch.setattr(module, "run_go_ts", _fake_run_go_ts)
    _patch_surface_makers(module, monkeypatch)
    monkeypatch.setattr(
        module, "DEFAULT_OUTPUT_ROOT", tmp_path / "results", raising=False
    )
    module.main()

    kwargs = captured["kwargs"]
    assert kwargs["go_params"] is not None
    assert kwargs["ts_params"] is not None
    assert kwargs["system_type"] is not None
    assert "system_type" not in kwargs["go_params"]
    assert "system_type" not in kwargs["ts_params"]
    assert kwargs.get("verbosity", 1) >= 1
    assert kwargs["seed"] == kwargs["go_params"]["seed"] == kwargs["ts_params"]["seed"]
    assert kwargs["output_root"] == tmp_path / "results"
    assert kwargs["output_stem"] == _EXPECTED_OUTPUT_STEMS[script_path.name]
    _assert_ga_timing_and_identity(kwargs["go_params"])
    assert kwargs["ts_params"]["write_timing_json"] is True
    if kwargs.get("surface_config") is not None:
        # Surface examples may stamp top-level surface_config via the preset;
        # it must agree with the run argument when both are set.
        go_sc = kwargs["go_params"].get("surface_config")
        if go_sc is not None:
            assert go_sc is kwargs["surface_config"]
        ts_sc = kwargs["ts_params"].get("surface_config")
        if ts_sc is not None:
            assert ts_sc is kwargs["surface_config"]


@pytest.mark.parametrize("script_path", _GO_ONLY_SCRIPT_PATHS, ids=lambda p: p.name)
def test_example_main_calls_run_go_with_current_api(
    monkeypatch: pytest.MonkeyPatch, script_path: Path, tmp_path: Path
) -> None:
    captured_calls: list[dict[str, Any]] = []

    def _fake_run_go(*args, **kwargs):
        captured_calls.append({"args": args, "kwargs": kwargs})
        return []

    module = _load_example_module(script_path)
    monkeypatch.setattr(module, "get_low_effort_torchsim_ga_params", _fake_go_params)
    monkeypatch.setattr(module, "run_go", _fake_run_go)
    _patch_surface_makers(module, monkeypatch)
    monkeypatch.setattr(
        module, "DEFAULT_OUTPUT_ROOT", tmp_path / "results", raising=False
    )
    module.main()

    assert len(captured_calls) == 4
    stem = _EXPECTED_OUTPUT_STEMS[script_path.name]
    campaign_root = tmp_path / "results" / f"{stem}_mace"
    fragment_symbols: list[tuple[str, ...] | None] = []
    for call in captured_calls:
        kwargs = call["kwargs"]
        assert kwargs["params"] is not None
        assert "system_type" not in kwargs["params"]
        assert kwargs.get("verbosity", 1) >= 1
        assert kwargs["seed"] == kwargs["params"]["seed"]
        assert kwargs["params"]["connectivity_factor"] == 1.8
        assert kwargs["params"]["n_jobs"] == -2
        _assert_ga_timing_and_identity(kwargs["params"])
        go_sc = kwargs["params"].get("surface_config")
        if go_sc is not None:
            assert go_sc is kwargs["surface_config"]
        assert kwargs["surface_config"] is not None
        output_dir = Path(kwargs["output_dir"])
        assert output_dir.parent == campaign_root
        assert output_dir.name.endswith("_searches")
        ads = kwargs.get("adsorbates")
        if ads is None:
            assert kwargs["system_type"] == "surface_cluster"
            assert (
                kwargs["params"].get("freeze_adsorbate_internal_geometry") is not True
            )
            fragment_symbols.append(None)
        else:
            assert kwargs["system_type"] == "surface_cluster_adsorbate"
            assert kwargs["params"]["freeze_adsorbate_internal_geometry"] is True
            fragment_symbols.append(_fragment_symbols(ads))

    assert fragment_symbols == [None, ("O",), ("O", "H"), ("O", "O", "H")]


@pytest.mark.parametrize("script_path", EXAMPLE_SCRIPTS, ids=lambda p: p.name)
def test_example_uses_low_effort_presets(script_path: Path) -> None:
    """Examples must build params from the shared low-effort presets.

    The Kaggle GPU matrix in
    ``tests/integration/test_gpu_examples_integration.py`` consumes the same
    GO+TS preset pair; importing them here is what keeps those scripts in sync.
    An example that falls back to ``get_torchsim_ga_params`` /
    ``get_ts_search_params`` would silently run at full production budget.
    GO-only examples use only the GA preset.
    """
    source = script_path.read_text(encoding="utf-8")
    assert "get_low_effort_torchsim_ga_params(" in source
    if script_path.name in _GO_ONLY_SCRIPTS:
        assert "get_low_effort_ts_search_params(" not in source
        assert "run_go_ts(" not in source
        assert "run_go(" in source
        return
    assert "get_low_effort_ts_search_params(" in source
