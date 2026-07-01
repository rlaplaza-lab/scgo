"""Smoke tests that example scripts match the current run_go_ts API."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from scgo.surface.config import SurfaceSystemConfig

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
EXAMPLE_SCRIPTS = sorted(EXAMPLES_DIR.glob("example_*.py"))


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


@pytest.mark.parametrize("script_path", EXAMPLE_SCRIPTS, ids=lambda p: p.name)
def test_example_main_calls_run_go_ts_with_current_api(
    monkeypatch: pytest.MonkeyPatch, script_path: Path, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def _fake_run_go_ts(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"ts_results": []}

    def _fake_surface_config(**_kwargs):
        from ase.build import graphene

        return SurfaceSystemConfig(slab=graphene(size=(2, 2, 1), vacuum=8.0))

    module = _load_example_module(script_path)
    monkeypatch.setattr(module, "get_torchsim_ga_params", _fake_go_params)
    monkeypatch.setattr(module, "get_ts_search_params", _fake_ts_params)
    monkeypatch.setattr(module, "run_go_ts", _fake_run_go_ts)
    if hasattr(module, "make_graphite_surface_config"):
        monkeypatch.setattr(
            module, "make_graphite_surface_config", _fake_surface_config
        )
    monkeypatch.setattr(
        module, "DEFAULT_OUTPUT_ROOT", tmp_path / "results", raising=False
    )
    module.main()

    kwargs = captured["kwargs"]
    assert kwargs["go_params"] is not None
    assert kwargs["ts_params"] is not None
    assert kwargs["system_type"] is not None
    assert kwargs.get("verbosity", 1) >= 1
    assert kwargs["seed"] == kwargs["go_params"]["seed"] == kwargs["ts_params"]["seed"]
