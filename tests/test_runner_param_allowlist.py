"""Tests for top-level GO param allowlisting in the runner API."""

from __future__ import annotations

from pathlib import Path

import pytest

from scgo.exceptions import SCGOValidationError
from scgo.param_presets import get_testing_params
from scgo.runner_go import _run_go_trials


def test_validation_n_jobs_accepted_by_param_gate(tmp_path: Path, monkeypatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(
        "scgo.runner_go.run_trials", lambda **kwargs: captured.update(kwargs) or []
    )
    params = get_testing_params()
    params["calculator"] = "EMT"
    params["validation_n_jobs"] = 2
    result = _run_go_trials(
        ["Pt", "Pt", "Pt", "Pt"],
        "gas_cluster",
        params=params,
        seed=0,
        verbosity=0,
        output_dir=tmp_path,
    )
    assert result == []
    # Explicit validation_n_jobs is forwarded verbatim.
    assert captured["validation_n_jobs"] == 2


def test_unexpected_top_level_param_rejected(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("scgo.runner_go.run_trials", lambda **_kwargs: [])
    params = get_testing_params()
    params["calculator"] = "EMT"
    params["not_a_real_param"] = 1
    with pytest.raises(SCGOValidationError, match="Unexpected parameter keys"):
        _run_go_trials(
            ["Pt", "Pt", "Pt", "Pt"],
            "gas_cluster",
            params=params,
            seed=0,
            verbosity=0,
            output_dir=tmp_path,
        )


def test_top_level_n_jobs_cascades_to_ga(tmp_path: Path, monkeypatch) -> None:
    """Top-level ``n_jobs`` fans out to GA population init and offspring."""
    captured: dict = {}
    monkeypatch.setattr(
        "scgo.runner_go.run_trials", lambda **kwargs: captured.update(kwargs) or []
    )
    params = get_testing_params()
    params["calculator"] = "EMT"
    params["optimizer_params"]["ga"]["population_size"] = 4
    params["n_jobs"] = -2
    _run_go_trials(
        ["Pt", "Pt", "Pt", "Pt"],
        "gas_cluster",
        params=params,
        seed=0,
        verbosity=0,
        output_dir=tmp_path,
    )
    ga_kwargs = captured["global_optimizer_kwargs"]
    assert ga_kwargs["n_jobs_population_init"] == -2
    assert ga_kwargs["n_jobs_offspring"] == -2


def test_per_stage_override_takes_precedence(tmp_path: Path, monkeypatch) -> None:
    """An explicit per-stage key overrides the top-level ``n_jobs`` for that stage."""
    captured: dict = {}
    monkeypatch.setattr(
        "scgo.runner_go.run_trials", lambda **kwargs: captured.update(kwargs) or []
    )
    params = get_testing_params()
    params["calculator"] = "EMT"
    params["optimizer_params"]["ga"]["population_size"] = 4
    params["n_jobs"] = -2
    params["optimizer_params"]["ga"]["n_jobs_offspring"] = 1
    _run_go_trials(
        ["Pt", "Pt", "Pt", "Pt"],
        "gas_cluster",
        params=params,
        seed=0,
        verbosity=0,
        output_dir=tmp_path,
    )
    ga_kwargs = captured["global_optimizer_kwargs"]
    assert ga_kwargs["n_jobs_population_init"] == -2  # inherits top-level
    assert ga_kwargs["n_jobs_offspring"] == 1  # explicit override


def test_validation_n_jobs_inherits_top_level_n_jobs(
    tmp_path: Path, monkeypatch
) -> None:
    """``validation_n_jobs`` inherits top-level ``n_jobs`` when not set directly."""
    captured: dict = {}
    monkeypatch.setattr(
        "scgo.runner_go.run_trials", lambda **kwargs: captured.update(kwargs) or []
    )
    params = get_testing_params()
    params["calculator"] = "EMT"
    params["optimizer_params"]["ga"]["population_size"] = 4
    params["n_jobs"] = -2
    _run_go_trials(
        ["Pt", "Pt", "Pt", "Pt"],
        "gas_cluster",
        params=params,
        seed=0,
        verbosity=0,
        output_dir=tmp_path,
    )
    assert captured["validation_n_jobs"] == -2
