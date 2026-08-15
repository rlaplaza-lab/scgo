"""Tests for top-level GO param allowlisting in the runner API."""

from __future__ import annotations

from pathlib import Path

import pytest

from scgo.exceptions import SCGOValidationError
from scgo.param_presets import get_default_params, get_testing_params
from scgo.runner_go import _EXTRA_ACCEPTED_TOP_LEVEL_KEYS, _run_go_trials
from scgo.utils.run_helpers import _VALID_ALGO_PARAMS, validate_algorithm_params


def test_validate_algorithm_params_accepts_known_keys():
    """T2.1-B: each algorithm accepts its declared key set without raising."""
    for algo, keys in _VALID_ALGO_PARAMS.items():
        validate_algorithm_params(dict.fromkeys(keys), algo)


def test_validate_algorithm_params_rejects_unknown_key():
    """T2.1-B: an unknown key for the chosen algorithm raises."""
    with pytest.raises(SCGOValidationError, match="Unexpected GA algorithm parameters"):
        validate_algorithm_params({"not_a_real_algo_key": 1}, "ga")


def test_accepted_top_level_keys_match_defaults_union():
    """T2.1-A: the allowlist is exactly defaults plus the extra accepted keys.

    Pins the union to the current 24-key set so drift in either ``get_default_params``
    or the extra set is caught.
    """
    expected = {
        "validate_with_hessian",
        "calculator",
        "calculator_kwargs",
        "surface_config",
        "fmax_threshold",
        "check_hessian",
        "imag_freq_threshold",
        "optimizer_params",
        "fitness_strategy",
        "diversity_reference_db",
        "diversity_max_references",
        "diversity_update_interval",
        "tag_final_minima",
        "connectivity_factor",
        "allow_cluster_fragmentation",
        "allow_adsorbate_surface_detachment",
        "enforce_adsorbate_subgraph_integrity",
        "freeze_adsorbate_internal_geometry",
        "adsorbate_definition",
        "adsorbate_fragment_template",
        "cluster_adsorbate_config",
        "n_jobs",
        "validation_n_jobs",
        "seed",
    }
    assert set(get_default_params().keys()) | _EXTRA_ACCEPTED_TOP_LEVEL_KEYS == expected


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
