"""Tests for top-level GO param allowlisting in the runner API."""

from __future__ import annotations

from pathlib import Path

import pytest

from scgo.exceptions import SCGOValidationError
from scgo.param_presets import get_testing_params
from scgo.runner_api import _run_go_trials


def test_validation_n_jobs_accepted_by_param_gate(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("scgo.runner_api.run_trials", lambda **_kwargs: [])
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


def test_unexpected_top_level_param_rejected(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("scgo.runner_api.run_trials", lambda **_kwargs: [])
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
