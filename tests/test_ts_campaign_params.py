"""Regression tests for ``params`` handling in transition-state campaigns.

``run_transition_state_campaign`` forwards ``params=params`` **and**
``**ts_kwargs`` to ``run_transition_state_search``. Because
``coerce_ts_params_to_runner_kwargs`` emits a ``"params"`` entry (the calculator
name and kwargs), the public ``run_ts_campaign`` used to raise
``TypeError: ... got multiple values for keyword argument 'params'``.
"""

from __future__ import annotations

from typing import Any

import pytest

from scgo.param_presets import get_ts_search_params
from scgo.runner_api import run_ts_campaign
from scgo.ts_search.transition_state_run import run_transition_state_campaign

SEARCH_TARGET = "scgo.ts_search.transition_state_run.run_transition_state_search"


@pytest.fixture
def captured_search(monkeypatch):
    """Stub ``run_transition_state_search`` and record its keyword arguments."""
    calls: list[dict[str, Any]] = []

    def _fake_search(composition, **kwargs):
        calls.append({"composition": list(composition), **kwargs})
        return []

    monkeypatch.setattr(SEARCH_TARGET, _fake_search)
    return calls


def _emt_ts_params() -> dict[str, Any]:
    return {
        **get_ts_search_params(
            system_type="gas_cluster",
            calculator="EMT",
            calculator_kwargs={},
        ),
        "use_torchsim": False,
        "use_parallel_neb": False,
    }


def test_campaign_accepts_params_inside_ts_kwargs(captured_search):
    """``ts_kwargs['params']`` must not collide with the explicit argument."""
    results = run_transition_state_campaign(
        [["Pt", "Pt", "Pt"]],
        "gas_cluster",
        verbosity=0,
        ts_kwargs={"params": {"calculator": "EMT", "calculator_kwargs": {}}},
    )

    assert results == {"Pt3": []}
    assert len(captured_search) == 1
    assert captured_search[0]["params"] == {
        "calculator": "EMT",
        "calculator_kwargs": {},
    }


def test_campaign_explicit_params_wins_over_ts_kwargs(captured_search):
    run_transition_state_campaign(
        [["Pt", "Pt", "Pt"]],
        "gas_cluster",
        params={"calculator": "EMT", "calculator_kwargs": {}},
        verbosity=0,
        ts_kwargs={"params": {"calculator": "MACE"}},
    )

    assert captured_search[0]["params"]["calculator"] == "EMT"


def test_campaign_accepts_system_type_inside_ts_kwargs(captured_search):
    run_transition_state_campaign(
        [["Pt", "Pt", "Pt"]],
        "gas_cluster",
        verbosity=0,
        ts_kwargs={"system_type": "gas_cluster"},
    )

    assert captured_search[0]["system_type"] == "gas_cluster"


def test_campaign_does_not_mutate_caller_ts_kwargs(captured_search):
    ts_kwargs = {"params": {"calculator": "EMT", "calculator_kwargs": {}}}

    run_transition_state_campaign(
        [["Pt", "Pt", "Pt"]],
        "gas_cluster",
        verbosity=0,
        ts_kwargs=ts_kwargs,
    )

    assert ts_kwargs == {"params": {"calculator": "EMT", "calculator_kwargs": {}}}


def test_run_ts_campaign_forwards_calculator_params(captured_search, tmp_path):
    """The public campaign entry point must run and keep the calculator config."""
    results = run_ts_campaign(
        [["Pt", "Pt", "Pt"]],
        ts_params=_emt_ts_params(),
        output_dir=tmp_path,
        verbosity=0,
        system_type="gas_cluster",
    )

    assert results == {"Pt3": []}
    assert captured_search[0]["params"]["calculator"] == "EMT"
    assert captured_search[0]["system_type"] == "gas_cluster"
