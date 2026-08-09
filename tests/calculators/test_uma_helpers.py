"""Tests for UMA / fairchem calculator wrapper."""

from __future__ import annotations

import importlib.util

import pytest

from scgo.exceptions import SCGOConfigurationError
from tests.conftest import skip_uma_in_github_actions

skip_uma_in_github_actions(allow_module_level=True)


def test_uma_class_is_ase_calculator():
    from ase.calculators.calculator import Calculator

    from scgo.calculators.uma_helpers import UMA

    assert issubclass(UMA, Calculator)


def test_get_calculator_class_uma():
    from scgo.utils.run_helpers import get_calculator_class

    cls = get_calculator_class("UMA")
    assert cls.__name__ == "UMA"


def test_get_default_uma_params():
    pytest.importorskip("fairchem.core")
    from scgo.param_presets import get_default_uma_params

    try:
        p = get_default_uma_params()
    except (ImportError, NameError, RuntimeError, OSError) as exc:
        pytest.skip(f"FairChem/TorchSim relaxer could not be built in this env: {exc}")
    assert p["calculator"] == "UMA"
    assert p["calculator_kwargs"]["model_name"] == "uma-s-1p2"
    assert p["calculator_kwargs"]["task_name"] == "oc25"


def test_get_ts_search_params_uma_default_torchsim_flags():
    pytest.importorskip("fairchem.core")
    from scgo.param_presets import get_ts_search_params

    ts = get_ts_search_params(
        calculator="UMA",
        calculator_kwargs={"model_name": "uma-s-1p2", "task_name": "oc25"},
        system_type="gas_cluster",
    )
    assert ts["calculator"] == "UMA"
    assert ts["use_torchsim"] is True
    assert ts["use_parallel_neb"] is True


def test_both_mlip_stacks_raises_when_both_importable():
    # ``find_spec("fairchem.core")`` raises ModuleNotFoundError (rather than
    # returning None) when the ``fairchem`` parent package is absent, so probe
    # the parent first.
    if (
        importlib.util.find_spec("mace") is None
        or importlib.util.find_spec("fairchem") is None
        or importlib.util.find_spec("fairchem.core") is None
    ):
        pytest.skip("needs both mace and fairchem.core importable")

    from scgo.utils.mlip_extras import ensure_mace_uma_not_both_installed

    with pytest.raises(SCGOConfigurationError, match="Multiple MLIP stacks"):
        ensure_mace_uma_not_both_installed()


@pytest.mark.requires_uma
def test_uma_calculator_stores_resolved_device(monkeypatch):
    """K6: an explicit ``device="cpu"`` must be readable from the calculator."""
    fairchem_core = pytest.importorskip("fairchem.core")

    from scgo.calculators.uma_helpers import UMA

    class _FakeInner:
        implemented_properties = ["energy", "forces"]

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeFairChemCalculator:
        @staticmethod
        def from_model_checkpoint(model_name, *, task_name=None, device=None):
            return _FakeInner(model_name=model_name, task_name=task_name, device=device)

    monkeypatch.setattr(
        fairchem_core, "FAIRChemCalculator", _FakeFairChemCalculator, raising=False
    )

    calc = UMA(model_name="uma-s-1p2", task_name="oc25", device="cpu")
    assert calc.device == "cpu"
    assert calc._inner.kwargs["device"] == "cpu"


@pytest.mark.requires_uma
def test_fairchem_model_shell_initializes_nn_module():
    """K7: the shell must accept an ``nn.Module`` predictor without raising."""
    from types import SimpleNamespace

    pytest.importorskip("fairchem.core")
    pytest.importorskip("torch_sim.models.fairchem")
    import torch

    from scgo.calculators import uma_helpers as uh

    predictor = torch.nn.Linear(2, 2)
    calc = SimpleNamespace(
        _inner=SimpleNamespace(predictor=predictor, task_name="oc25"),
        task_name="oc25",
    )

    model = uh.try_extract_torchsim_model_from_uma_calculator(calc)

    assert model is not None
    assert model.predictor is predictor
    assert model.task_name == "oc25"
    # ``nn.Module.__init__`` ran, so module bookkeeping exists.
    assert "predictor" in dict(model.named_children())


@pytest.mark.requires_uma
def test_fairchem_model_shell_returns_none_on_broken_predictor():
    """K7: an unusable predictor yields None (TorchSim reloads the checkpoint)."""
    from types import SimpleNamespace

    pytest.importorskip("fairchem.core")
    pytest.importorskip("torch_sim.models.fairchem")

    from scgo.calculators import uma_helpers as uh

    predictor = SimpleNamespace(device="not-a-real-device")
    calc = SimpleNamespace(
        _inner=SimpleNamespace(predictor=predictor, task_name="oc25"),
        task_name="oc25",
    )

    assert uh.try_extract_torchsim_model_from_uma_calculator(calc) is None
