"""Tests for UPET calculator wrapper."""

from __future__ import annotations

import pytest

from scgo.exceptions import SCGOConfigurationError


def test_upet_class_is_ase_calculator():
    from ase.calculators.calculator import Calculator

    from scgo.calculators.upet_helpers import UPET

    assert issubclass(UPET, Calculator)


def test_get_calculator_class_upet():
    from scgo.utils.run_helpers import get_calculator_class

    cls = get_calculator_class("UPET")
    assert cls.__name__ == "UPET"


def test_get_default_upet_params():
    pytest.importorskip("upet")
    from scgo.param_presets import get_default_upet_params

    try:
        p = get_default_upet_params()
    except ImportError as exc:
        pytest.skip(f"UPET/TorchSim relaxer could not be built in this env: {exc}")
    assert p["calculator"] == "UPET"
    assert p["calculator_kwargs"]["model_name"] == "pet-mad-s"
    assert p["calculator_kwargs"]["version"] == "1.5.0"


def test_get_ts_search_params_upet_default_torchsim_flags():
    pytest.importorskip("upet")
    from scgo.param_presets import get_ts_search_params

    ts = get_ts_search_params(
        calculator="UPET",
        calculator_kwargs={"model_name": "pet-mad-s", "version": "1.5.0"},
        system_type="gas_cluster",
    )
    assert ts["calculator"] == "UPET"
    assert ts["use_torchsim"] is True
    assert ts["use_parallel_neb"] is True


def test_multiple_mlip_stacks_raises_when_more_than_one_importable():
    from scgo.utils.mlip_extras import installed_mlip_stacks

    mace, uma, upet = installed_mlip_stacks()
    if sum((mace, uma, upet)) < 2:
        pytest.skip("needs at least two MLIP stacks importable")

    from scgo.utils.mlip_extras import ensure_mace_uma_not_both_installed

    with pytest.raises(SCGOConfigurationError, match="Multiple MLIP stacks"):
        ensure_mace_uma_not_both_installed()


def test_parse_upet_model_and_size():
    from scgo.calculators.torchsim_helpers import _parse_upet_model_and_size

    assert _parse_upet_model_and_size("pet-mad-s") == ("pet-mad", "s")
    assert _parse_upet_model_and_size("pet-oam-xl") == ("pet-oam", "xl")


def test_prepare_atoms_for_metatomic_torchsim_zeros_non_pbc_cell():
    from ase import Atoms

    from scgo.calculators.torchsim_helpers import (
        _prepare_atoms_for_metatomic_torchsim,
        _restore_ase_cell_from_reference,
    )

    atoms = Atoms("Pt4", positions=[[0, 0, 0]] * 4, cell=[10, 10, 10], pbc=False)
    prepared = _prepare_atoms_for_metatomic_torchsim(atoms)
    assert prepared.cell.sum() == 0.0
    relaxed = prepared.copy()
    relaxed.positions[0, 0] += 0.1
    _restore_ase_cell_from_reference(relaxed, atoms)
    assert relaxed.cell[0, 0] == 10.0
    assert list(relaxed.pbc) == [False, False, False]


def test_torchsim_batch_relaxer_upet_requires_model_identity():
    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer
    from scgo.exceptions import SCGOValidationError

    with pytest.raises(SCGOValidationError, match="upet_model_name"):
        TorchSimBatchRelaxer(model_kind="upet")


@pytest.mark.requires_upet
def test_upet_calculator_stores_resolved_device(monkeypatch):
    """K6: an explicit ``device="cpu"`` must be readable from the calculator."""
    upet_calculator = pytest.importorskip("upet.calculator")

    from scgo.calculators.upet_helpers import UPET

    class _FakeUPETCalculator:
        implemented_properties = ["energy", "forces"]

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(upet_calculator, "UPETCalculator", _FakeUPETCalculator)

    calc = UPET(model_name="pet-mad-s", device="cpu")
    assert calc.device == "cpu"
    assert calc._inner.kwargs["device"] == "cpu"


@pytest.mark.requires_upet
def test_upet_extractor_uses_the_calculator_device(monkeypatch):
    """K6: the TorchSim extractor must honour the calculator's stored device."""
    from types import ModuleType
    from unittest.mock import MagicMock, patch

    import torch

    from scgo.calculators import upet_helpers as uh

    atomistic = object()
    meta_calc = MagicMock()
    meta_calc.model = atomistic
    # No device on the inner calculator: only ``calculator.device`` can rescue
    # an explicit CPU request on a CUDA-capable box.
    meta_calc.device = None

    inner = MagicMock()
    inner.calculator = meta_calc
    inner.non_conservative = False

    calc = MagicMock()
    calc._inner = inner
    calc.non_conservative = False
    calc.device = "cpu"

    fake_metatomic = MagicMock(return_value="wrapped")
    neighbors_mod = ModuleType("metatomic_torchsim._neighbors")
    neighbors_mod.HAS_NVALCHEMIOPS = True
    root_mod = ModuleType("metatomic_torchsim")
    root_mod.MetatomicModel = fake_metatomic
    root_mod._neighbors = neighbors_mod

    with patch.dict(
        "sys.modules",
        {
            "metatomic_torchsim": root_mod,
            "metatomic_torchsim._neighbors": neighbors_mod,
        },
    ):
        assert uh.try_extract_torchsim_model_from_upet_calculator(calc) == "wrapped"

    assert fake_metatomic.call_args.kwargs["device"] == torch.device("cpu")
