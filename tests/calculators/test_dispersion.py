"""DFT-D3(BJ) wiring for the MatPES r²SCAN MACE default."""

from __future__ import annotations

import pytest

from scgo.calculators.dispersion import (
    DISPERSION_UNSET,
    MATPES_R2SCAN_MODEL,
    R2SCAN_D3BJ_A1,
    R2SCAN_D3BJ_A2,
    R2SCAN_D3BJ_S6,
    R2SCAN_D3BJ_S8,
    R2SCAN_DISPERSION_XC,
    resolve_mace_dispersion_xc,
    resolve_relaxer_dispersion_xc,
)
from scgo.exceptions import SCGOValidationError


def test_matpes_default_dispersion_is_r2scan():
    assert resolve_mace_dispersion_xc(MATPES_R2SCAN_MODEL) == R2SCAN_DISPERSION_XC


def test_other_mace_models_default_to_no_dispersion():
    assert resolve_mace_dispersion_xc("mace_mp_small") is None
    assert resolve_mace_dispersion_xc("mace_off_small") is None


def test_explicit_none_disables_matpes_dispersion():
    assert resolve_mace_dispersion_xc(MATPES_R2SCAN_MODEL, None) is None


def test_r2scan_dispersion_rejected_for_other_mace_models():
    with pytest.raises(SCGOValidationError, match="mace_matpes_0"):
        resolve_mace_dispersion_xc("mace_mp_small", R2SCAN_DISPERSION_XC)


def test_supplied_torchsim_model_is_not_wrapped_unless_asked():
    assert (
        resolve_relaxer_dispersion_xc(
            model_kind="mace",
            mace_model_name=MATPES_R2SCAN_MODEL,
            dispersion_xc=DISPERSION_UNSET,
            supplied_model=True,
        )
        is None
    )


def test_relaxer_default_loads_r2scan_only_for_matpes():
    assert (
        resolve_relaxer_dispersion_xc(
            model_kind="mace",
            mace_model_name=MATPES_R2SCAN_MODEL,
            dispersion_xc=DISPERSION_UNSET,
            supplied_model=False,
        )
        == R2SCAN_DISPERSION_XC
    )
    assert (
        resolve_relaxer_dispersion_xc(
            model_kind="mace",
            mace_model_name="mace_mp_small",
            dispersion_xc=DISPERSION_UNSET,
            supplied_model=False,
        )
        is None
    )
    assert (
        resolve_relaxer_dispersion_xc(
            model_kind="fairchem",
            mace_model_name=MATPES_R2SCAN_MODEL,
            dispersion_xc=DISPERSION_UNSET,
            supplied_model=False,
        )
        is None
    )


def test_ts_runner_copies_default_mace_dispersion():
    from scgo.param_presets import get_ts_search_params
    from scgo.utils.ts_runner_kwargs import coerce_ts_params_to_runner_kwargs

    ts = get_ts_search_params(system_type="gas_cluster")
    kw = coerce_ts_params_to_runner_kwargs(ts, system_type="gas_cluster")
    assert kw["torchsim_params"]["dispersion_xc"] == R2SCAN_DISPERSION_XC


def test_uma_task_rejects_wrong_fine_tune_task():
    from scgo.calculators.uma_helpers import validate_uma_task

    with pytest.raises(SCGOValidationError, match="omol"):
        validate_uma_task("uma-s-1p2p1-ft-d4rev", "oc25")
    with pytest.raises(SCGOValidationError, match="omat"):
        validate_uma_task("uma-s-1p2p1-ft-r2scan", "omol")
    validate_uma_task("uma-s-1p2p1", "oc25")


def test_r2scan_fine_tune_warns_about_missing_d3(caplog):
    import logging

    from scgo.calculators.uma_helpers import validate_uma_task

    with caplog.at_level(logging.WARNING):
        validate_uma_task("uma-s-1p2p1-ft-r2scan", "omat")
    assert "does not attach DFT-D3" in caplog.text


@pytest.mark.requires_mace
def test_r2scan_d3_model_uses_simple_dftd3_coefficients():
    from scgo.calculators.dispersion import make_r2scan_d3_model

    model = make_r2scan_d3_model(device="cpu", dtype="float64", compute_stress=False)
    assert model.a1 == pytest.approx(R2SCAN_D3BJ_A1)
    assert model.a2 == pytest.approx(R2SCAN_D3BJ_A2)
    assert model.s8 == pytest.approx(R2SCAN_D3BJ_S8)
    assert model.s6 == pytest.approx(R2SCAN_D3BJ_S6)
    assert model.compute_stress is False
    assert model.compute_forces is True


def test_dftd3_reference_tables_match_grimme_radii():
    import numpy as np

    from scgo.calculators.dispersion import _D3_REFERENCE_NPZ

    with np.load(_D3_REFERENCE_NPZ) as loaded:
        rcov = np.array(loaded["rcov"])
        c6ab = np.array(loaded["c6ab"])
    assert rcov.shape == (95,)
    assert c6ab.shape == (95, 95, 5, 5)
    assert rcov[1] == pytest.approx(0.80628308, rel=1e-5)
    assert c6ab[18, 18].max() == pytest.approx(64.6462, rel=1e-4)


@pytest.mark.requires_mace
def test_r2scan_d3_attracts_a_separated_pair():
    from ase import Atoms

    from scgo.calculators.dispersion import (
        make_r2scan_d3_model,
        r2scan_d3_energy_forces,
    )

    atoms = Atoms("Ar2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 4.0]], pbc=False)
    model = make_r2scan_d3_model(device="cpu", dtype="float64", compute_stress=False)
    energy, forces = r2scan_d3_energy_forces(atoms, model)
    assert energy == pytest.approx(-0.00400789, rel=1e-4)
    assert forces.shape == (2, 3)
    assert forces[0, :2] == pytest.approx(0.0)
    assert forces[0, 2] == pytest.approx(0.00186845, rel=1e-4)
    assert forces[1] == pytest.approx(-forces[0])


@pytest.mark.requires_mace
def test_non_matpes_relaxer_rejects_r2scan_before_loading_weights():
    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    with pytest.raises(SCGOValidationError, match="mace_matpes_0"):
        TorchSimBatchRelaxer(
            device="cpu",
            mace_model_name="mace_mp_small",
            dispersion_xc=R2SCAN_DISPERSION_XC,
        )


@pytest.mark.requires_mace
def test_explicit_r2scan_wraps_a_caller_supplied_model():
    import torch
    from torch_sim.models.interface import ModelInterface, SumModel

    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    class _Tiny(ModelInterface):
        def __init__(self) -> None:
            super().__init__()
            self._device = torch.device("cpu")
            self._dtype = torch.float32
            self._compute_stress = False
            self._compute_forces = True

        def forward(self, state, **kwargs):  # noqa: ANN001
            del state, kwargs
            return {}

    wrapped = TorchSimBatchRelaxer(
        device="cpu",
        model=_Tiny(),
        mace_model_name=MATPES_R2SCAN_MODEL,
        dispersion_xc=R2SCAN_DISPERSION_XC,
        max_steps=1,
    )
    assert isinstance(wrapped.model, SumModel)
    assert wrapped.dispersion_xc == R2SCAN_DISPERSION_XC
