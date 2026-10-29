"""DFT-D3(BJ) correction shared by the ASE and TorchSim MACE paths.

Coefficients are the simple-dftd3 ``r2scan`` / ``d3.bj`` parameters
(Ehlert et al., J. Chem. Phys. 154, 061101, 2021). Both paths use
TorchSim's ``D3DispersionModel`` (nvalchemiops), not ``mace_mp(dispersion=True)``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

from scgo.exceptions import SCGOValidationError

# MatPES r²SCAN checkpoint. The only MACE model this correction is valid for.
MATPES_R2SCAN_MODEL = "mace_matpes_0"
R2SCAN_DISPERSION_XC = "r2scan"

# simple-dftd3 assets/parameters.toml [parameter.r2scan] d3.bj. s6 is the
# D3(BJ) default of 1.
R2SCAN_D3BJ_A1 = 0.49484001
R2SCAN_D3BJ_A2 = 5.73083694
R2SCAN_D3BJ_S8 = 0.78981345
R2SCAN_D3BJ_S6 = 1.0


class _DispersionUnset:
    """Marker: dispersion follows the MatPES r²SCAN default."""

    def __repr__(self) -> str:
        return "DISPERSION_UNSET"


DISPERSION_UNSET: _DispersionUnset = _DispersionUnset()
# Internal only. Public signatures use ``str | None`` so docs do not link
# this private marker; ``None`` turns the correction off, and omitting the
# argument (the marker) follows the MatPES default.
DispersionXC = str | None | _DispersionUnset


def resolve_mace_dispersion_xc(
    model_name: str,
    dispersion_xc: DispersionXC = DISPERSION_UNSET,
) -> str | None:
    """Return the dispersion functional for an ASE MACE calculator.

    Unset means r²SCAN-D3 on ``mace_matpes_0`` and no correction otherwise.
    ``None`` turns the correction off. Any other combination raises.
    """
    if dispersion_xc is DISPERSION_UNSET:
        if model_name == MATPES_R2SCAN_MODEL:
            return R2SCAN_DISPERSION_XC
        return None
    return _require_r2scan_on_matpes(
        model_name=model_name,
        dispersion_xc=dispersion_xc,
    )


def resolve_relaxer_dispersion_xc(
    *,
    model_kind: str,
    mace_model_name: str,
    dispersion_xc: DispersionXC,
    supplied_model: bool,
) -> str | None:
    """Return the dispersion functional for a TorchSim relaxer.

    A caller-supplied ``model`` is left bare unless ``dispersion_xc`` is set
    explicitly. Loading the default MatPES checkpoint applies r²SCAN-D3.
    """
    if dispersion_xc is DISPERSION_UNSET:
        kind = str(model_kind or "mace").strip().lower()
        if (
            not supplied_model
            and kind == "mace"
            and mace_model_name == MATPES_R2SCAN_MODEL
        ):
            return R2SCAN_DISPERSION_XC
        return None
    kind = str(model_kind or "mace").strip().lower()
    if dispersion_xc is None:
        return None
    if kind != "mace":
        raise SCGOValidationError(
            "dispersion_xc is only supported for MACE "
            f"({MATPES_R2SCAN_MODEL!r}). Got model_kind={model_kind!r}, "
            f"dispersion_xc={dispersion_xc!r}."
        )
    return _require_r2scan_on_matpes(
        model_name=mace_model_name,
        dispersion_xc=dispersion_xc,
    )


def _require_r2scan_on_matpes(
    *,
    model_name: str,
    dispersion_xc: str | None,
) -> str | None:
    if dispersion_xc is None:
        return None
    if dispersion_xc == R2SCAN_DISPERSION_XC and model_name == MATPES_R2SCAN_MODEL:
        return R2SCAN_DISPERSION_XC
    raise SCGOValidationError(
        f"DFT-D3(BJ) dispersion_xc={R2SCAN_DISPERSION_XC!r} applies only to "
        f"MACE model {MATPES_R2SCAN_MODEL!r}. Got model_name={model_name!r}, "
        f"dispersion_xc={dispersion_xc!r}. Pass dispersion_xc=None for other "
        "MACE checkpoints (mace_off already includes dispersion; the PBE "
        "checkpoints are not r²SCAN)."
    )


def dispersion_xc_for_mace_kwargs(
    calculator_kwargs: Mapping[str, Any] | None,
) -> str | None:
    """Resolve dispersion for a MACE ``calculator_kwargs`` dict.

    A missing ``dispersion_xc`` follows the MatPES default. An explicit value
    is checked against ``model_name``.
    """
    ck = calculator_kwargs or {}
    model_name = str(ck.get("model_name", MATPES_R2SCAN_MODEL))
    raw = ck.get("dispersion_xc", DISPERSION_UNSET)
    return resolve_mace_dispersion_xc(model_name, raw)


def drop_mace_dispersion_unless_matpes(
    calculator_kwargs: dict[str, Any],
    model_name: str,
    *,
    explicit: bool,
) -> None:
    """Drop the MatPES default correction when the checkpoint is not MatPES."""
    if explicit or model_name == MATPES_R2SCAN_MODEL:
        return
    calculator_kwargs.pop("dispersion_xc", None)


def _torch_dtype(dtype: object) -> Any:
    """Map a torch dtype or a ``float32`` / ``float64`` name to a dtype."""
    import torch

    if isinstance(dtype, torch.dtype):
        return dtype
    name = str(dtype).removeprefix("torch.")
    mapping = {
        "float64": torch.float64,
        "float32": torch.float32,
    }
    resolved = mapping.get(name)
    if resolved is None:
        raise SCGOValidationError(
            f"Unsupported dispersion dtype {dtype!r}; expected float32 or float64."
        )
    return resolved


def _torch_device(device: object) -> Any:
    import torch

    if isinstance(device, torch.device):
        return device
    return torch.device(str(device))


# Grimme DFT-D3 tables for elements 1-94, in atomic units (Bohr, Hartree·Bohr^6).
# TorchSim converts positions from Å before the kernel runs.
_D3_REFERENCE_NPZ = Path(__file__).parent / "data" / "dftd3_reference.npz"
_d3_parameter_cache: dict[tuple[str, str], Any] = {}


def _r2scan_d3_parameters(*, device: object, dtype: object) -> Any:
    """Load the Grimme DFT-D3 reference tables as ``D3Parameters``."""
    import torch
    from nvalchemiops.torch.interactions.dispersion import D3Parameters

    resolved_device = _torch_device(device)
    resolved_dtype = _torch_dtype(dtype)
    key = (str(resolved_device), str(resolved_dtype))
    cached = _d3_parameter_cache.get(key)
    if cached is not None:
        return cached
    with np.load(_D3_REFERENCE_NPZ) as loaded:
        params = D3Parameters(
            rcov=torch.as_tensor(
                np.array(loaded["rcov"]),
                device=resolved_device,
                dtype=resolved_dtype,
            ),
            r4r2=torch.as_tensor(
                np.array(loaded["r4r2"]),
                device=resolved_device,
                dtype=resolved_dtype,
            ),
            c6ab=torch.as_tensor(
                np.array(loaded["c6ab"]),
                device=resolved_device,
                dtype=resolved_dtype,
            ),
            cn_ref=torch.as_tensor(
                np.array(loaded["cn_ref"]),
                device=resolved_device,
                dtype=resolved_dtype,
            ),
        )
    _d3_parameter_cache[key] = params
    return params


def make_r2scan_d3_model(
    *,
    device: object,
    dtype: object,
    compute_stress: bool = False,
) -> Any:
    """Build a TorchSim DFT-D3(BJ) model with the r²SCAN coefficients.

    ``compute_stress`` defaults to False so a stress-free MLIP result is not
    replaced by a D3-only stress.
    """
    from torch_sim.models.dispersion import D3DispersionModel

    resolved_device = _torch_device(device)
    resolved_dtype = _torch_dtype(dtype)
    return D3DispersionModel(
        a1=R2SCAN_D3BJ_A1,
        a2=R2SCAN_D3BJ_A2,
        s8=R2SCAN_D3BJ_S8,
        s6=R2SCAN_D3BJ_S6,
        d3_params=_r2scan_d3_parameters(device=resolved_device, dtype=resolved_dtype),
        device=resolved_device,
        dtype=resolved_dtype,
        compute_forces=True,
        compute_stress=compute_stress,
    )


def wrap_model_with_r2scan_d3(model: Any) -> Any:
    """Return ``SumModel(model, r2scan_d3)`` at the parent device and dtype."""
    from torch_sim.models.interface import SumModel

    d3 = make_r2scan_d3_model(
        device=model.device,
        dtype=model.dtype,
        compute_stress=False,
    )
    return SumModel(model, d3)


def r2scan_d3_energy_forces(
    atoms: Atoms,
    model: Any,
) -> tuple[float, np.ndarray]:
    """Return D3 energy (eV) and forces (eV/Å) for one structure."""
    import torch
    import torch_sim as ts

    state = ts.initialize_state(atoms, model.device, model.dtype)
    with torch.inference_mode():
        out = model(state)
    energy = float(out["energy"].detach().cpu().reshape(-1)[0])
    forces = np.asarray(out["forces"].detach().cpu().numpy(), dtype=np.float64)
    return energy, forces
