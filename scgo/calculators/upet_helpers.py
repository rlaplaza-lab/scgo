"""UPET (Universal Models for Advanced Atomistic Simulations) calculator."""

from __future__ import annotations

from typing import Any

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from scgo.calculators.torch_device import resolve_torch_device
from scgo.utils.logging import get_logger
from scgo.utils.mlip_extras import ensure_mace_uma_not_both_installed

_MISSING_UPET_MSG = (
    "upet is not installed. Install with: pip install 'scgo[upet]' "
    "(do not combine with the [mace] or [uma] extras in the same environment)."
)


class UPET(Calculator):
    """ASE calculator wrapping UPET checkpoints via ``upet.calculator.UPETCalculator``.

    Parameters mirror common SCGO ``calculator_kwargs`` patterns: ``model_name``
    is a UPET model identifier (e.g. ``\"pet-mad-s\"``); ``version`` selects the
    checkpoint release (default ``\"1.5.0\"``). Device defaults to CUDA when
    available, else CPU.
    """

    def __init__(
        self,
        model_name: str = "pet-mad-s",
        version: str = "1.5.0",
        device: str | None = None,
        checkpoint_path: str | None = None,
        non_conservative: bool = False,
        **kwargs: Any,
    ) -> None:
        ensure_mace_uma_not_both_installed()
        try:
            from upet.calculator import UPETCalculator
        except ImportError as e:
            raise ImportError(_MISSING_UPET_MSG) from e

        dev = resolve_torch_device(device, allow_mps=False, backend_name="UPET")

        if checkpoint_path is not None:
            name = f"UPET-{checkpoint_path}"
        else:
            name = f"UPET-{model_name}-v{version}"
        super().__init__(name=name, **kwargs)

        logger = get_logger(__name__)
        logger.info(
            'Initializing UPET calculator ("%s", version=%s) on device: "%s"',
            model_name,
            version,
            dev,
        )

        self.model_name = model_name
        self.version = version
        self.checkpoint_path = checkpoint_path
        self.non_conservative = non_conservative
        self._inner = UPETCalculator(
            model=model_name if checkpoint_path is None else None,
            version=version,
            device=dev,
            checkpoint_path=checkpoint_path,
            non_conservative=non_conservative,
        )
        self.implemented_properties = list(self._inner.implemented_properties)

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        if properties is None:
            properties = self.implemented_properties
        super().calculate(atoms, properties, system_changes)
        self._inner.calculate(
            atoms=self.atoms,
            properties=properties,
            system_changes=system_changes,
        )
        self.results = self._inner.results


def _unwrap_metatomic_ase_calculator(calc: object) -> object | None:
    """Return the innermost Metatomic ASE calculator, unwrapping symmetrizers."""
    current: object | None = calc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if getattr(current, "model", None) is not None:
            return current
        nested = getattr(current, "calculator", None)
        if nested is None or nested is current:
            break
        current = nested
    return None


def try_extract_torchsim_model_from_upet_calculator(
    calculator: Calculator,
) -> object | None:
    """Reuse the AtomisticModel already loaded on an ASE UPET calculator.

    Returns a TorchSim-ready ``metatomic_torchsim.MetatomicModel`` wrapping the
    live model, or ``None`` if extraction fails (caller falls back to reload).
    """
    try:
        import metatomic_torchsim._neighbors as _mt_neighbors  # type: ignore
        import torch
        from metatomic_torchsim import MetatomicModel  # type: ignore
    except ImportError:
        return None

    # nvalchemiops CUDA NL can fail for non-cubic gas-phase cells; match
    # torchsim_helpers._load_default_upet_model.
    _mt_neighbors.HAS_NVALCHEMIOPS = False

    inner = getattr(calculator, "_inner", None)
    if inner is None:
        inner = calculator

    ase_calc = getattr(inner, "calculator", None)
    if ase_calc is None:
        ase_calc = inner

    meta_calc = _unwrap_metatomic_ase_calculator(ase_calc)
    if meta_calc is None:
        return None

    atomistic = getattr(meta_calc, "model", None)
    if atomistic is None:
        return None

    non_conservative = bool(getattr(calculator, "non_conservative", False))
    if not non_conservative:
        non_conservative = bool(getattr(inner, "non_conservative", False))

    device_raw = getattr(calculator, "device", None)
    if device_raw is None:
        device_raw = getattr(meta_calc, "device", None)
    if device_raw is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device_raw, torch.device):
        device = device_raw
    else:
        device = torch.device(str(device_raw))

    return MetatomicModel(
        atomistic,
        device=device,
        non_conservative=non_conservative,
        compute_stress=False,
    )
