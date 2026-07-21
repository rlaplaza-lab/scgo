"""UPET (Universal Models for Advanced Atomistic Simulations) calculator."""

from __future__ import annotations

from typing import Any

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

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
            import torch
            from upet.calculator import UPETCalculator
        except ImportError as e:
            raise ImportError(_MISSING_UPET_MSG) from e

        if device is None:
            dev: str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            d = str(device).lower()
            dev = "cuda" if "cuda" in d else "cpu"

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
