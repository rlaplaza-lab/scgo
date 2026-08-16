"""MACE machine learning potential wrapper for cluster optimization.

This module provides a simplified wrapper around the MACE foundation models
loaded through ``mace_mp``, handling device selection and initialization for
seamless integration with global optimization workflows.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum
from typing import Any

import torch

from scgo.utils.mlip_extras import ensure_mace_uma_not_both_installed


@contextmanager
def torch_load_weights_only_false() -> Iterator[None]:
    """Temporarily force ``torch.load(..., weights_only=False)``.

    MACE checkpoints and e3nn ``constants.pt`` pickle full graphs / custom
    globals; PyTorch 2.6+ defaults ``weights_only=True`` and rejects them.
    The patch is restored on exit so the rest of the process keeps the
    safer default. SCGO only loads foundation checkpoints from trusted
    sources (same policy as upstream MACE).
    """
    orig_load = torch.load

    def _load(*args: Any, **kwargs: Any) -> Any:
        kwargs["weights_only"] = False
        return orig_load(*args, **kwargs)

    torch.load = _load  # type: ignore[method-assign]
    try:
        yield
    finally:
        torch.load = orig_load  # type: ignore[method-assign]


# e3nn/o3/_wigner.py unpickles constants.pt via torch.load at import time.
# Patch only for the duration of the mace import.
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

with torch_load_weights_only_false():
    from mace.calculators import mace_mp

from scgo.calculators.torch_device import resolve_torch_device
from scgo.utils.logging import get_logger


class MaceUrls(StrEnum):
    """Checkpoint download URLs for MACE models."""

    mace_mp_small = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b/mace_agnesi_small.model"
    mace_mpa_medium = "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mpa_0/mace-mpa-0-medium.model"
    mace_off_small = "https://github.com/ACEsuit/mace-off/blob/main/mace_off23/MACE-OFF23_small.model?raw=true"
    mace_matpes_0 = "https://github.com/ACEsuit/mace-foundations/releases/download/mace_matpes_0/MACE-matpes-r2scan-omat-ft.model"


class MACE(Calculator):
    """A wrapper for MACE foundation-model calculators for global optimization.

    This class simplifies the initialization of a MACE calculator, handling
    automatic device selection (CUDA/MPS/CPU) and model loading. It serves as a
    standard ASE-compliant calculator, making it easy to integrate MACE into
    global optimization workflows.

    """

    implemented_properties: list[str] = ["energy", "forces"]

    def __init__(
        self,
        model_name: str = "mace_matpes_0",
        device: str | None = None,
        default_dtype: str = "float64",
        **kwargs,
    ):
        """Initializes the MACE calculator.

        Args:
            model_name: The name of the pretrained MACE model to use. Can be:
                - A MaceUrls enum member name (e.g., "mace_matpes_0")
                - A direct URL to a model file
                - A standard mace_mp model name (e.g., "small", "medium", "large")
                Defaults to "mace_matpes_0" (r2scan variant).
            device: The computing device to run the model on. If None, it will
                auto-detect CUDA or MPS (for Apple Silicon) and fall back to CPU
                if neither is available. Defaults to None.
            default_dtype: The default floating-point precision for calculations.
                "float64" is recommended for stable optimizations.
                Defaults to "float64".
            ``**kwargs``: Additional keyword arguments passed to the base ASE
                Calculator class.
        """
        ensure_mace_uma_not_both_installed()
        selected_device = resolve_torch_device(
            device, allow_mps=True, backend_name="MACE"
        )

        # Resolve model name to URL if it's a MaceUrls enum member
        if hasattr(MaceUrls, model_name):
            model_selector = getattr(MaceUrls, model_name)
        else:
            model_selector = model_name

        super().__init__(**kwargs)
        self.model_name = model_name
        # Store the resolved device so downstream helpers (e.g. TorchSim model
        # extraction) can honour an explicit device="cpu" instead of guessing.
        self.device = selected_device

        logger = get_logger(__name__)
        logger.info(
            'Initializing MACE calculator ("%s" model) on device: "%s"',
            model_name,
            selected_device,
        )

        # Patch torch.load only while loading the checkpoint (PyTorch >=2.6).
        with torch_load_weights_only_false():
            self._mace_calc = mace_mp(
                model=model_selector,
                device=selected_device,
                default_dtype=default_dtype,
            )

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ):
        """Performs the MACE calculation by delegating to the wrapped calculator.

        This method is called by ASE algorithms. It sets up the calculation,
        calls the underlying MACE calculator, and stores the results.

        Args:
            atoms: The Atoms object to perform the calculation on. If None, the
                calculator's internal atoms object is used.
            properties: A list of properties to calculate (e.g., ["energy", "forces"]).
                If None, defaults to `self.implemented_properties`.
            system_changes: A list of strings specifying what has changed since
                the last calculation. See ASE documentation for details.
        """
        if properties is None:
            properties = self.implemented_properties

        # Call the base class's calculate method to handle setup
        super().calculate(atoms, properties, system_changes)

        # Delegate the actual computation to the wrapped MACE calculator instance
        self._mace_calc.calculate(
            atoms=self.atoms,
            properties=properties,
            system_changes=system_changes,
        )
        self.results = self._mace_calc.results


def infer_mace_model_name_from_calculator(calculator: Calculator) -> str | None:
    """Return the MACE foundation model name from an ASE calculator, if known."""
    model_name = getattr(calculator, "model_name", None)
    if isinstance(model_name, str) and model_name:
        return model_name
    name = getattr(calculator, "name", "") or ""
    if name.startswith("MACE-"):
        suffix = name.removeprefix("MACE-")
        return suffix or None
    return None


def try_extract_torchsim_model_from_mace_calculator(
    calculator: Calculator,
) -> object | None:
    """Return the raw MACE torch module already loaded on an ASE MACE calculator.

    Returns ``None`` when the calculator exposes no module (the caller then lets
    TorchSim reload the checkpoint); TorchSim wraps the module in ``MaceModel``
    before use.
    """
    mace_calc = getattr(calculator, "_mace_calc", None)
    if mace_calc is None:
        return None
    models = getattr(mace_calc, "models", None)
    if models:
        return models[0]
    return getattr(mace_calc, "model", None)
