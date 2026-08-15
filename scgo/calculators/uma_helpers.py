"""UMA (Universal Models for Atoms) calculator via fairchem-core."""

from __future__ import annotations

from typing import Any

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from scgo.calculators.torch_device import resolve_torch_device
from scgo.exceptions import SCGONotImplementedError
from scgo.utils.logging import get_logger
from scgo.utils.mlip_extras import ensure_mace_uma_not_both_installed

logger = get_logger(__name__)

_MISSING_FAIRCHEM_MSG = (
    "fairchem-core is not installed. Install with: pip install 'scgo[uma]' "
    "(do not combine with the [mace] extra in the same environment)."
)


class UMA(Calculator):
    r"""ASE calculator wrapping FAIRChem UMA checkpoints (fairchem-core).

    Parameters mirror common SCGO ``calculator_kwargs`` patterns: ``model_name``
    is a fairchem pretrained name or path; ``task_name`` selects the UMA task
    (e.g. ``\"omat\"``, ``\"oc20\"``). Device defaults to CUDA when available,
    else CPU (fairchem expects ``\"cuda\"`` or ``\"cpu\"``).
    """

    def __init__(
        self,
        model_name: str = "uma-s-1p2",
        task_name: str | None = "oc25",
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        ensure_mace_uma_not_both_installed()
        try:
            from fairchem.core import FAIRChemCalculator
        except ImportError as e:
            raise SCGONotImplementedError(_MISSING_FAIRCHEM_MSG) from e

        dev = resolve_torch_device(device, allow_mps=False, backend_name="UMA")

        super().__init__(**kwargs)

        logger.info(
            'Initializing UMA calculator ("%s") on device: "%s"', model_name, dev
        )

        self.model_name = model_name
        self.task_name = task_name
        # Store the resolved device so downstream helpers (e.g. TorchSim model
        # extraction) can honour an explicit device="cpu" instead of guessing.
        self.device = dev
        self._inner = FAIRChemCalculator.from_model_checkpoint(
            model_name,
            task_name=task_name,
            device=dev,
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


def infer_uma_model_name_from_calculator(calculator: Calculator) -> str | None:
    """Return the UMA/FairChem model name from an ASE calculator, if known."""
    model_name = getattr(calculator, "model_name", None)
    if isinstance(model_name, str) and model_name:
        return model_name
    name = getattr(calculator, "name", "") or ""
    if name.startswith("UMA-"):
        suffix = name.removeprefix("UMA-")
        return suffix or None
    return None


def try_extract_torchsim_model_from_uma_calculator(
    calculator: Calculator,
) -> object | None:
    """Reuse the FairChem predictor already loaded on an ASE UMA calculator.

    ``torch_sim.models.fairchem.FairChemModel`` only accepts a checkpoint name
    or path in its public constructor, so this builds a TorchSim-ready shell
    around the live ``predictor`` instead of reloading weights.

    Returns ``None`` when the shell cannot be built (the caller then lets
    TorchSim reload the checkpoint).
    """
    try:
        import torch
        from torch_sim.models.fairchem import FairChemModel  # type: ignore
    except ImportError:
        return None

    inner = getattr(calculator, "_inner", None)
    if inner is None:
        return None

    predictor = getattr(inner, "predictor", None)
    if predictor is None:
        return None

    task_name = getattr(calculator, "task_name", None)
    if task_name is None:
        task_name = getattr(inner, "task_name", None)

    try:
        device = getattr(predictor, "device", None)
        device = torch.device(
            resolve_torch_device(device, allow_mps=False, backend_name="UMA")
        )

        model = FairChemModel.__new__(FairChemModel)
        # ``__new__`` skips ``nn.Module.__init__``, so the module bookkeeping
        # dicts are missing and assigning ``predictor`` (an ``nn.Module``)
        # raises "cannot assign module before Module.__init__() call".
        torch.nn.Module.__init__(model)
        model._dtype = torch.float32
        model._compute_stress = False
        model._compute_forces = True
        model._memory_scales_with = "n_atoms"
        model._device = device
        model.predictor = predictor
        model.task_name = task_name
        model.implemented_properties = ["energy", "forces"]
    except (AttributeError, TypeError, RuntimeError) as exc:
        logger.debug(
            "Could not build a TorchSim FairChemModel shell from the live UMA "
            "calculator (%s); TorchSim will reload the checkpoint",
            exc,
        )
        return None
    return model
