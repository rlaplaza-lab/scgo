"""Energy calculator interfaces and helpers.

This package contains helper modules for various energy calculators:

- MACE: Machine learning potentials based on MACE models (optional ``[mace]`` extra)
- UMA: FAIRChem UMA checkpoints (optional ``[uma]`` extra)
- UPET: Universal PET models via metatomic (optional ``[upet]`` extra)
- TorchSim: GPU-accelerated batch relaxation (requires ``[mace]``, ``[uma]``, or ``[upet]``)
- VASP/ORCA export helpers live under :mod:`scgo.calculators.vasp_helpers` and
  :mod:`scgo.calculators.orca_helpers` (not re-exported here).

Note:
    MACE, UMA, and TorchSim symbols load lazily so ``import scgo.calculators``
    works with only the core dependencies. Install ``scgo[mace]``, ``scgo[uma]``,
    or ``scgo[upet]`` for the corresponding stack (not more than one per env).
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "MACE",
    "MaceUrls",
    "UMA",
    "UPET",
    "TorchSimBatchRelaxer",
    "MemoryScalerCache",
    "get_global_memory_scaler_cache",
]


def __getattr__(name: str) -> Any:
    if name == "MACE":
        from .mace_helpers import MACE

        return MACE
    if name == "MaceUrls":
        from .mace_helpers import MaceUrls

        return MaceUrls
    if name == "UMA":
        from .uma_helpers import UMA

        return UMA
    if name == "UPET":
        from .upet_helpers import UPET

        return UPET
    if name in (
        "TorchSimBatchRelaxer",
        "MemoryScalerCache",
        "get_global_memory_scaler_cache",
    ):
        from .torchsim_helpers import (
            MemoryScalerCache,
            TorchSimBatchRelaxer,
            get_global_memory_scaler_cache,
        )

        return {
            "TorchSimBatchRelaxer": TorchSimBatchRelaxer,
            "MemoryScalerCache": MemoryScalerCache,
            "get_global_memory_scaler_cache": get_global_memory_scaler_cache,
        }[name]

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return sorted(__all__)
