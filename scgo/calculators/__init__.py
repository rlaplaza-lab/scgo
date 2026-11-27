"""Energy calculator interfaces and helpers.

This package contains helper modules for various energy calculators:

- MACE: Machine learning potentials based on MACE models (optional ``[mace]`` extra)
- UMA: FAIRChem UMA checkpoints (optional ``[uma]`` extra)
- UPET: Universal PET models via metatomic (optional ``[upet]`` extra)
- TorchSim: GPU-accelerated batch relaxation (requires ``[mace]``, ``[uma]``, or
  ``[upet]``)

Note:
    MACE, UMA, UPET, and TorchSim symbols load lazily so ``import scgo.calculators``
    works with only the core dependencies. Install ``scgo[mace]``, ``scgo[uma]``,
    or ``scgo[upet]`` for the corresponding stack (not more than one per env).
"""

from __future__ import annotations

from scgo.utils.lazy_exports import lazy_submodule_dir, lazy_submodule_getattr

__all__ = [
    "MACE",
    "MaceUrls",
    "UMA",
    "UPET",
    "TorchSimBatchRelaxer",
]

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "MACE": ("scgo.calculators.mace_helpers", "MACE"),
    "MaceUrls": ("scgo.calculators.mace_helpers", "MaceUrls"),
    "UMA": ("scgo.calculators.uma_helpers", "UMA"),
    "UPET": ("scgo.calculators.upet_helpers", "UPET"),
    "TorchSimBatchRelaxer": (
        "scgo.calculators.torchsim_helpers",
        "TorchSimBatchRelaxer",
    ),
}

__getattr__ = lazy_submodule_getattr(globals(), _LAZY_ATTRS, cache=False)
__dir__ = lazy_submodule_dir(__all__)
