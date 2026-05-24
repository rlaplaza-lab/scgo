"""Optional MLIP install extras (MACE vs UMA) — detect conflicts."""

from __future__ import annotations

import importlib.util
import os

from scgo.utils.logging import get_logger

logger = get_logger(__name__)


def clear_torch_force_no_weights_only_load_env() -> None:
    """Remove env override that triggers e3nn import warnings on MACE load."""
    os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)


def _import_spec_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ModuleNotFoundError, ValueError, ImportError):
        return False


def installed_mace_and_uma() -> tuple[bool, bool]:
    """Return (mace_stack_present, uma_stack_present) using importlib only."""
    mace = _import_spec_available("mace")
    uma = _import_spec_available("fairchem.core")
    return mace, uma


def ensure_mace_uma_not_both_installed() -> None:
    """Fail if both stacks are importable (unsupported mixed environment)."""
    mace, uma = installed_mace_and_uma()
    if mace and uma:
        msg = (
            "Both the MACE stack and fairchem-core are importable. "
            "Prefer a single extra: pip install 'scgo[mace]' or pip install 'scgo[uma]' "
            "in separate environments to avoid dependency conflicts."
        )
        logger.warning(msg)
        raise RuntimeError(msg)
