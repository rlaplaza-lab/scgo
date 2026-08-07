"""Optional MLIP install extras (MACE vs UMA vs UPET) — detect conflicts."""

from __future__ import annotations

import importlib.util
import os

from scgo.exceptions import SCGORuntimeError
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


def installed_mlip_stacks() -> tuple[bool, bool, bool]:
    """Return (mace_stack_present, uma_stack_present, upet_stack_present)."""
    mace = _import_spec_available("mace")
    uma = _import_spec_available("fairchem.core")
    upet = _import_spec_available("upet")
    return mace, uma, upet


def installed_mace_and_uma() -> tuple[bool, bool]:
    """Return (mace_stack_present, uma_stack_present) using importlib only."""
    mace, uma, _ = installed_mlip_stacks()
    return mace, uma


def ensure_mace_uma_not_both_installed() -> None:
    """Fail if more than one MLIP stack is importable (unsupported mixed env)."""
    mace, uma, upet = installed_mlip_stacks()
    installed: list[str] = []
    if mace:
        installed.append("MACE (scgo[mace])")
    if uma:
        installed.append("UMA/fairchem (scgo[uma])")
    if upet:
        installed.append("UPET (scgo[upet])")
    if len(installed) <= 1:
        return
    msg = (
        f"Multiple MLIP stacks are importable: {', '.join(installed)}. "
        "Install exactly one extra: pip install 'scgo[mace]', 'scgo[uma]', "
        "or 'scgo[upet]' in separate environments to avoid dependency conflicts."
    )
    logger.warning(msg)
    raise SCGORuntimeError(msg)
