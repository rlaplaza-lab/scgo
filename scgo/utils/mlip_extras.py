"""Optional MLIP install extras (MACE vs UMA vs UPET) — detect conflicts."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os

from scgo.exceptions import SCGOConfigurationError
from scgo.utils.logging import get_logger

logger = get_logger(__name__)

# SCGO MLIP extras and the importable module that *uniquely* identifies each
# extra's stack. ``torch_sim`` is shared by all three extras, so it must not be
# used for discrimination — only these names are unique to a single extra.
_MLIP_EXTRA_MODULES: dict[str, str] = {
    "mace": "mace",
    "uma": "fairchem.core",
    "upet": "upet",
}

# Distributions that uniquely identify each MLIP stack. Probing these directly
# is what tells us whether a stack is *really* installed in this environment.
_MLIP_EXTRA_DISTRIBUTIONS: dict[str, tuple[str, ...]] = {
    "mace": ("mace-torch",),
    "uma": ("fairchem-core",),
    "upet": ("upet",),
}


def clear_torch_force_no_weights_only_load_env() -> None:
    """Remove env override that triggers e3nn import warnings on MACE load."""
    os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)


def _import_spec_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ModuleNotFoundError, ValueError, ImportError):
        return False


def _distribution_available(name: str) -> bool:
    """Return True when ``name`` is an installed distribution in this env."""
    try:
        importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return False
    except (OSError, ValueError) as exc:
        # Editable / partially written metadata should never break detection.
        logger.debug("Could not read distribution metadata for %s: %s", name, exc)
        return False
    return True


def _installed_scgo_extras() -> set[str]:
    """Return the set of MLIP extras whose real stack is installed.

    Detection probes the *actual* MLIP distributions (``mace-torch``,
    ``fairchem-core``, ``upet``) instead of parsing SCGO's own
    ``Requires-Dist`` metadata. The metadata route lists every declared extra
    regardless of what is installed (making detection a no-op) and can be
    missing or unreadable for editable installs.
    """
    found: set[str] = set()
    for extra, distributions in _MLIP_EXTRA_DISTRIBUTIONS.items():
        if any(_distribution_available(dist) for dist in distributions):
            found.add(extra)
    return found


def installed_mlip_stacks() -> tuple[bool, bool, bool]:
    """Return (mace_stack_present, uma_stack_present, upet_stack_present).

    A stack counts as present only when its identifying distribution is
    installed *and* its unique identifying module is importable.
    """
    installed_extras = _installed_scgo_extras()
    mace = "mace" in installed_extras and _import_spec_available(
        _MLIP_EXTRA_MODULES["mace"]
    )
    uma = "uma" in installed_extras and _import_spec_available(
        _MLIP_EXTRA_MODULES["uma"]
    )
    upet = "upet" in installed_extras and _import_spec_available(
        _MLIP_EXTRA_MODULES["upet"]
    )
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
    raise SCGOConfigurationError(msg)
