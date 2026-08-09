"""Optional MLIP install extras (MACE vs UMA vs UPET) — detect conflicts."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
from collections.abc import Iterable

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


def clear_torch_force_no_weights_only_load_env() -> None:
    """Remove env override that triggers e3nn import warnings on MACE load."""
    os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)


def _import_spec_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ModuleNotFoundError, ValueError, ImportError):
        return False


def _installed_scgo_extras() -> set[str]:
    """Return the set of optional-dependency extras actually installed for scgo.

    Detection is based on the installed ``scgo`` distribution's ``Requires-Dist``
    metadata rather than bare ``find_spec`` checks. ``torch_sim`` and other shared
    transitive dependencies (e.g. ``mace`` pulled in via ``torch-sim-atomistic``)
    are importable across suites, so a name-based probe would falsely report
    multiple MLIP stacks as present even in an isolated single-extra environment.
    """
    found: set[str] = set()
    try:
        metadata = importlib.metadata.metadata("scgo")
    except importlib.metadata.PackageNotFoundError:
        return found

    requires: Iterable[str] = []
    raw = metadata.get_all("Requires-Dist")
    if raw:
        requires = raw
    else:
        # Fall back to the functional ``requires()`` API for older metadata
        # layouts whose ``PackageMetadata`` has no usable ``get_all``.
        try:
            requires = importlib.metadata.requires("scgo") or []
        except importlib.metadata.PackageNotFoundError:
            return found

    for line in requires:
        # Lines look like: ``fairchem-core>=2.19.0; extra == "uma"``.
        marker = line.lower().split("extra ==")
        if len(marker) < 2:
            continue
        extra = marker[1].split("]")[0].split(";")[0].strip().strip('"').strip("'")
        found.add(extra)
    return found


def installed_mlip_stacks() -> tuple[bool, bool, bool]:
    """Return (mace_stack_present, uma_stack_present, upet_stack_present).

    A stack counts as present only when its SCGO extra is recorded as installed
    *and* its unique identifying module is importable. This keeps one suite's
    shared transitive dependency from leaking into another suite's detection.
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
