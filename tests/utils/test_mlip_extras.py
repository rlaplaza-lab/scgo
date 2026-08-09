"""Regression tests for MLIP extras detection.

Detection used to parse SCGO's own ``Requires-Dist`` metadata, which lists every
*declared* extra regardless of what is installed (a no-op) and can be missing or
unreadable for editable installs. Detection now probes the real distributions
(``mace-torch``, ``fairchem-core``, ``upet``).
"""

from __future__ import annotations

import importlib.metadata
import importlib.util

import pytest

from scgo.exceptions import SCGOConfigurationError
from scgo.utils import mlip_extras

_MODULES = {
    "mace-torch": "mace",
    "fairchem-core": "fairchem.core",
    "upet": "upet",
}


class _FakeDistribution:
    version = "0.0.0"


def _install(monkeypatch, *distributions: str) -> None:
    """Pretend exactly ``distributions`` are installed (dist + module)."""
    present = set(distributions)
    present_modules = {_MODULES[d] for d in present}

    def fake_distribution(name: str):
        if name in present:
            return _FakeDistribution()
        raise importlib.metadata.PackageNotFoundError(name)

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, *args, **kwargs):
        if name in _MODULES.values():
            return object() if name in present_modules else None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.metadata, "distribution", fake_distribution)
    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)


def test_no_stack_installed(monkeypatch):
    _install(monkeypatch)

    assert mlip_extras.installed_mlip_stacks() == (False, False, False)
    assert mlip_extras.ensure_mace_uma_not_both_installed() is None


@pytest.mark.parametrize(
    ("dist", "expected"),
    [
        ("mace-torch", (True, False, False)),
        ("fairchem-core", (False, True, False)),
        ("upet", (False, False, True)),
    ],
)
def test_single_stack_resolves(monkeypatch, dist, expected):
    _install(monkeypatch, dist)

    assert mlip_extras.installed_mlip_stacks() == expected
    # Exactly one stack: no conflict.
    assert mlip_extras.ensure_mace_uma_not_both_installed() is None


def test_both_mace_and_uma_installed_raises(monkeypatch):
    _install(monkeypatch, "mace-torch", "fairchem-core")

    assert mlip_extras.installed_mlip_stacks() == (True, True, False)

    with pytest.raises(SCGOConfigurationError, match="Multiple MLIP stacks"):
        mlip_extras.ensure_mace_uma_not_both_installed()


def test_mace_and_upet_installed_raises(monkeypatch):
    _install(monkeypatch, "mace-torch", "upet")

    with pytest.raises(SCGOConfigurationError, match="Multiple MLIP stacks"):
        mlip_extras.ensure_mace_uma_not_both_installed()


def test_distribution_present_but_module_missing_is_not_a_stack(monkeypatch):
    def fake_distribution(name: str):
        if name == "mace-torch":
            return _FakeDistribution()
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "distribution", fake_distribution)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None)

    assert mlip_extras.installed_mlip_stacks() == (False, False, False)


def test_unreadable_distribution_metadata_does_not_raise(monkeypatch):
    def broken_distribution(name: str):
        raise OSError(f"broken metadata for {name}")

    monkeypatch.setattr(importlib.metadata, "distribution", broken_distribution)

    assert mlip_extras.installed_mlip_stacks() == (False, False, False)
    assert mlip_extras.ensure_mace_uma_not_both_installed() is None
