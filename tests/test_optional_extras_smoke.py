"""Smoke tests that pass without the MACE optional extra (e.g. UMA-only CI)."""

from __future__ import annotations

import re


def test_import_scgo_without_eager_torchsim():
    import scgo

    assert re.fullmatch(r"\d+\.\d+\.\d+", str(scgo.__version__)), (
        f"scgo.__version__ {scgo.__version__!r} is not a semver string"
    )
    assert hasattr(scgo, "run_go")
    assert callable(scgo.run_go)


def test_ga_go_importable_with_mace_extra():
    """``ga_go`` is exported from :mod:`scgo.algorithms`."""
    from scgo.algorithms import ga_go

    assert callable(ga_go)
