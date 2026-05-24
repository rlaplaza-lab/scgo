"""Smoke tests that pass without the MACE optional extra (e.g. UMA-only CI)."""

from __future__ import annotations


def test_import_scgo_without_eager_torchsim():
    import scgo

    assert scgo.__version__
    assert hasattr(scgo, "run_go")


def test_ga_go_torchsim_importable_with_mace_extra():
    """``ga_go`` / ``ga_go_torchsim`` are exported from :mod:`scgo.algorithms`."""
    from scgo.algorithms import ga_go, ga_go_torchsim

    assert ga_go is ga_go_torchsim
