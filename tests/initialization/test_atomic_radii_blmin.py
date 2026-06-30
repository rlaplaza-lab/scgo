"""Tests for unified build_blmin using scgo covalent radii."""

from __future__ import annotations

from scgo.initialization.atomic_radii import (
    build_blmin,
    build_blmin_from_zs,
    get_covalent_radius,
)


def test_build_blmin_symmetric_pairs() -> None:
    blmin = build_blmin(["Pt", "O"], ratio=0.7)
    pt_z, o_z = 78, 8
    expected = (get_covalent_radius("Pt") + get_covalent_radius("O")) * 0.7
    assert abs(blmin[(pt_z, o_z)] - expected) < 1e-9
    assert abs(blmin[(o_z, pt_z)] - expected) < 1e-9


def test_build_blmin_from_zs_matches_symbols() -> None:
    from_zs = build_blmin_from_zs([78, 8], ratio=0.7)
    from_syms = build_blmin(["Pt", "O"], ratio=0.7)
    assert from_zs == from_syms
