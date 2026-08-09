"""Regression tests for composition-string parsing validation.

The comma-separated branch of :func:`parse_composition_arg` used to title-case
each token and return it without ever checking that the result is a real
element, so typos such as ``"Qq,Pt"`` silently produced bogus compositions.
"""

from __future__ import annotations

import pytest
from ase.data import atomic_numbers

from scgo.exceptions import SCGOValidationError
from scgo.runner_composition import parse_composition_arg


@pytest.mark.parametrize(
    "comp_str",
    [
        "Qq,Pt",
        "Pt,Qq",
        "Xx,Yy",
        "Pt,H2",
        "pt,zz",
    ],
)
def test_comma_composition_rejects_unknown_element(comp_str):
    with pytest.raises(SCGOValidationError):
        parse_composition_arg(comp_str)


@pytest.mark.parametrize(
    "comp_str,expected",
    [
        ("Pt,Au", ["Pt", "Au"]),
        ("Pt,Pt,Pt,Au", ["Pt", "Pt", "Pt", "Au"]),
        ("pt,AU", ["Pt", "Au"]),
        ("H,O,O", ["H", "O", "O"]),
        (" Pt , Au ", ["Pt", "Au"]),
    ],
)
def test_comma_composition_accepts_known_elements(comp_str, expected):
    assert parse_composition_arg(comp_str) == expected


def test_comma_composition_result_is_always_a_real_element():
    parsed = parse_composition_arg("Pt,Au,Ru,W")
    assert all(symbol in atomic_numbers for symbol in parsed)


def test_compact_formula_path_still_works():
    assert parse_composition_arg("Pt3Au") == ["Pt", "Pt", "Pt", "Au"]
