"""Tests for parse_composition_arg: formats, capitalization rules, and rejections."""

from __future__ import annotations

import pytest

from scgo import parse_composition_arg
from scgo.exceptions import SCGOValidationError


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        ("Pt,Pt,Au", ["Pt", "Pt", "Au"]),
        ("pt, pt ,Au", ["Pt", "Pt", "Au"]),
        ("Pt3Au", ["Pt", "Pt", "Pt", "Au"]),
        ("Pt10", ["Pt"] * 10),
        ("pt3", ["Pt"] * 3),
        ("pt", ["Pt"]),
        ("fe10", ["Fe"] * 10),
        ("na", ["Na"]),
        ("HO2", ["H", "O", "O"]),
        ("H2O", ["H", "H", "O"]),
        ("HO2Ru9W2", ["H", "O", "O"] + ["Ru"] * 9 + ["W", "W"]),
        ("CO2", ["C", "O", "O"]),
        ("SnO2", ["Sn", "O", "O"]),
        (" Ru9W2O ", ["Ru"] * 9 + ["W", "W", "O"]),
        ("Cu", ["Cu"]),
    ],
)
def test_parse_composition_arg_valid(formula: str, expected: list[str]) -> None:
    assert parse_composition_arg(formula) == expected


@pytest.mark.parametrize(
    "formula",
    [
        "",
        "pt3au",
        "ho2",
        "ho2ru9w2",
        "cu",
        "co2",
        "sn",
        "xyz",
        "Pt0",
        "pt0",
        "pt0au",
        "AuPt0",
        "Pt3au",
    ],
)
def test_parse_composition_arg_rejects_invalid(formula: str) -> None:
    with pytest.raises(SCGOValidationError, match="Unable to parse composition string"):
        parse_composition_arg(formula)


@pytest.mark.parametrize(
    ("formula", "message_fragment"),
    [
        ("ho2", "Ambiguous"),
        ("cu", "Ambiguous"),
        ("co2", "Ambiguous"),
        ("sn", "Ambiguous"),
        ("pt3au", "multiple elements"),
        ("ho2ru9w2", "multiple elements"),
        ("", "Empty composition"),
        ("Pt0", "Zero atom count"),
    ],
)
def test_parse_composition_arg_error_detail(
    formula: str, message_fragment: str
) -> None:
    with pytest.raises(SCGOValidationError, match=message_fragment):
        parse_composition_arg(formula)
