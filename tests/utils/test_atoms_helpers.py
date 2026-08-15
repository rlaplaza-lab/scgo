"""Tests for atoms_helpers module.

Consolidates the single-purpose ``parse_energy_from_xyz_comment`` cases into two
parametrized tests: one for valid inputs (returns the parsed float) and one for
invalid inputs (returns ``None``).
"""

from __future__ import annotations

import pytest

from scgo.utils.atoms_helpers import parse_energy_from_xyz_comment

VALID_CASES = [
    ({"energy": -123.456}, -123.456),
    ({"E": "-123.456"}, -123.456),
    ({"key1": "100.0", "key2": "200.0", "energy": "-50.5"}, -50.5),
    ({"E": "999.123"}, 999.123),
    ({"energy": "0.0"}, 0.0),
    ({"E": "-1.23e-10"}, -1.23e-10),
    ({"energy": 42}, 42.0),
    ({"energy": -100.0}, -100.0),
    ({"E": "-200.5"}, -200.5),
    ({"energy": "300.75"}, 300.75),
    ({"E": 400}, 400.0),
]


INVALID_CASES = [
    {},
    {"energy": "not_a_number"},
    {"energy": None},
    None,
    "not_a_dict",
    [-123.456],
]


@pytest.mark.parametrize("comment,expected", VALID_CASES)
def test_parse_energy_valid(comment, expected):
    """Valid energy comments parse to the expected float."""
    result = parse_energy_from_xyz_comment(comment)
    assert result == pytest.approx(expected)


@pytest.mark.parametrize("comment", INVALID_CASES)
def test_parse_energy_invalid(comment):
    """Malformed / empty / wrong-typed inputs yield ``None``."""
    assert parse_energy_from_xyz_comment(comment) is None
