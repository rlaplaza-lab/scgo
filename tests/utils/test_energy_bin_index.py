"""Tests for shared energy-bin index helper."""

import pytest

from scgo.utils.helpers import energy_bin_index


@pytest.mark.parametrize(
    ("energy", "width", "round_nearest", "expected"),
    [
        (1.4, 0.5, False, 2),
        (1.4, 0.5, True, 3),
        (1.2, 0.5, True, 2),
        (10.0, 0.0, False, 0),
    ],
)
def test_energy_bin_index(energy, width, round_nearest, expected):
    assert energy_bin_index(energy, width, round_nearest=round_nearest) == expected
