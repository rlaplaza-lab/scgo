"""Tests for atomic radius patching and caching."""

import logging

import numpy as np
import pytest
from ase.data import atomic_numbers, vdw_radii

from scgo.initialization.atomic_radii import (
    clear_atomic_radii_cache,
    get_covalent_radius,
    get_vdw_radius,
)
from scgo.initialization.initializers import compute_cell_side


@pytest.fixture(autouse=True)
def _reset_radii_cache():
    clear_atomic_radii_cache()
    yield
    clear_atomic_radii_cache()


class TestAtomicRadii:
    def test_covalent_radius_known_element(self):
        assert get_covalent_radius("Pt") == pytest.approx(1.36)

    def test_vdw_radius_known_element(self):
        z = atomic_numbers["Pt"]
        assert get_vdw_radius("Pt") == pytest.approx(float(vdw_radii[z]))

    def test_vdw_radius_nan_element_is_finite(self):
        z = atomic_numbers["Co"]
        assert not np.isfinite(vdw_radii[z])
        r = get_vdw_radius("Co")
        assert np.isfinite(r)
        assert r > 0

    def test_vdw_radius_cached(self):
        first = get_vdw_radius("Co")
        second = get_vdw_radius("Co")
        assert first == second

    def test_patch_logged_once(self, caplog):
        with caplog.at_level(logging.INFO):
            for _ in range(5):
                get_vdw_radius("Co")
        co_messages = [r.message for r in caplog.records if "Co" in r.message]
        assert len(co_messages) == 1
        assert "interpolated" in co_messages[0]

    def test_unknown_element_raises(self):
        with pytest.raises(ValueError, match="Unknown element symbol"):
            get_vdw_radius("Xx")

    def test_compute_cell_side_uses_patched_vdw(self):
        side = compute_cell_side(["Pt", "Pt", "Pt", "Pt", "Co"], vacuum=10.0)
        assert np.isfinite(side)
        assert side > 0
