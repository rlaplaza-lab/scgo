"""Regression tests for the template rotation cache key (bug 1.3.4).

``get_structure_signature`` only encodes sorted interatomic distances, so
isostructural clusters of different elements (``Pt3`` vs ``Au3``) share a
signature. Keying the rotation cache on ``(signature, cell_side)`` therefore made
``Au3`` reuse the cached ``Pt3`` rotations and come back as a ``Pt3`` cluster.
The chemical symbols are now part of the key.
"""

from typing import Any

import numpy as np
import pytest
from ase import Atoms

from scgo.database.cache import get_global_cache
from scgo.initialization.initialization_config import (
    CONNECTIVITY_FACTOR,
    MIN_DISTANCE_FACTOR_DEFAULT,
)
from scgo.initialization.initializers import (
    TEMPLATE_ROTATIONS_CACHE_NS,
    _apply_template_rotation_and_validate,
)

TRIANGLE_POSITIONS = [[0.0, 0.0, 0.0], [2.7, 0.0, 0.0], [1.35, 2.34, 0.0]]
CELL_SIDE = 20.0


def _rotation_cache_keys() -> list[Any]:
    """Return the keys currently stored in the template-rotation namespace."""
    cache = get_global_cache()
    return [
        key[1]
        for key in list(cache._cache.keys())
        if key[0] == TEMPLATE_ROTATIONS_CACHE_NS
    ]


def _rotate(atoms: Atoms, seed: int = 0) -> Atoms | None:
    """Run the cached rotation/validation helper on ``atoms``."""
    return _apply_template_rotation_and_validate(
        atoms,
        CELL_SIDE,
        np.random.default_rng(seed),
        MIN_DISTANCE_FACTOR_DEFAULT,
        CONNECTIVITY_FACTOR,
    )


@pytest.mark.requires_cache_isolation
class TestRotationCacheKey:
    """Distinct compositions must not share rotation cache entries."""

    def test_pt3_and_au3_use_distinct_cache_keys(self):
        """Identical geometries with different elements get different keys."""
        get_global_cache().clear_namespace(TEMPLATE_ROTATIONS_CACHE_NS)

        assert _rotate(Atoms("Pt3", positions=TRIANGLE_POSITIONS)) is not None
        keys_after_pt = _rotation_cache_keys()
        assert len(keys_after_pt) == 1

        assert _rotate(Atoms("Au3", positions=TRIANGLE_POSITIONS)) is not None
        keys_after_au = _rotation_cache_keys()

        assert len(keys_after_au) == 2, (
            "Pt3 and Au3 share a distance signature and collided in the "
            f"rotation cache: {keys_after_au}"
        )
        assert len(set(keys_after_au)) == 2

    def test_cached_rotations_keep_their_composition(self):
        """A cache hit never swaps the chemical symbols of the template."""
        get_global_cache().clear_namespace(TEMPLATE_ROTATIONS_CACHE_NS)

        pt_result = _rotate(Atoms("Pt3", positions=TRIANGLE_POSITIONS))
        au_result = _rotate(Atoms("Au3", positions=TRIANGLE_POSITIONS))

        assert pt_result is not None
        assert au_result is not None
        assert set(pt_result.get_chemical_symbols()) == {"Pt"}
        assert set(au_result.get_chemical_symbols()) == {"Au"}

    def test_same_composition_reuses_cache_entry(self):
        """Repeating the same template does not add a second cache entry."""
        get_global_cache().clear_namespace(TEMPLATE_ROTATIONS_CACHE_NS)

        _rotate(Atoms("Pt3", positions=TRIANGLE_POSITIONS))
        _rotate(Atoms("Pt3", positions=TRIANGLE_POSITIONS), seed=1)

        assert len(_rotation_cache_keys()) == 1

    def test_cell_side_still_part_of_key(self):
        """The cell side remains part of the cache key."""
        get_global_cache().clear_namespace(TEMPLATE_ROTATIONS_CACHE_NS)

        atoms = Atoms("Pt3", positions=TRIANGLE_POSITIONS)
        _apply_template_rotation_and_validate(
            atoms,
            CELL_SIDE,
            np.random.default_rng(0),
            MIN_DISTANCE_FACTOR_DEFAULT,
            CONNECTIVITY_FACTOR,
        )
        _apply_template_rotation_and_validate(
            atoms,
            CELL_SIDE + 5.0,
            np.random.default_rng(0),
            MIN_DISTANCE_FACTOR_DEFAULT,
            CONNECTIVITY_FACTOR,
        )

        assert len(_rotation_cache_keys()) == 2
