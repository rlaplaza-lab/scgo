"""Regression tests for connectivity_factor in template probing (bug 1.3.6).

``_find_valid_template_types`` probed every generator with the *default*
connectivity factor and cached the outcome under ``n_atoms`` alone. Callers that
passed a non-default ``connectivity_factor`` therefore got results computed for a
different factor (or a cached result from an unrelated factor). The factor is now
threaded through the probe and is part of the cache key.
"""

import pytest

from scgo.initialization import templates as templates_module
from scgo.initialization.initialization_config import CONNECTIVITY_FACTOR
from scgo.initialization.templates import (
    _find_valid_template_types,
    generate_template_matches,
)

STRICT_FACTOR = 0.3
LENIENT_FACTOR = CONNECTIVITY_FACTOR  # 1.4


@pytest.fixture
def isolated_template_cache(monkeypatch):
    """Give each test a private template-validity cache."""
    cache: dict = {}
    monkeypatch.setattr(templates_module, "_VALID_TEMPLATE_TYPES_CACHE", cache)
    monkeypatch.setattr(templates_module, "_VALID_TEMPLATE_TYPES_INFLIGHT", {})
    return cache


class TestConnectivityFactorCacheKey:
    """The cache key must distinguish connectivity factors."""

    def test_cache_key_includes_connectivity_factor(self, isolated_template_cache):
        """Two factors produce two cache entries, not one."""
        _find_valid_template_types(8, STRICT_FACTOR)
        _find_valid_template_types(8, LENIENT_FACTOR)

        assert set(isolated_template_cache) == {
            (8, STRICT_FACTOR),
            (8, LENIENT_FACTOR),
        }

    def test_repeated_calls_reuse_the_same_entry(self, isolated_template_cache):
        """The same factor still hits the cache instead of re-probing."""
        first = _find_valid_template_types(8, STRICT_FACTOR)
        second = _find_valid_template_types(8, STRICT_FACTOR)

        assert first == second
        assert set(isolated_template_cache) == {(8, STRICT_FACTOR)}

    def test_default_factor_matches_explicit_default(self, isolated_template_cache):
        """Omitting the factor is equivalent to passing the default."""
        implicit = _find_valid_template_types(13)
        explicit = _find_valid_template_types(13, CONNECTIVITY_FACTOR)

        assert implicit == explicit
        assert set(isolated_template_cache) == {(13, float(CONNECTIVITY_FACTOR))}


class TestConnectivityFactorAffectsValidity:
    """A stricter connectivity factor must be able to invalidate templates."""

    def test_strict_factor_yields_fewer_valid_types(self, isolated_template_cache):
        """The strict probe is a strict subset of the lenient probe for 8 atoms."""
        strict = set(_find_valid_template_types(8, STRICT_FACTOR))
        lenient = set(_find_valid_template_types(8, LENIENT_FACTOR))

        assert strict != lenient
        assert strict < lenient

    def test_moderate_factor_difference_is_visible(self, isolated_template_cache):
        """Two plausible factors give different validity sets for 20 atoms."""
        stricter = set(_find_valid_template_types(20, 0.8))
        default = set(_find_valid_template_types(20, LENIENT_FACTOR))

        assert stricter != default

    def test_cached_strict_result_does_not_leak_to_default(
        self, isolated_template_cache
    ):
        """A strict probe cached first does not poison the default probe."""
        strict = _find_valid_template_types(8, STRICT_FACTOR)
        default = _find_valid_template_types(8)

        assert strict != default
        assert len(default) > len(strict)


class TestGenerateTemplateMatchesForwardsFactor:
    """``generate_template_matches`` must pass its factor into the probe."""

    def test_factor_is_forwarded_to_probe(self, monkeypatch, rng):
        """The connectivity factor reaches _find_valid_template_types."""
        seen: list[tuple] = []

        def _spy(n_atoms, connectivity_factor=CONNECTIVITY_FACTOR):
            seen.append((n_atoms, connectivity_factor))
            return []

        monkeypatch.setattr(templates_module, "_find_valid_template_types", _spy)

        generate_template_matches(
            composition=["Pt"] * 13,
            n_atoms=13,
            rng=rng,
            cell_side=20.0,
            connectivity_factor=2.5,
        )

        assert seen == [(13, 2.5)]
