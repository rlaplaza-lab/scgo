"""Regression tests for the multi-element template weight penalty (bug 1.3.1).

``MULTI_ELEMENT_TEMPLATE_PENALTY`` (0.9) is larger than the base weight of the
low-ranked template types (``cube``/``tetrahedron`` at 0.8). Without clamping,
``_calculate_template_weight`` could return a negative weight; when the pool also
contained a high-weight type the total stayed positive and the negative entry
reached ``rng.choice(p=...)``, which raises
``ValueError: probabilities are not non-negative``.
"""

import numpy as np
import pytest

from scgo.initialization import initializers as initializers_module
from scgo.initialization.initialization_config import (
    CONNECTIVITY_FACTOR,
    MIN_DISTANCE_FACTOR_DEFAULT,
    MULTI_ELEMENT_TEMPLATE_PENALTY,
    PLACEMENT_RADIUS_SCALING_DEFAULT,
    TEMPLATE_BASE_WEIGHTS,
)
from scgo.initialization.initializers import (
    _calculate_template_weight,
    _try_template_generation,
)
from scgo.initialization.templates import generate_template_matches

MULTI_ELEMENT_COMPOSITION_8 = ["Pt"] * 4 + ["Au"] * 4


class TestTemplateWeightNonNegative:
    """The per-template weight must never go below zero."""

    @pytest.mark.parametrize("template_type", sorted(TEMPLATE_BASE_WEIGHTS))
    @pytest.mark.parametrize("total_candidates", [1, 2, 3, 5, 10])
    def test_weight_non_negative_for_multi_element(
        self, template_type, total_candidates
    ):
        """Multi-element runs never produce a negative template weight."""
        counts = {template_type: total_candidates}
        weight = _calculate_template_weight(
            template_type,
            n_unique_elements=2,
            template_type_counts=counts,
            total_candidates=total_candidates,
        )
        assert weight >= 0.0

    def test_low_base_weight_type_is_clamped_not_negative(self):
        """A base weight below the penalty is clamped to zero, not negative."""
        low_base_type = min(TEMPLATE_BASE_WEIGHTS, key=TEMPLATE_BASE_WEIGHTS.get)
        assert TEMPLATE_BASE_WEIGHTS[low_base_type] < MULTI_ELEMENT_TEMPLATE_PENALTY

        # Two candidates of two distinct types: the diversity boost is small
        # (0.075), so the raw value would be 0.8 + 0.075 - 0.9 = -0.025.
        counts = {low_base_type: 1, "icosahedron": 1}
        weight = _calculate_template_weight(
            low_base_type,
            n_unique_elements=2,
            template_type_counts=counts,
            total_candidates=2,
        )
        assert weight == 0.0

    def test_weights_form_valid_probability_vector(self):
        """Weights from a mixed pool normalize to a valid probability vector."""
        counts = {"cube": 1, "icosahedron": 1}
        weights = [
            _calculate_template_weight(
                template_type,
                n_unique_elements=2,
                template_type_counts=counts,
                total_candidates=2,
            )
            for template_type in ("cube", "icosahedron")
        ]
        total = sum(weights)
        assert total > 0
        probabilities = [w / total for w in weights]

        rng = np.random.default_rng(0)
        # Raised "probabilities are not non-negative" before the clamp fix.
        selected = int(rng.choice(len(probabilities), p=probabilities))
        assert 0 <= selected < len(probabilities)

    def test_single_element_weights_unchanged(self):
        """Single-element runs keep the unpenalized weight."""
        weight = _calculate_template_weight(
            "cube",
            n_unique_elements=1,
            template_type_counts={"cube": 1},
            total_candidates=1,
        )
        assert weight == pytest.approx(TEMPLATE_BASE_WEIGHTS["cube"])


class TestTemplateSelectionWithLowBasePool:
    """End-to-end selection with a pool dominated by low-base template types."""

    def _restricted_candidates(self):
        """Return a two-template pool containing the low-base ``cube`` type."""
        rng = np.random.default_rng(7)
        candidates = generate_template_matches(
            composition=MULTI_ELEMENT_COMPOSITION_8,
            n_atoms=8,
            rng=rng,
            cell_side=20.0,
        )
        by_type = {c.info.get("template_type"): c for c in candidates}
        assert "cube" in by_type, "expected a cube template for 8 atoms"
        assert "icosahedron" in by_type, "expected an icosahedron template for 8 atoms"
        return [by_type["cube"], by_type["icosahedron"]]

    def test_multi_element_template_generation_does_not_raise(self, monkeypatch):
        """Weighted selection succeeds when a low-base template is in the pool."""
        restricted = self._restricted_candidates()

        def _fake_matches(*_args, **_kwargs):
            return [c.copy() for c in restricted]

        monkeypatch.setattr(
            initializers_module, "generate_template_matches", _fake_matches
        )

        result = _try_template_generation(
            composition=list(MULTI_ELEMENT_COMPOSITION_8),
            n_atoms=8,
            cell_side=20.0,
            rng=np.random.default_rng(3),
            placement_radius_scaling=PLACEMENT_RADIUS_SCALING_DEFAULT,
            min_distance_factor=MIN_DISTANCE_FACTOR_DEFAULT,
            connectivity_factor=CONNECTIVITY_FACTOR,
        )

        assert result is not None
        assert len(result) == 8
