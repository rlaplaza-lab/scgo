"""Regression tests for template caps in strategy allocation (bug 1.3.2).

``_calculate_target_allocations`` capped the template target at
``2 * n_templates``, so with plentiful structures the allocator handed out more
template slots than there were templates, re-using templates instead of
respecting the availability guard used elsewhere in the module.
"""

from collections import Counter

from ase import Atoms

from scgo.initialization.strategy_allocation import (
    _allocate_initialization_strategies,
    _calculate_target_allocations,
)


def _make_templates(n_templates: int) -> list[Atoms]:
    """Build ``n_templates`` distinct dummy template clusters."""
    return [
        Atoms("Pt2", positions=[[0.0, 0.0, 0.0], [2.5 + 0.1 * i, 0.0, 0.0]])
        for i in range(n_templates)
    ]


class TestTemplateTargetCap:
    """Target counts must never exceed the number of available templates."""

    def test_target_allocation_capped_by_n_templates(self):
        """A large structure budget cannot request more templates than exist."""
        targets = _calculate_target_allocations(
            n_templates=5, n_seed_combinations=0, n_structures=100
        )
        assert targets["template"] <= 5

    def test_target_allocation_scales_below_cap(self):
        """Small structure budgets are still limited by n_structures."""
        targets = _calculate_target_allocations(
            n_templates=5, n_seed_combinations=0, n_structures=2
        )
        assert targets["template"] <= 2


class TestAllocateInitializationStrategies:
    """End-to-end allocation behaviour with plentiful templates."""

    def test_template_allocations_respect_template_count(self, rng):
        """Template allocations never exceed the number of templates available."""
        n_templates = 5
        n_structures = 100
        templates = _make_templates(n_templates)

        allocations = _allocate_initialization_strategies(
            n_structures=n_structures,
            templates=templates,
            n_seed_formulas=0,
            n_seed_combinations=0,
            rng=rng,
            n_atoms=2,
        )

        template_allocations = [a for a in allocations if a[0] == "template"]
        assert len(template_allocations) <= n_templates

        # Every template is used, and none is used more than once.
        usage = Counter(idx for _, idx in template_allocations)
        assert set(usage) == set(range(n_templates))
        assert max(usage.values()) == 1

        # The full budget is still allocated.
        assert len(allocations) == n_structures

    def test_allocation_with_seeds_still_capped(self, rng):
        """The template cap holds when seed combinations are also available."""
        n_templates = 4
        n_structures = 60
        templates = _make_templates(n_templates)

        allocations = _allocate_initialization_strategies(
            n_structures=n_structures,
            templates=templates,
            n_seed_formulas=2,
            n_seed_combinations=3,
            rng=rng,
            n_atoms=2,
        )

        template_count = sum(1 for s, _ in allocations if s == "template")
        assert template_count <= n_templates
        assert len(allocations) == n_structures

    def test_fewer_structures_than_templates(self, rng):
        """With fewer structures than templates the budget is still respected."""
        templates = _make_templates(10)

        allocations = _allocate_initialization_strategies(
            n_structures=3,
            templates=templates,
            n_seed_formulas=0,
            n_seed_combinations=0,
            rng=rng,
            n_atoms=2,
        )

        template_count = sum(1 for s, _ in allocations if s == "template")
        assert template_count <= 10
        assert len(allocations) == 3
