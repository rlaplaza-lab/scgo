"""Tests for automatic global-search budgets."""

from __future__ import annotations

from math import ceil, exp, log, log1p

import pytest
from ase.build import fcc111

from scgo.surface.config import SurfaceSystemConfig
from scgo.surface.partition import resolve_slab_search_partition
from scgo.system_types import AdsorbateDefinition, SystemType
from scgo.utils.helpers import (
    POPULATION_MAX,
    POPULATION_MIN,
    POPULATION_OFFSET,
    POPULATION_SLOPE,
    auto_niter,
    auto_niter_local_relaxation,
    auto_niter_ts,
    auto_population_size,
)
from scgo.utils.run_helpers import resolve_auto_params
from scgo.utils.search_budget import (
    GROWTH_FACTOR,
    SYMMETRY_SUPPORTED,
    SYMMETRY_VACUUM,
    _split_slots,
    bh_auto_budget,
    estimate_search_complexity,
    ga_auto_budget,
)


class TestGaBudget:
    """Search slots follow the animal count; generations stay on the atom-count curve."""

    def test_population_follows_the_size_parameters(self):
        n_atoms = 10
        raw = POPULATION_OFFSET + POPULATION_SLOPE * log1p(n_atoms)
        expected = int(min(POPULATION_MAX, max(POPULATION_MIN, raw)))
        assert auto_population_size(["Pt"] * n_atoms) == expected
        assert ga_auto_budget(["Pt"] * n_atoms) == (expected, expected)
        gas = estimate_search_complexity(["Pt"] * n_atoms)
        assert gas.evaluations == pytest.approx(expected * expected)

    @pytest.mark.parametrize("n_atoms", [4, 10, 11, 13, 20, 30])
    def test_generations_stay_on_the_atom_count_curve(self, n_atoms):
        floor = auto_niter(["Pt"] * n_atoms)
        population, generations = ga_auto_budget(["Pt"] * n_atoms)
        assert generations == floor
        assert population >= floor
        assert population <= 1000

    @pytest.mark.parametrize(
        ("composition", "system_type", "population", "generations"),
        [
            (["Pt"] * 10, "gas_cluster", 86, 86),
            (["Pt"] * 10, "surface_cluster", 91, 86),
            (["Pt"] * 38, "gas_cluster", 336, 131),
            (["Pt"] * 55, "gas_cluster", 908, 143),
            (["Pt"] * 55, "surface_cluster", 958, 143),
            (["Pt"] * 5 + ["Co"] * 5, "gas_cluster", 102, 86),
        ],
    )
    def test_documented_examples(
        self,
        composition: list[str],
        system_type: SystemType,
        population: int,
        generations: int,
    ):
        """Pairs quoted on the search-effort page."""
        assert ga_auto_budget(composition, system_type=system_type) == (
            population,
            generations,
        )
        assert generations == auto_niter(composition)

    def test_population_cap_raises_generations_only_after_1000(self):
        population, generations, raised = _split_slots(1000 * 200, floor=80)
        assert (population, generations) == (1000, 200)
        assert raised

        population, generations, raised = _split_slots(1000 * 5000, floor=80)
        assert (population, generations) == (1000, 1000)
        assert raised

        population, generations = ga_auto_budget(["Pt"] * 75)
        assert population == 1000
        assert generations > auto_niter(["Pt"] * 75)

    def test_mixed_core_widens_the_population(self):
        composition = [
            "Pt",
            "Pt",
            "Co",
            "Co",
            "Co",
            "Co",
            "Co",
            "Co",
            "Pd",
            "Pd",
            "Pd",
            "Pd",
        ]
        pure_population, pure_generations = ga_auto_budget(["Pt"] * 12)
        mixed_population, mixed_generations = ga_auto_budget(composition)
        assert mixed_generations == pure_generations
        assert mixed_population > pure_population
        pure = estimate_search_complexity(["Pt"] * 12)
        mixed = estimate_search_complexity(composition)
        assert mixed.evaluations / pure.evaluations == pytest.approx(
            exp(GROWTH_FACTOR * (mixed.log_core - pure.log_core))
        )

    @pytest.mark.parametrize("n_atoms", [10, 12, 20, 40, 55])
    def test_supported_ratio_is_the_symmetry_factor(self, n_atoms):
        """Same animal growth; only the in-plane rotation group differs."""
        gas = estimate_search_complexity(["Pt"] * n_atoms, system_type="gas_cluster")
        supported = estimate_search_complexity(
            ["Pt"] * n_atoms,
            system_type="surface_cluster",
        )
        assert gas.dimension == 3
        assert supported.dimension == 2
        log_ratio = log(SYMMETRY_VACUUM / SYMMETRY_SUPPORTED)
        assert supported.log_core - gas.log_core == pytest.approx(log_ratio)
        assert supported.evaluations / gas.evaluations == pytest.approx(
            exp(GROWTH_FACTOR * log_ratio)
        )

    def test_budgets_stay_inside_hard_caps(self):
        composition = ["Pt", "Co", "Pd", "Au"] * 40
        population, generations = ga_auto_budget(composition)
        assert 3 <= population <= 1000
        assert 3 <= generations <= 1000


class TestAdsorbates:
    """Adsorbate atoms are sites, not extra core atoms."""

    def test_one_fragment_barely_changes_the_core_budget(self):
        core = ["Pt"] * 10
        definition = AdsorbateDefinition(
            core_symbols=list(core),
            adsorbate_symbols=["O", "H"],
            adsorbate_fragment_lengths=[2],
        )
        bare = estimate_search_complexity(core, system_type="gas_cluster")
        placed = estimate_search_complexity(
            [*core, "O", "H"],
            system_type="gas_cluster_adsorbate",
            adsorbate_definition=definition,
        )
        assert placed.n_core == bare.n_core == 10
        assert placed.log_sites > 0.0
        assert placed.n_mobile == len(core) + 2
        assert placed.evaluations / bare.evaluations == pytest.approx(
            exp(GROWTH_FACTOR * placed.log_sites)
        )
        assert placed.evaluations > bare.evaluations

    def test_several_fragments_raise_the_budget_above_the_bare_core(self):
        core = ["Pt"] * 20
        definition = AdsorbateDefinition(
            core_symbols=list(core),
            adsorbate_symbols=["H", "H", "H"],
            adsorbate_fragment_lengths=[1, 1, 1],
        )
        bare = estimate_search_complexity(core, system_type="gas_cluster")
        placed = estimate_search_complexity(
            [*core, "H", "H", "H"],
            system_type="gas_cluster_adsorbate",
            adsorbate_definition=definition,
        )
        assert placed.n_core == 20
        assert placed.log_sites > bare.log_sites
        assert placed.evaluations / bare.evaluations == pytest.approx(
            exp(GROWTH_FACTOR * placed.log_sites)
        )
        assert placed.evaluations > bare.evaluations

    def test_supported_sites_follow_the_exposed_area(self):
        core = ["Pt"] * 20
        definition = AdsorbateDefinition(
            core_symbols=list(core),
            adsorbate_symbols=["H", "H", "H"],
            adsorbate_fragment_lengths=[1, 1, 1],
        )
        composition = [*core, "H", "H", "H"]
        gas = estimate_search_complexity(
            composition,
            system_type="gas_cluster_adsorbate",
            adsorbate_definition=definition,
        )
        supported = estimate_search_complexity(
            composition,
            system_type="surface_cluster_adsorbate",
            adsorbate_definition=definition,
        )
        assert supported.log_sites == pytest.approx(gas.log_sites)
        assert supported.log_sites > 0.0
        assert supported.evaluations / gas.evaluations == pytest.approx(
            exp(GROWTH_FACTOR * log(SYMMETRY_VACUUM / SYMMETRY_SUPPORTED))
        )


class TestBasinHopping:
    """A unary three-atom basin-hopping run uses the atom-count hop count."""

    def test_three_atom_hop_count_matches_the_size_curve(self):
        assert bh_auto_budget(["Pt", "Pt", "Pt"]) == auto_niter(["Pt", "Pt", "Pt"])

    def test_larger_cluster_gets_more_hops_without_passing_the_cap(self):
        small = bh_auto_budget(["Pt"] * 3)
        large = bh_auto_budget(["Pt"] * 30)
        assert small < large <= 1000


class TestSlabSearch:
    """Bare slabs use the mobile top layer, not an empty formula."""

    def test_empty_formula_uses_the_mobile_top_layer(self):
        slab = fcc111("Pt", size=(2, 2, 3), vacuum=8.0, orthogonal=True)
        config = SurfaceSystemConfig(
            slab=slab,
            fix_all_slab_atoms=False,
            n_relax_top_slab_layers=1,
        )
        n_mobile = len(resolve_slab_search_partition(config).mobile_slab_symbols)
        assert n_mobile > 1
        population, generations = ga_auto_budget(
            [],
            system_type="surface",
            surface_config=config,
        )
        expected = auto_niter(["Pt"] * n_mobile)
        assert (population, generations) == (expected, expected)
        assert expected != auto_niter([])
        assert (
            bh_auto_budget([], system_type="surface", surface_config=config) == expected
        )

    def test_adsorbates_scale_hops_from_the_atom_count_not_the_slot_square(self):
        """Slab hops are ``ceil(M Ω_sites^α)``; GA slots stay on ``M² Ω_sites^α``."""
        slab = fcc111("Pt", size=(2, 2, 3), vacuum=8.0, orthogonal=True)
        config = SurfaceSystemConfig(
            slab=slab,
            fix_all_slab_atoms=False,
            n_relax_top_slab_layers=1,
        )
        definition = AdsorbateDefinition(
            core_symbols=[],
            adsorbate_symbols=["H", "H", "H"],
            adsorbate_fragment_lengths=[1, 1, 1],
        )
        composition = ["H", "H", "H"]
        kwargs = {
            "system_type": "surface_adsorbate",
            "surface_config": config,
            "adsorbate_definition": definition,
        }
        bare_hops = bh_auto_budget([], system_type="surface", surface_config=config)
        estimate = estimate_search_complexity(composition, **kwargs)
        floor = auto_niter(["Pt"] * estimate.n_mobile)
        assert estimate.log_sites > 0.0
        assert estimate.evaluations / (floor * floor) == pytest.approx(
            exp(GROWTH_FACTOR * estimate.log_sites)
        )
        hops = bh_auto_budget(composition, **kwargs)
        expected = min(1000, ceil(floor * exp(GROWTH_FACTOR * estimate.log_sites)))
        assert hops == expected
        assert bare_hops < hops
        population, generations = ga_auto_budget(composition, **kwargs)
        assert hops < population * generations


class TestResolver:
    """Automatic GA and BH settings go through the search budget."""

    def test_explicit_integers_win(self):
        resolved = resolve_auto_params(
            {"niter": 7, "population_size": 9},
            ["Pt"] * 10,
            "ga",
        )
        assert resolved["niter"] == 7
        assert resolved["population_size"] == 9

    def test_simple_optimizer_stays_on_the_size_curve(self):
        composition = ["Pt"] * 2
        resolved = resolve_auto_params({"niter": "auto"}, composition, "simple")
        assert resolved["niter"] == auto_niter(composition)

    def test_local_relaxation_and_neb_curves_are_unchanged(self):
        assert auto_niter_local_relaxation(["Pt"]) == 84
        assert auto_niter_ts(["Pt"]) == 174
        assert auto_population_size(["Pt"] * 10) == 86
        assert auto_niter(["Pt"] * 10) == 86
