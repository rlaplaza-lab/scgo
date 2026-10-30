"""Tests for adaptive mutation weight calculation."""

import pytest

from scgo.system_types import AdsorbateDefinition
from scgo.utils.mutation_weights import (
    calculate_generation_adjustment,
    calculate_size_adjustment,
    calculate_system_type_weights,
    decay_operator_credit,
    get_adaptive_mutation_config,
    scale_weights_by_operator_credit,
    snapshot_operator_credit,
)


def test_calculate_size_adjustment_small():
    """Test size adjustments for small cluster."""
    params = calculate_size_adjustment(3)

    assert params["rattle_strength"] == pytest.approx(1.0)
    assert params["mutation_probability"] == pytest.approx(0.30)
    assert "rattle_prop" in params


def test_calculate_size_adjustment_medium():
    """Test size adjustments for medium cluster."""
    params = calculate_size_adjustment(7)

    assert params["rattle_strength"] == pytest.approx(0.8)
    assert params["mutation_probability"] == pytest.approx(0.25)


def test_calculate_size_adjustment_large():
    """Test size adjustments for large cluster."""
    params = calculate_size_adjustment(15)

    assert params["rattle_strength"] == pytest.approx(0.6)
    assert params["mutation_probability"] == pytest.approx(0.20)


def test_calculate_generation_adjustment_early():
    """Test generation adjustments for early stage."""
    params = calculate_generation_adjustment(0, 10)

    assert params["mutation_probability_multiplier"] > 1.0
    assert params["rattle_strength_multiplier"] > 1.0


def test_calculate_generation_adjustment_mid():
    """Test generation adjustments for mid stage."""
    params = calculate_generation_adjustment(5, 10)

    assert params["mutation_probability_multiplier"] == pytest.approx(1.0)
    assert params["rattle_strength_multiplier"] == pytest.approx(1.0)


def test_calculate_generation_adjustment_late():
    """Test generation adjustments for late stage."""
    params = calculate_generation_adjustment(9, 10)

    assert params["mutation_probability_multiplier"] < 1.0
    assert params["rattle_strength_multiplier"] < 1.0


def test_get_adaptive_mutation_config_adaptive():
    """Test full adaptive configuration."""
    composition = ["Pt", "Pt", "Pt", "Pt"]
    config = get_adaptive_mutation_config(
        composition,
        current_generation=0,
        total_generations=10,
        use_adaptive=True,
    )

    assert "operator_weights" in config
    assert "use_permutation" in config
    assert "mutation_probability" in config
    assert "rattle_strength" in config
    assert "rattle_prop" in config

    # Check that values are in reasonable ranges
    assert 0.1 <= config["mutation_probability"] <= 0.65
    assert 0.3 <= config["rattle_strength"] <= 1.2
    assert abs(sum(config["operator_weights"].values()) - 1.0) < 1e-6
    assert "anisotropic_in_plane_strength" in config
    assert "anisotropic_normal_strength" in config
    assert "anisotropic_rattle_prop" in config


def test_get_adaptive_mutation_config_non_adaptive():
    """Test that non-adaptive mode returns default configuration."""
    composition = ["Pt", "Pt", "Pt", "Pt"]
    config = get_adaptive_mutation_config(composition, use_adaptive=False)

    # Should match the original default configuration
    assert config["use_permutation"] is False
    assert config["mutation_probability"] == pytest.approx(0.2)
    assert config["rattle_strength"] == pytest.approx(0.8)


def test_adaptive_config_evolution():
    """Test that configuration changes across generations."""
    composition = ["Pt", "Pt", "Pt", "Pt"]

    config_early = get_adaptive_mutation_config(
        composition,
        current_generation=0,
        total_generations=10,
    )
    config_late = get_adaptive_mutation_config(
        composition,
        current_generation=9,
        total_generations=10,
    )

    # Mutation probability should decrease over generations
    assert config_early["mutation_probability"] > config_late["mutation_probability"]

    # Rattle strength should decrease over generations
    assert config_early["rattle_strength"] > config_late["rattle_strength"]


def test_composition_effect_on_permutation():
    """Test that permutation weight varies with composition mixing."""
    # Pure cluster - no permutation
    config_pure = get_adaptive_mutation_config(["Pt"] * 4)
    assert not config_pure["use_permutation"]

    # Well-mixed cluster - high permutation weight
    config_mixed = get_adaptive_mutation_config(["Au", "Au", "Pt", "Pt"])
    assert config_mixed["use_permutation"]
    assert (
        config_mixed["operator_weights"]["permutation"]
        >= config_mixed["operator_weights"]["rattle"]
    )

    # Core-shell cluster - moderate permutation weight
    config_core_shell = get_adaptive_mutation_config(["Au"] + ["Pt"] * 5)
    assert config_core_shell["use_permutation"]


def test_size_effect_on_mutation_probability():
    """Test that mutation probability decreases with cluster size."""
    small_config = get_adaptive_mutation_config(["Pt"] * 3)
    medium_config = get_adaptive_mutation_config(["Pt"] * 7)
    large_config = get_adaptive_mutation_config(["Pt"] * 15)

    assert (
        small_config["mutation_probability"]
        > medium_config["mutation_probability"]
        > large_config["mutation_probability"]
    )


def test_stagnation_boost_increases_exploration_pressure():
    """Stagnation should increase mutation probability and flat operators."""
    composition = ["Pt", "Pt", "Pt", "Pt"]
    baseline = get_adaptive_mutation_config(
        composition,
        current_generation=3,
        total_generations=10,
        generations_without_improvement=0,
    )
    stalled = get_adaptive_mutation_config(
        composition,
        current_generation=3,
        total_generations=10,
        generations_without_improvement=8,
        stagnation_trigger=4,
        stagnation_full_trigger=8,
        aggressive_burst_multiplier=1.8,
        max_mutation_probability=0.65,
    )

    assert stalled["mutation_probability"] >= baseline["mutation_probability"]
    assert (
        stalled["operator_weights"]["anisotropic_rattle"]
        >= baseline["operator_weights"]["anisotropic_rattle"]
    )


def test_stagnation_boost_respects_max_probability_cap():
    """Mutation probability should never exceed configured cap."""
    composition = ["Pt", "Pt", "Pt", "Pt"]
    config = get_adaptive_mutation_config(
        composition,
        current_generation=0,
        total_generations=10,
        generations_without_improvement=100,
        stagnation_trigger=1,
        stagnation_full_trigger=2,
        aggressive_burst_multiplier=3.0,
        max_mutation_probability=0.55,
    )
    assert config["mutation_probability"] <= 0.55 + 1e-12


def test_gas_cluster_adaptive_omits_in_plane_slide() -> None:
    weights, use_perm = calculate_system_type_weights(
        "gas_cluster",
        ["Pt", "Pt", "Pt"],
    )
    assert not use_perm
    assert "in_plane_slide" not in weights
    assert "fragment_reposition" not in weights


def test_adsorbate_weights_use_core_composition_not_adsorbate_elements() -> None:
    comp = ["Pt", "Pt", "Pt", "O", "H"]
    ads = AdsorbateDefinition(
        core_symbols=["Pt", "Pt", "Pt"],
        adsorbate_symbols=["O", "H"],
        adsorbate_fragment_lengths=[2],
    )
    weights, use_perm = calculate_system_type_weights(
        "gas_cluster_adsorbate",
        comp,
        adsorbate_definition=ads,
    )
    assert not use_perm
    assert "permutation" not in weights
    assert weights["fragment_reposition"] > weights.get("rotational", 0.0)


def test_surface_cluster_adaptive_includes_orientation_ops() -> None:
    weights, _ = calculate_system_type_weights("surface_cluster", ["Pt", "Pt", "Pt"])
    assert weights["in_plane_slide"] > 0.05
    assert weights["in_plane_rotate"] > 0.05


def test_surface_cluster_adsorbate_prioritizes_fragment_reposition() -> None:
    comp = ["Pt", "Pt", "Pt", "O", "H"]
    ads = AdsorbateDefinition(
        core_symbols=["Pt", "Pt", "Pt"],
        adsorbate_symbols=["O", "H"],
        adsorbate_fragment_lengths=[2],
    )
    weights, _ = calculate_system_type_weights(
        "surface_cluster_adsorbate",
        comp,
        adsorbate_definition=ads,
    )
    assert weights["fragment_reposition"] >= 0.18
    assert weights["in_plane_rotate"] > 0.0


def test_stagnation_boost_increases_permutation_for_alloys() -> None:
    """Order-disorder swaps must gain mass under stagnation."""
    from scgo.utils.mutation_weights import _apply_stagnation_boost

    base, _ = calculate_system_type_weights("gas_cluster", ["Au", "Au", "Pt", "Pt"])
    boosted = _apply_stagnation_boost(base, level=1.0, burst_multiplier=1.8)
    assert boosted["permutation"] > base["permutation"]
    assert abs(sum(boosted.values()) - 1.0) < 1e-6


def test_gas_alloy_table_lists_only_gas_registered_operators() -> None:
    """Surface-only keys (mirror / in_plane_slide) stay out of gas tables."""
    weights, use_perm = calculate_system_type_weights(
        "gas_cluster", ["Au", "Au", "Pt", "Pt"]
    )
    assert use_perm is True
    assert "mirror" not in weights
    assert "in_plane_slide" not in weights
    # Effective distribution unchanged versus the pre-cleanup table: the same
    # relative shares over the registered subset (well-mixed alloy branch).
    expected = {
        "rattle": 0.13,
        "overlap_relief": 0.12,
        "permutation": 0.18,
        "shell_swap": 0.14,
        "flattening": 0.10,
        "rotational": 0.07,
        "anisotropic_rattle": 0.10,
        "breathing": 0.04,
    }
    total = sum(expected.values())
    for name, raw in expected.items():
        assert weights[name] == pytest.approx(raw / total)


def test_early_generations_favor_global_operators() -> None:
    early = get_adaptive_mutation_config(
        ["Pt"] * 10, current_generation=0, total_generations=10
    )
    late = get_adaptive_mutation_config(
        ["Pt"] * 10, current_generation=9, total_generations=10
    )
    assert (
        early["operator_weights"]["flattening"] > late["operator_weights"]["flattening"]
    )
    assert early["operator_weights"]["rattle"] < late["operator_weights"]["rattle"]
    assert (
        early["exploratory_crossover_probability"]
        > late["exploratory_crossover_probability"]
    )
    assert late["permutation_probability"] < early["permutation_probability"]


def test_stagnation_raises_exploratory_crossover_and_unary_fraction() -> None:
    calm = get_adaptive_mutation_config(
        ["Pt", "Au"] * 5,
        current_generation=3,
        total_generations=10,
        generations_without_improvement=0,
    )
    stagnant = get_adaptive_mutation_config(
        ["Pt", "Au"] * 5,
        current_generation=3,
        total_generations=10,
        generations_without_improvement=8,
    )
    assert (
        stagnant["exploratory_crossover_probability"]
        > calm["exploratory_crossover_probability"]
    )
    assert calm["unary_offspring_fraction"] == 0.0
    assert stagnant["unary_offspring_fraction"] > 0.0


def test_operator_credit_downweights_duplicates() -> None:
    scaled = scale_weights_by_operator_credit(
        [0.5, 0.5],
        ["dead", "live"],
        {
            "dead": {"trials": 10, "unique": 0},
            "live": {"trials": 10, "unique": 8},
        },
    )
    assert scaled[0] < scaled[1]
    assert sum(scaled) == pytest.approx(1.0)


def test_credit_prior_keeps_a_short_losing_streak_above_the_floor() -> None:
    scaled = scale_weights_by_operator_credit(
        [0.5, 0.5],
        ["cold", "idle"],
        {"cold": {"trials": 4, "unique": 0}},
    )
    assert scaled[0] > 0.3
    assert scaled[0] < scaled[1]


def test_operator_credit_snapshot_is_independent_of_later_decay() -> None:
    stats = {"rattle": {"trials": 8, "unique": 4}}
    decay_operator_credit(stats)
    assert stats["rattle"] == {"trials": 4, "unique": 2}
    frozen = snapshot_operator_credit(stats)
    stats["rattle"]["trials"] = 99
    assert frozen["rattle"]["trials"] == 4
