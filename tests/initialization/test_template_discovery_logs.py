import logging

from ase import Atoms
from ase.cluster import Icosahedron

from scgo.initialization import templates


def test_grow_template_logs_discovery_failure(monkeypatch, caplog, rng):
    """When template growth cannot generate facet positions, a concise "discovery"
    debug message should be logged (not a per-structure fallback message).
    """

    # Force _generate_batch_positions_on_convex_hull to return no candidates
    def fake_generate_batch_positions_on_convex_hull(*args, **kwargs):
        return []

    monkeypatch.setattr(
        templates,
        "_generate_batch_positions_on_convex_hull",
        fake_generate_batch_positions_on_convex_hull,
    )

    base = Icosahedron("Pt", 2)
    base.center()
    base.set_cell([30.0, 30.0, 30.0])
    comp = ["Pt"] * 20

    caplog.set_level(logging.DEBUG)
    result = templates.grow_template_via_facets(
        base,
        comp,
        placement_radius_scaling=templates.PLACEMENT_RADIUS_SCALING_DEFAULT,
        cell_side=30.0,
        rng=rng,
        min_distance_factor=templates.MIN_DISTANCE_FACTOR_DEFAULT,
        connectivity_factor=templates.CONNECTIVITY_FACTOR,
    )

    assert result is None
    # Ensure the concise discovery clarification is present in logs
    assert any("discovery failure" in r.getMessage() for r in caplog.records)


def test_shrink_refuses_large_removal_logs_debug(caplog, rng):
    """Refuse shrinking by ≥50% of the base and log a DEBUG reason."""
    caplog.set_level(logging.DEBUG)
    # Base = 13 (icosahedron), target = 4 → removal ratio > 0.5
    result = templates._generate_template_with_atom_adjustment(
        base_template_type="icosahedron",
        base_n_atoms=13,
        target_n_atoms=4,
        composition=["Pt"] * 4,
        rng=rng,
        cell_side=20.0,
    )
    assert result is None
    assert any("Template shrink refused" in r.getMessage() for r in caplog.records)


def test_validate_and_add_template_logs_validation_drop(monkeypatch, caplog):
    """Validation rejects should emit DEBUG with the error message."""

    def fake_validate_cluster(*args, **kwargs):
        return Atoms("Pt"), False, "clash detected"

    monkeypatch.setattr(templates, "validate_cluster", fake_validate_cluster)
    caplog.set_level(logging.DEBUG)
    results: list[Atoms] = []
    ok = templates._validate_and_add_template(
        atoms=Atoms("Pt"),
        results=results,
        template_type="icosahedron",
        template_description="for 1 atoms",
        min_distance_factor=templates.MIN_DISTANCE_FACTOR_DEFAULT,
        connectivity_factor=templates.CONNECTIVITY_FACTOR,
    )
    assert ok is False
    assert results == []
    assert any(
        "Template validation failed" in r.getMessage()
        and "clash detected" in r.getMessage()
        for r in caplog.records
    )
