"""Regression tests for linear/planar Co4 hull fallback (QH6154)."""

from __future__ import annotations

import logging

import numpy as np
import pytest
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii

from scgo.cluster_adsorbate.sites import (
    compute_surface_site_candidates,
    count_site_candidates,
)
from scgo.initialization.geometry_helpers import (
    _classify_seed_geometry,
    _generate_batch_positions_on_convex_hull,
    clear_convex_hull_cache,
    get_convex_hull_vertex_indices,
    resolve_cluster_extent,
    try_convex_hull,
)
from scgo.initialization.initializers import compute_cell_side
from scgo.initialization.templates import (
    generate_template_matches,
    remove_atoms_from_vertices,
)


def _co_bond() -> float:
    return 2.0 * float(covalent_radii[atomic_numbers["Co"]])


def _linear_co4() -> Atoms:
    a = _co_bond()
    return Atoms("Co4", positions=[[0.0, 0.0, i * a] for i in range(4)], pbc=False)


def _planar_co4_square() -> Atoms:
    a = _co_bond()
    s = a / np.sqrt(2.0)
    return Atoms(
        "Co4",
        positions=[
            [0.0, -s, 0.0],
            [0.0, 0.0, -s],
            [0.0, s, 0.0],
            [0.0, 0.0, s],
        ],
        pbc=False,
    )


def _center_in_vacuum(atoms: Atoms) -> Atoms:
    out = atoms.copy()
    side = compute_cell_side(["Co"] * len(out), vacuum=8.0)
    out.set_cell([side, side, side])
    out.center()
    return out


@pytest.fixture(autouse=True)
def _clear_hull_caches():
    clear_convex_hull_cache()
    yield
    clear_convex_hull_cache()


def test_linear_and_planar_extent():
    linear = _linear_co4()
    planar = _planar_co4_square()
    assert resolve_cluster_extent(linear.get_positions()).kind == "linear"
    assert _classify_seed_geometry(linear) == "linear"
    assert set(get_convex_hull_vertex_indices(linear)) == {0, 3}

    assert resolve_cluster_extent(planar.get_positions()).kind == "planar"
    assert _classify_seed_geometry(planar) == "planar"
    assert set(get_convex_hull_vertex_indices(planar)) == {0, 1, 2, 3}


def test_no_qhull_dump_in_logs(caplog):
    atoms = _center_in_vacuum(_linear_co4())
    with caplog.at_level(logging.DEBUG, logger="scgo.initialization.geometry_helpers"):
        assert try_convex_hull(atoms.get_positions()) is None
    assert "QH6154" not in caplog.text
    assert "While executing" not in caplog.text


def test_sites_from_degenerate_cores():
    linear_sites = compute_surface_site_candidates(_linear_co4())
    assert len(linear_sites["vertex"]) == 2
    assert len(linear_sites["edge"]) == 1
    assert len(linear_sites["facet"]) == 0

    planar_sites = compute_surface_site_candidates(_planar_co4_square())
    assert count_site_candidates(planar_sites) > 0
    assert len(planar_sites["facet"]) == 1
    normal = planar_sites["facet"][0].normal
    assert abs(float(np.dot(normal, [1.0, 0.0, 0.0]))) == pytest.approx(1.0, abs=1e-6)


def test_growth_places_off_linear_axis(rng):
    atoms = _linear_co4()
    candidates = _generate_batch_positions_on_convex_hull(
        atoms, n_candidates=4, bond_distance=_co_bond(), rng=rng
    )
    assert candidates
    center = atoms.get_center_of_mass()
    axis = np.array([0.0, 0.0, 1.0])
    for cand in candidates:
        radial = cand - center
        off_axis = radial - float(np.dot(radial, axis)) * axis
        assert float(np.linalg.norm(off_axis)) > 0.5


def test_co4_template_matches_reject_linear(rng):
    matches = generate_template_matches(["Co"] * 4, 4, rng=rng)
    assert matches
    assert "decahedron" not in {m.info.get("template_type") for m in matches}
    for atoms in matches:
        assert resolve_cluster_extent(atoms.get_positions()).kind != "linear"


def test_planar_shrink_does_not_use_pca_vertices(rng):
    """Planar magic shells cannot be shrunk via 2D extent vertices."""
    planar = _planar_co4_square()
    assert remove_atoms_from_vertices(planar, n_remove=1, rng=rng) is None
