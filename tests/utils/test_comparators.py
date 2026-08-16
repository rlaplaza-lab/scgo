"""Comparator regression and fingerprint tests (consolidated).

Merges the former ``test_comparators_mic.py`` and
``test_comparator_composition.py`` suites and the comparator tests that lived in
``test_fitness_strategies.py``.
"""

from __future__ import annotations

import math

import numpy as np
from ase import Atoms
from ase.build import fcc111

from scgo.ase_ga_patches.population import count_looks_like
from scgo.surface.config import SurfaceSystemConfig
from scgo.system_types import resolve_neb_mic, resolve_structure_mic
from scgo.utils.comparators import (
    _SORTED_DIST_FP_INFO_KEY,
    PureInteratomicDistanceComparator,
    _compute_sorted_dist_list,
    get_sorted_dist_list,
)
from scgo.utils.diversity_scorer import DiversityScorer

POSITIONS = [[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [1.25, 2.1, 0.0]]


# --- MIC flag behavior -------------------------------------------------------


def test_pure_comparator_mic_false_does_not_fold_under_pbc() -> None:
    """Periodic near-images must differ when mic=False and match when mic=True."""
    cell = [8.0, 8.0, 12.0]
    a1 = Atoms(
        "Pt2",
        positions=[[0.10, 0.0, 0.0], [7.90, 0.0, 0.0]],
        cell=cell,
        pbc=[True, True, False],
    )
    a2 = Atoms(
        "Pt2",
        positions=[[0.10, 0.0, 0.0], [-0.10, 0.0, 0.0]],
        cell=cell,
        pbc=[True, True, False],
    )
    no_mic = PureInteratomicDistanceComparator(n_top=2, mic=False)
    with_mic = PureInteratomicDistanceComparator(n_top=2, mic=True)
    assert bool(no_mic.looks_like(a1, a2)) is False
    assert bool(with_mic.looks_like(a1, a2)) is True

    fp_no = get_sorted_dist_list(a1, mic=False)
    fp_yes = get_sorted_dist_list(a1, mic=True)
    # Without MIC the long in-cell Pt–Pt distance remains ~7.8 Å.
    assert float(fp_no[78][0]) > 7.0
    assert float(fp_yes[78][0]) < 1.0


def test_resolve_structure_mic_gas_vs_surface() -> None:
    assert resolve_structure_mic("gas_cluster") is False
    assert resolve_structure_mic("gas_cluster_adsorbate") is False
    assert resolve_neb_mic("gas_cluster") is False
    assert resolve_neb_mic("gas_cluster_adsorbate") is False

    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    cfg_on = SurfaceSystemConfig(slab=slab)
    assert cfg_on.comparator_use_mic is True
    assert resolve_structure_mic("surface_cluster", cfg_on) is True
    assert resolve_neb_mic("surface_cluster") is True
    assert resolve_neb_mic("surface_cluster_adsorbate") is True

    cfg_off = SurfaceSystemConfig(slab=slab, comparator_use_mic=False)
    assert resolve_structure_mic("surface_cluster", cfg_off) is False


# --- Composition mismatch handling -------------------------------------------


def test_looks_like_false_for_same_elements_different_counts():
    """Au2Pt vs AuPt2: same element set, different counts -> not similar."""
    a1 = Atoms("Au2Pt", positions=POSITIONS)
    a2 = Atoms("AuPt2", positions=POSITIONS)
    comp = PureInteratomicDistanceComparator()

    assert comp.looks_like(a1, a2) is False

    cum_diff, max_diff = comp.get_differences(a1, a2)
    assert math.isinf(cum_diff)
    assert math.isinf(max_diff)


def test_looks_like_false_for_disjoint_elements():
    """Pt3 vs Au3 (disjoint elements) returns False instead of raising."""
    a1 = Atoms("Pt3", positions=POSITIONS)
    a2 = Atoms("Au3", positions=POSITIONS)
    comp = PureInteratomicDistanceComparator()

    assert comp.looks_like(a1, a2) is False


def test_same_composition_comparison_still_works():
    """Identical / distinct same-composition structures keep their verdicts."""
    a1 = Atoms("Au2Pt", positions=POSITIONS)
    same = a1.copy()
    different = Atoms(
        "Au2Pt", positions=[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [2.5, 4.3, 0.0]]
    )
    comp = PureInteratomicDistanceComparator()

    assert bool(comp.looks_like(a1, same)) is True
    assert bool(comp.looks_like(a1, different)) is False


def test_ga_style_population_dedup_does_not_crash():
    """GA dedup (count_looks_like) over a mixed-composition pool must not raise."""
    pool = [
        Atoms("Au2Pt", positions=POSITIONS),
        Atoms("AuPt2", positions=POSITIONS),
        Atoms("Pt3", positions=POSITIONS),
        Atoms("Au2Pt", positions=POSITIONS),
    ]
    for i, atoms in enumerate(pool):
        atoms.info["confid"] = i

    comp = PureInteratomicDistanceComparator()

    counts = [count_looks_like(a, pool, comp) for a in pool]

    # Only the two identical Au2Pt entries match each other.
    assert counts == [1, 0, 0, 1]


def test_diversity_scorer_handles_mixed_compositions():
    """Diversity scoring over mixed compositions stays finite and non-negative."""
    comp = PureInteratomicDistanceComparator()
    scorer = DiversityScorer(
        [Atoms("Au2Pt", positions=POSITIONS), Atoms("AuPt2", positions=POSITIONS)],
        comp,
    )

    score = scorer.score(Atoms("Pt3", positions=POSITIONS))
    assert math.isfinite(score)
    assert score >= 0.0


# --- Fingerprint helpers -----------------------------------------------------


def test_get_sorted_dist_list():
    """Test the get_sorted_dist_list function with a simple H2O molecule."""
    atoms = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    dist_dict = get_sorted_dist_list(atoms)
    assert 1 in dist_dict  # H
    assert 8 in dist_dict  # O
    assert len(dist_dict[1]) == 1  # H-H distance
    assert len(dist_dict[8]) == 0  # No O-O distance
    assert math.isclose(dist_dict[1][0], 1.0)


def test_comparator_identical_structures():
    """Test that identical structures are recognized as similar."""
    atoms1 = Atoms("Pt3", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    atoms2 = atoms1.copy()
    comp = PureInteratomicDistanceComparator()
    assert comp.looks_like(atoms1, atoms2)


def test_comparator_different_structures():
    """Test that different structures are recognized as dissimilar."""
    atoms1 = Atoms("Pt3", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    atoms2 = Atoms("Pt3", positions=[[0, 0, 0], [2, 0, 0], [0, 2, 0]])
    comp = PureInteratomicDistanceComparator()
    assert not comp.looks_like(atoms1, atoms2)


def test_comparator_different_composition_returns_false():
    """Different compositions are not similar (must not raise)."""
    atoms1 = Atoms("Pt3", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    atoms2 = Atoms("Au3", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    comp = PureInteratomicDistanceComparator()
    assert comp.looks_like(atoms1, atoms2) is False


def test_comparator_tolerance():
    """Test that tolerance parameter affects structure comparison."""
    atoms1 = Atoms("Pt2", positions=[[0, 0, 0], [0, 0, 2.5]])
    atoms2 = Atoms("Pt2", positions=[[0, 0, 0], [0, 0, 2.51]])
    # With default tolerance, they should look alike
    comp_default = PureInteratomicDistanceComparator()
    assert comp_default.looks_like(atoms1, atoms2)

    # With a very small tolerance, they should not look alike
    comp_strict = PureInteratomicDistanceComparator(tol=0.001)
    assert not comp_strict.looks_like(atoms1, atoms2)


def test_sorted_dist_list_cache_hit_on_repeated_looks_like():
    """Repeated looks_like / get_sorted_dist_list should populate and reuse cache."""
    atoms1 = Atoms("Pt3", positions=[[0, 0, 0], [2.5, 0, 0], [1.25, 2.1, 0]])
    atoms2 = atoms1.copy()
    atoms2.positions[2, 1] += 1e-4

    first = get_sorted_dist_list(atoms1)
    assert _SORTED_DIST_FP_INFO_KEY in atoms1.info
    cached_id = id(atoms1.info[_SORTED_DIST_FP_INFO_KEY]["pair_cor"])
    second = get_sorted_dist_list(atoms1)
    assert second is first
    assert id(second) == cached_id

    comp = PureInteratomicDistanceComparator()
    assert comp.looks_like(atoms1, atoms2)
    assert comp.looks_like(atoms1, atoms2)
    assert _SORTED_DIST_FP_INFO_KEY in atoms1.info
    assert _SORTED_DIST_FP_INFO_KEY in atoms2.info

    # Position change invalidates cache
    atoms1.positions[0, 0] += 0.5
    refreshed = get_sorted_dist_list(atoms1)
    assert refreshed is not first
    assert atoms1.info[_SORTED_DIST_FP_INFO_KEY]["pair_cor"] is refreshed
    assert get_sorted_dist_list(atoms1) is refreshed


def test_sorted_dist_list_mic_matches_nested_get_distance():
    """MIC fingerprint via get_all_distances agrees with nested get_distance."""
    atoms = Atoms(
        "Cu4",
        positions=[
            [0.0, 0.0, 0.0],
            [2.5, 0.0, 0.0],
            [0.0, 2.5, 0.0],
            [2.5, 2.5, 0.0],
        ],
        cell=[5.0, 5.0, 20.0],
        pbc=[True, True, False],
    )
    vectorized = get_sorted_dist_list(atoms, mic=True)

    # Reference: nested ASE get_distance (legacy path)
    numbers = atoms.numbers
    ref: dict[int, list] = {}
    for n in set(numbers):
        i_un = [i for i, z in enumerate(numbers) if z == n]
        d = [
            atoms.get_distance(n1, n2, mic=True)
            for i, n1 in enumerate(i_un)
            for n2 in i_un[i + 1 :]
        ]
        d.sort()
        ref[n] = d

    assert set(vectorized) == set(ref)
    for n in ref:
        assert np.allclose(vectorized[n], ref[n], atol=1e-10)

    # Non-MIC gas-phase fingerprint still works for the same geometry without PBC
    gas = atoms.copy()
    gas.set_pbc(False)
    gas_fp = _compute_sorted_dist_list(gas, mic=False)
    assert 29 in gas_fp
    assert len(gas_fp[29]) == 6
