"""Regression tests: composition mismatch must not raise in the comparator.

``PureInteratomicDistanceComparator.looks_like`` used to raise
``SCGOValidationError`` for structures with different compositions, which
crashed GA population dedup on mixed-composition pools. Different compositions
are simply *not* similar, so the comparator now reports a maximal difference.
"""

from __future__ import annotations

import math

from ase import Atoms
from ase.build import fcc111

from scgo.surface.config import SurfaceSystemConfig
from scgo.system_types import resolve_neb_mic, resolve_structure_mic
from scgo.utils.comparators import (
    PureInteratomicDistanceComparator,
    get_sorted_dist_list,
)

POSITIONS = [[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [1.25, 2.1, 0.0]]


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
    from scgo.ase_ga_patches.population import count_looks_like

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
    from scgo.utils.diversity_scorer import DiversityScorer

    comp = PureInteratomicDistanceComparator()
    scorer = DiversityScorer(
        [Atoms("Au2Pt", positions=POSITIONS), Atoms("AuPt2", positions=POSITIONS)],
        comp,
    )

    score = scorer.score(Atoms("Pt3", positions=POSITIONS))
    assert math.isfinite(score)
    assert score >= 0.0


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
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    cfg_on = SurfaceSystemConfig(slab=slab)
    assert cfg_on.comparator_use_mic is True
    assert resolve_structure_mic("surface_cluster", cfg_on) is True
    assert resolve_neb_mic("surface_cluster") is True
    assert resolve_neb_mic("surface_cluster_adsorbate") is True

    cfg_off = SurfaceSystemConfig(slab=slab, comparator_use_mic=False)
    assert resolve_structure_mic("surface_cluster", cfg_off) is False
