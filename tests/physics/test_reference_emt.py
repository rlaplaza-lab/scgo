"""Fast EMT reference tests with stringent geometry and barrier assertions."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT
from ase.optimize import LBFGS
from numpy.random import default_rng

from scgo.cluster_adsorbate import ClusterAdsorbateConfig, place_fragment_on_cluster
from scgo.cluster_adsorbate.placement import radii_derived_height_bounds
from scgo.initialization import create_initial_cluster
from scgo.ts_search.transition_state import interpolate_path
from scgo.utils.helpers import perform_local_relaxation
from tests.constants import EMT_PT2_BOND_ANG, EMT_PT2_BOND_TOL_ANG
from tests.test_utils import assert_nn_distances_in_band


def _oh_template() -> Atoms:
    return Atoms("OH", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.96]], pbc=False)


def _pt3_triangle() -> Atoms:
    return Atoms(
        "Pt3",
        positions=[
            [0.0, 0.0, 0.0],
            [2.5, 0.0, 0.0],
            [1.25, 2.165, 0.0],
        ],
        pbc=False,
    )


def test_pt2_bond_length_at_emt_minimum() -> None:
    atoms = Atoms("Pt2", positions=[[0, 0, 0], [EMT_PT2_BOND_ANG, 0, 0]])
    atoms.calc = EMT()
    e0 = atoms.get_potential_energy()
    energy = perform_local_relaxation(atoms, EMT(), LBFGS, fmax=0.001, steps=20)
    final_bond = atoms.get_distance(0, 1)
    assert final_bond == pytest.approx(EMT_PT2_BOND_ANG, abs=EMT_PT2_BOND_TOL_ANG)
    assert energy <= e0 + 1e-6
    assert np.isfinite(energy)


def test_pt3_nn_distances_after_init(rng) -> None:
    atoms = create_initial_cluster(["Pt", "Pt", "Pt"], rng=rng)
    assert_nn_distances_in_band(atoms)


def test_oh_placement_height_within_bounds() -> None:
    core = _pt3_triangle()
    frag = _oh_template()
    cfg = ClusterAdsorbateConfig(max_placement_attempts=200)
    rng = default_rng(42)
    metadata: dict[str, str] = {}
    placed = place_fragment_on_cluster(
        core,
        frag,
        rng,
        cfg,
        anchor_index=0,
        bond_axis=(0, 1),
        placement_metadata=metadata,
    )
    assert placed is not None
    assert metadata.get("site_type") is not None
    h_min, h_max = radii_derived_height_bounds(frag, core, anchor_index=0)
    combined = core + placed
    o_idx = len(core)
    min_pt_o = min(combined.get_distance(o_idx, i) for i in range(len(core)))
    assert h_min - 0.15 <= min_pt_o <= h_max + 0.15


@pytest.mark.slow
def test_misaligned_vs_aligned_neb_max_energy(h2_reactant, h2_product) -> None:
    """Aligned interpolation should yield a lower maximum image energy than misaligned."""
    calc = EMT()
    aligned = interpolate_path(
        h2_reactant,
        h2_product,
        n_images=5,
        method="idpp",
        align_endpoints=True,
    )
    misaligned = interpolate_path(
        h2_reactant,
        h2_product,
        n_images=5,
        method="idpp",
        align_endpoints=False,
    )
    for img in aligned + misaligned:
        img.calc = calc

    def _max_energy(images: list[Atoms]) -> float:
        return max(img.get_potential_energy() for img in images)

    assert _max_energy(aligned) <= _max_energy(misaligned) + 0.05

