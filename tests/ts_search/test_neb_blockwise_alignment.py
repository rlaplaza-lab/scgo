"""NEB: blockwise endpoint matching (slab + core + adsorbate)."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111

from scgo.exceptions import SCGOValidationError
from scgo.ts_search.transition_state import (
    _align_endpoints_blockwise,
    _align_product_kabsch_to_reactant,
    interpolate_path,
    validate_initial_neb_path,
)


def test_blockwise_reorders_adsorbate_block_to_reactant() -> None:
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.2, 0.0],
            [2.0, 0.0, 0.1],
            [1.2, 0.7, 0.3],
        ],
    )
    react = Atoms(symbols=["Pt", "Pt", "O", "H"], positions=pos, pbc=False)
    prod = Atoms(
        symbols=["Pt", "Pt", "H", "O"],
        positions=np.vstack([pos[:2], pos[3:4], pos[2:3]]),
        pbc=False,
    )
    _align_endpoints_blockwise(react, prod, n_slab=1, n_core=1, n_ads=2)
    np.testing.assert_array_almost_equal(
        prod.get_positions()[2:4], react.get_positions()[2:4]
    )


def test_interpolate_path_accepts_block_dims_for_gas_adsorbate() -> None:
    sym = ["Pt", "Pt", "H"]
    pos = np.random.default_rng(0).random((3, 3))
    a1 = Atoms(symbols=sym, positions=pos, pbc=False, cell=[20, 20, 20])
    a2 = Atoms(symbols=sym, positions=pos.copy(), pbc=False, cell=[20, 20, 20])
    out = interpolate_path(
        a1,
        a2,
        n_images=2,
        method="linear",
        mic=False,
        align_endpoints=True,
        system_type="gas_cluster_adsorbate",
        n_slab=0,
        n_core_mobile=2,
        n_adsorbate_mobile=1,
    )
    assert len(out) == 2 + 2
    assert len(out[0]) == 3 and len(out[-1]) == 3


def test_interpolate_path_blockwise_mic_on_periodic_surface() -> None:
    """Blockwise matching + MIC on slab/core/adsorbate under in-plane PBC."""
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    slab.pbc = [True, True, False]
    n_slab = len(slab)
    z0 = slab.get_positions()[:, 2].max() + 1.5

    core_pos = np.array([[0.5, 0.5, z0], [1.5, 0.6, z0]])
    ads_pos = np.array([[1.0, 1.2, z0 + 0.2], [1.1, 1.3, z0 + 0.9]])
    react = slab.copy() + Atoms(
        symbols=["Pt", "Pt", "O", "H"], positions=np.vstack([core_pos, ads_pos])
    )
    prod_ads = ads_pos[[1, 0]]
    prod_core = core_pos + np.array([slab.cell[0, 0] - 0.1, 0.0, 0.0])
    prod = slab.copy() + Atoms(
        symbols=["Pt", "Pt", "H", "O"],
        positions=np.vstack([prod_core, prod_ads]),
    )

    images = interpolate_path(
        react,
        prod,
        n_images=2,
        method="linear",
        mic=True,
        align_endpoints=True,
        system_type="surface_cluster_adsorbate",
        n_slab=n_slab,
        n_core_mobile=2,
        n_adsorbate_mobile=2,
    )

    disp = images[-1].get_positions() - images[0].get_positions()
    assert float(np.max(np.linalg.norm(disp[:n_slab], axis=1))) < 1e-2
    mobile_disp = np.linalg.norm(disp[n_slab:], axis=1)
    assert float(np.max(mobile_disp)) < 0.25
    rms = float(np.sqrt(np.mean(mobile_disp**2)))
    assert rms < 0.15


def test_fragment_wise_matching_swaps_crossed_oh() -> None:
    """Two OH fragments crossed on product are restored by COM fragment matching."""
    core = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.7, 0.0]])
    oh1 = np.array([[0.0, 0.0, 1.5], [0.0, 0.0, 2.46]])
    oh2 = np.array([[2.0, 0.0, 1.5], [2.0, 0.0, 2.46]])
    react = Atoms(
        symbols=["Pt", "Pt", "Pt", "O", "H", "O", "H"],
        positions=np.vstack([core, oh1, oh2]),
        pbc=False,
    )
    # Product: same core, but OH fragments swapped (and H/O order swapped in frag0).
    prod = Atoms(
        symbols=["Pt", "Pt", "Pt", "H", "O", "H", "O"],
        positions=np.vstack([core, oh2[[1, 0]], oh1[[1, 0]]]),
        pbc=False,
    )
    _align_endpoints_blockwise(
        react,
        prod,
        n_slab=0,
        n_core=3,
        n_ads=4,
        adsorbate_fragment_lengths=[2, 2],
    )
    np.testing.assert_allclose(prod.get_positions()[3:5], oh1, atol=1e-8)
    np.testing.assert_allclose(prod.get_positions()[5:7], oh2, atol=1e-8)
    assert list(prod.numbers[3:7]) == list(react.numbers[3:7])


def test_core_anchored_kabsch_ignores_adsorbate_drag() -> None:
    """Kabsch fit on core should not be pulled by a large adsorbate hop."""
    react = Atoms(
        symbols=["Pt", "Pt", "O", "H"],
        positions=[
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.96],
        ],
        pbc=False,
    )
    # Product: core slightly rotated/translated; OH hopped far away.
    prod_pos = np.array(
        [
            [0.1, 0.05, 0.0],
            [2.05, -0.05, 0.0],
            [4.5, 3.0, 2.0],
            [4.5, 3.0, 2.96],
        ]
    )
    aligned_full = _align_product_kabsch_to_reactant(
        react, prod_pos, n_slab=0, n_core_mobile=None
    )
    aligned_core = _align_product_kabsch_to_reactant(
        react, prod_pos, n_slab=0, n_core_mobile=2
    )
    core_rms_full = float(
        np.sqrt(np.mean(np.sum((aligned_full[:2] - react.positions[:2]) ** 2, axis=1)))
    )
    core_rms_core = float(
        np.sqrt(np.mean(np.sum((aligned_core[:2] - react.positions[:2]) ** 2, axis=1)))
    )
    assert core_rms_core <= core_rms_full + 1e-9
    assert core_rms_core < 0.15


def test_validate_initial_neb_path_rejects_clash() -> None:
    # Three images so the middle one is treated as an interior clash check.
    a = Atoms("H2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    mid = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]])
    b = a.copy()
    with pytest.raises(SCGOValidationError, match="clashing/discontinuous"):
        validate_initial_neb_path(
            [a, mid, b], max_endpoint_mismatch=1.25, clash_distance=0.7
        )


def test_validate_initial_neb_path_rejects_huge_residual() -> None:
    a = Atoms("Pt2", positions=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    b = Atoms("Pt2", positions=[[8.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    with pytest.raises(SCGOValidationError, match="clashing/discontinuous"):
        validate_initial_neb_path([a, b], max_endpoint_mismatch=1.25)


def test_validate_initial_neb_path_noop_without_mismatch_gate() -> None:
    a = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]])
    validate_initial_neb_path([a, a.copy()], max_endpoint_mismatch=None)


def test_validate_initial_neb_energy_profile_rejects_huge_barrier() -> None:
    from scgo.ts_search.transition_state import validate_initial_neb_energy_profile

    with pytest.raises(SCGOValidationError, match="discontinuous"):
        validate_initial_neb_energy_profile(
            [0.0, 10.0, 20.0, 0.2], max_spurious_barrier=8.0
        )


def test_validate_initial_neb_energy_profile_allows_endpoint_max() -> None:
    from scgo.ts_search.transition_state import validate_initial_neb_energy_profile

    # Adsorbate OH hops often start endpoint-max on IDPP; climb can still succeed.
    validate_initial_neb_energy_profile([0.0, 0.2, 0.5, 1.0], max_spurious_barrier=8.0)


def test_validate_initial_neb_energy_profile_accepts_modest_barrier() -> None:
    from scgo.ts_search.transition_state import validate_initial_neb_energy_profile

    validate_initial_neb_energy_profile([0.0, 0.5, 1.1, 0.2], max_spurious_barrier=8.0)


def test_idpp_band_optimization_priority_prefers_robust_interior() -> None:
    from scgo.ts_search.transition_state import idpp_band_optimization_priority

    robust = idpp_band_optimization_priority([0.0, 0.5, 1.2, 0.2])
    endpoint = idpp_band_optimization_priority([0.0, 0.2, 0.5, 1.0])
    soft = idpp_band_optimization_priority([0.0, 0.35, 0.45, 0.4])
    assert robust[0] == 2
    assert endpoint[0] == 1
    assert soft[0] == 0
    assert robust > endpoint > soft


def test_neb_uses_two_stage_climb_skips_soft_interior_barriers() -> None:
    from scgo.ts_search.transition_state import neb_uses_two_stage_climb

    assert (
        neb_uses_two_stage_climb(True, 100, initial_energies=[0.0, 0.2, 0.5, 1.0])
        is False
    )
    assert (
        neb_uses_two_stage_climb(True, 100, initial_energies=[0.0, 0.4, 0.9, 0.2])
        is False
    )
    assert (
        neb_uses_two_stage_climb(True, 100, initial_energies=[0.0, 0.5, 1.2, 0.2])
        is True
    )


def test_validate_initial_neb_energy_profile_rejects_endpoint_drift() -> None:
    from scgo.ts_search.transition_state import validate_initial_neb_energy_profile

    with pytest.raises(SCGOValidationError, match="product energy drifted"):
        validate_initial_neb_energy_profile(
            [0.0, 0.5, 1.0, 0.2],
            reference_reactant_energy=0.0,
            reference_product_energy=-6.0,
            max_endpoint_energy_drift=0.5,
        )


def test_validate_initial_neb_energy_profile_rejects_one_sided_slide() -> None:
    from scgo.ts_search.transition_state import validate_initial_neb_energy_profile

    # Interior max only 0.1 eV above the higher endpoint.
    with pytest.raises(SCGOValidationError, match="prominence"):
        validate_initial_neb_energy_profile(
            [0.0, 0.3, 0.5, 0.4],
            reference_reactant_energy=0.0,
            reference_product_energy=0.4,
            min_saddle_prominence=0.40,
        )


def test_copy_atoms_isolates_nested_info_from_metadata_writes() -> None:
    from scgo.metadata.atoms import set_tags
    from scgo.utils.helpers import copy_atoms, extract_energy_from_atoms

    src = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    src.info["key_value_pairs"] = {"raw_score": 1.0}
    clone = copy_atoms(src)
    set_tags(clone, raw_score=-9.0, potential_energy=9.0)
    assert extract_energy_from_atoms(src) == pytest.approx(-1.0)
    assert extract_energy_from_atoms(clone) == pytest.approx(9.0)
    # ASE Atoms.copy() alone would have shared the nested dicts.
    shallow = src.copy()
    set_tags(shallow, raw_score=-3.0)
    assert extract_energy_from_atoms(src) == pytest.approx(3.0)
