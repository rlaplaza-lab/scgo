"""Unified connectivity-policy validation gates (consolidation + correctness).

Covers the single :func:`~scgo.system_types.validate_connectivity_policy`
entry point, the gateway factor resolution, the MIC clash fix, the
``simple_go`` config-factor path, and the TS/surface gate parity.
"""

from __future__ import annotations

import pytest
from ase import Atoms
from ase.build import fcc111
from ase.calculators.emt import EMT

from scgo.cluster_adsorbate import ClusterAdsorbateConfig
from scgo.exceptions import SCGOValidationError
from scgo.surface import make_surface_config
from scgo.surface.deposition import combine_slab_adsorbate, slab_surface_extreme
from scgo.system_types import (
    AdsorbateDefinition,
    resolve_connectivity_factor,
    validate_connectivity_policy,
    validate_minimum_structure,
    validate_structure_for_system_type,
)
from scgo.ts_search.transition_state_run import _apply_surface_ts_geometry_gate


def _pt_slab() -> Atoms:
    return fcc111("Pt", size=(2, 2, 2), vacuum=6.0, orthogonal=True)


def _pt_slab_large() -> Atoms:
    # Larger cell so the minimum-image convention (slab pbc) does not merge two
    # well-separated mobile atoms into one connected component.
    return fcc111("Pt", size=(4, 4, 2), vacuum=6.0, orthogonal=True)


# --- Task 2: factor resolution precedence ------------------------------------


def test_resolve_connectivity_factor_precedence() -> None:
    ca = ClusterAdsorbateConfig(structure_connectivity_factor=2.5)
    surf = make_surface_config(_pt_slab())

    # Explicit value wins over both configs.
    assert (
        resolve_connectivity_factor(
            1.1, cluster_adsorbate_config=ca, surface_config=surf
        )
        == 1.1
    )
    # cluster_adsorbate_config beats surface_config.
    assert (
        resolve_connectivity_factor(
            None, cluster_adsorbate_config=ca, surface_config=surf
        )
        == 2.5
    )
    # surface_config beats module default.
    assert resolve_connectivity_factor(
        None, cluster_adsorbate_config=None, surface_config=surf
    ) == float(surf.structure_connectivity_factor)
    # module default when nothing is set.
    assert (
        resolve_connectivity_factor(
            None, cluster_adsorbate_config=None, surface_config=None
        )
        == 1.4
    )


# --- Task 1: unified policy across system types -----------------------------


def test_validate_connectivity_policy_gas_single_component() -> None:
    connected = Atoms(
        "Pt2", positions=[[0, 0, 0], [2.5, 0, 0]], cell=[20, 20, 20], pbc=False
    )
    fragmented = Atoms(
        "Pt2", positions=[[0, 0, 0], [8.0, 0, 0]], cell=[20, 20, 20], pbc=False
    )
    assert validate_connectivity_policy(
        connected,
        uses_surface=False,
        n_slab=0,
        n_core_mobile=2,
        connectivity_factor=1.4,
        use_mic=False,
    )[0]
    ok, msg = validate_connectivity_policy(
        fragmented,
        uses_surface=False,
        n_slab=0,
        n_core_mobile=2,
        connectivity_factor=1.4,
        use_mic=False,
    )
    assert not ok
    assert "not connected" in msg


def test_validate_connectivity_policy_surface_entry_point() -> None:
    """Pin :func:`validate_connectivity_policy` directly (surface path)."""
    slab = _pt_slab_large()
    n_slab = len(slab)
    z_top = slab_surface_extreme(slab, 2, upper=True)

    def _combined(positions: list[list[float]]) -> Atoms:
        mobile = Atoms(
            symbols=["Pt"] * len(positions),
            positions=positions,
            cell=slab.cell,
            pbc=slab.pbc,
        )
        return combine_slab_adsorbate(slab, mobile)

    # Two disconnected, slab-bound subgroups (each internally bonded) accepted
    # when fragmentation is allowed.
    ok, _ = validate_connectivity_policy(
        _combined(
            [
                [0.0, 0.0, z_top + 2.0],
                [0.0, 2.0, z_top + 2.0],
                [6.0, 5.0, z_top + 2.0],
                [6.0, 7.0, z_top + 2.0],
            ]
        ),
        uses_surface=True,
        n_slab=n_slab,
        n_core_mobile=4,
        connectivity_factor=1.4,
        use_mic=False,
        allow_cluster_fragmentation=True,
    )
    assert ok

    # Same structure without fragmentation -> rejected (single component).
    ok, msg = validate_connectivity_policy(
        _combined(
            [
                [0.0, 0.0, z_top + 2.0],
                [0.0, 2.0, z_top + 2.0],
                [6.0, 5.0, z_top + 2.0],
                [6.0, 7.0, z_top + 2.0],
            ]
        ),
        uses_surface=True,
        n_slab=n_slab,
        n_core_mobile=4,
        connectivity_factor=1.4,
        use_mic=False,
    )
    assert not ok
    assert "not connected" in msg


def test_bare_surface_has_no_connectivity_gate() -> None:
    """Bare ``surface`` skips the per-minimum connectivity gate entirely."""
    slab = _pt_slab()
    z_top = slab_surface_extreme(slab, 2, upper=True)
    # Disconnected mobile (would fail the surface connectivity policy) — but the
    # bare ``surface`` type never invokes it.
    mobile = Atoms(
        symbols=["Pt", "Pt"],
        positions=[[0.0, 0.0, z_top + 2.0], [5.0, 5.0, z_top + 2.0]],
        cell=slab.cell,
        pbc=slab.pbc,
    )
    combined = combine_slab_adsorbate(slab, mobile)
    surf = make_surface_config(slab)
    validate_structure_for_system_type(
        combined, system_type="surface", surface_config=surf
    )


# --- Task 2: gateway honors ClusterAdsorbateConfig ---------------------------


def test_gateway_honors_cluster_adsorbate_config() -> None:
    atoms = Atoms(
        "Pt2", positions=[[0, 0, 0], [4.0, 0, 0]], cell=[20, 20, 20], pbc=False
    )
    # Default factor (1.4) -> threshold 3.58 < 4.0 -> disconnected -> reject.
    with pytest.raises(SCGOValidationError):
        validate_structure_for_system_type(atoms, system_type="gas_cluster")
    # High factor via ClusterAdsorbateConfig -> connected -> accept.
    cfg = ClusterAdsorbateConfig(structure_connectivity_factor=3.0)
    validate_structure_for_system_type(
        atoms, system_type="gas_cluster", cluster_adsorbate_config=cfg
    )


# --- Task 2 (gas/adsorbate): clash + whole-region connectivity ---------------


def test_gas_adsorbate_requires_whole_region_connectivity() -> None:
    core = Atoms(
        "Pt2", positions=[[0, 0, 0], [2.5, 0, 0]], cell=[20, 20, 20], pbc=False
    )
    fragment = Atoms("O", positions=[[15.0, 0, 0]], cell=[20, 20, 20], pbc=False)
    combined = core + fragment
    with pytest.raises(SCGOValidationError):
        validate_structure_for_system_type(
            combined, system_type="gas_cluster_adsorbate"
        )


def test_gas_adsorbate_rejects_clashing_minimum() -> None:
    """Overlapping atoms must still be rejected at the gateway (clash sibling)."""
    clashing = Atoms(
        "Pt2",
        positions=[[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]],
        cell=[20.0, 20.0, 20.0],
        pbc=False,
    )
    with pytest.raises(SCGOValidationError):
        validate_structure_for_system_type(clashing, system_type="gas_cluster")


# --- Task 4: simple_go honors per-config factor ------------------------------


def test_simple_go_honors_config_factor(tmp_path, rng) -> None:
    from scgo.algorithms.simple_go import simple_go

    atoms = Atoms(
        "Pt2",
        positions=[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        cell=[20.0, 20.0, 20.0],
        pbc=False,
    )
    atoms.calc = EMT()

    # Default factor rejects the fragmented (4.0 A apart) dimer.
    strict = simple_go(
        atoms,
        str(tmp_path / "strict"),
        rng=rng,
        niter_local_relaxation=1,
        verbosity=0,
        system_type="gas_cluster",
    )
    assert strict == []

    # A permissive ClusterAdsorbateConfig factor connects it.
    permissive = ClusterAdsorbateConfig(structure_connectivity_factor=3.0)
    loose = simple_go(
        atoms,
        str(tmp_path / "loose"),
        rng=rng,
        niter_local_relaxation=1,
        verbosity=0,
        system_type="gas_cluster",
        cluster_adsorbate_config=permissive,
    )
    assert loose != []


# --- Task 7: MIC clash + connectivity on surface -----------------------------


def test_surface_validation_mic_true() -> None:
    cell = [10.0, 10.0, 20.0]
    slab = Atoms(
        "Pt2", positions=[[0.0, 0, 0], [1.0, 0, 0]], cell=cell, pbc=[True, True, False]
    )
    n_slab = len(slab)
    z_top = slab_surface_extreme(slab, 2, upper=True)

    def _combined(x0: float, x1: float) -> Atoms:
        mobile = Atoms(
            "Pt2",
            positions=[[x0, 0, z_top + 1.5], [x1, 0, z_top + 1.5]],
            cell=cell,
            pbc=[True, True, False],
        )
        return combine_slab_adsorbate(slab, mobile)

    # Real-space 8.8 A apart, 1.2 A apart through the x-periodic image.
    combined = _combined(0.5, 9.3)
    # use_mic=False -> genuinely disconnected -> rejected.
    ok, _ = validate_supported_cluster_deposit_wrapped(combined, n_slab, use_mic=False)
    assert not ok
    # use_mic=True -> bonded through the periodic image and slab-bound -> accepted.
    ok, _ = validate_supported_cluster_deposit_wrapped(combined, n_slab, use_mic=True)
    assert ok


def validate_supported_cluster_deposit_wrapped(
    combined: Atoms, n_slab: int, *, use_mic: bool
) -> tuple[bool, str]:
    from scgo.surface.validation import validate_supported_cluster_deposit

    return validate_supported_cluster_deposit(
        combined, n_slab, surface_normal_axis=2, use_mic=use_mic
    )


# --- Task 9: TS gate matches the GO gate -------------------------------------


def _surface_adsorbate_valid() -> tuple[Atoms, object]:
    slab = _pt_slab()
    z_top = slab_surface_extreme(slab, 2, upper=True)
    mobile = Atoms(
        "Pt", positions=[[0.0, 0.0, z_top + 2.0]], cell=slab.cell, pbc=slab.pbc
    )
    combined = combine_slab_adsorbate(slab, mobile)
    surf = make_surface_config(slab)
    return combined, surf


def test_ts_gate_matches_go_gate() -> None:
    combined, surf = _surface_adsorbate_valid()
    ads_def = AdsorbateDefinition(core_symbols=[], adsorbate_symbols=["Pt"])

    # GO path accepts it.
    validate_minimum_structure(
        combined,
        system_type="surface_adsorbate",
        surface_config=surf,
        adsorbate_definition=ads_def,
    )

    # TS path routes through the same gate -> stays "success".
    results = [
        {
            "status": "success",
            "reactant_structure": combined,
            "product_structure": combined,
            "transition_state": combined,
        }
    ]
    _apply_surface_ts_geometry_gate(
        results,
        surface_config=surf,
        system_type="surface_adsorbate",
        adsorbate_definition=ads_def,
    )
    assert results[0]["status"] == "success"


def test_ts_gate_parity_honors_cluster_adsorbate_config() -> None:
    """TS gate re-resolves the factor from ClusterAdsorbateConfig (parity with GO)."""
    slab = _pt_slab_large()
    z_top = slab_surface_extreme(slab, 2, upper=True)
    surf = make_surface_config(slab)
    ads_def = AdsorbateDefinition(core_symbols=[], adsorbate_symbols=["Pt", "Pt"])

    # Two separate Pt atoms ~5.0 A apart, each slab-bound. With MIC on the (large)
    # slab the in-plane image distance stays ~5.0 A, so at the default factor 1.4
    # the Pt-Pt threshold (~3.8 A) is below 5.0 A -> two components -> rejected by
    # both GO and TS gates; at factor 3.0 (~8.2 A) -> one component -> accepted.
    def _two_pt(separation: float) -> Atoms:
        mobile = Atoms(
            "Pt2",
            positions=[
                [0.0, 0.0, z_top + 2.0],
                [separation, 0.0, z_top + 2.0],
            ],
            cell=slab.cell,
            pbc=slab.pbc,
        )
        return combine_slab_adsorbate(slab, mobile)

    tight = _two_pt(5.0)
    assert not validate_connectivity_policy(  # sanity: two components
        tight,
        uses_surface=True,
        n_slab=len(slab),
        n_core_mobile=2,
        connectivity_factor=1.4,
        use_mic=False,
    )[0]

    # GO acceptance under the permissive config.
    permissive = ClusterAdsorbateConfig(structure_connectivity_factor=3.0)
    validate_minimum_structure(
        tight,
        system_type="surface_adsorbate",
        surface_config=surf,
        adsorbate_definition=ads_def,
        cluster_adsorbate_config=permissive,
    )

    # TS gate, called WITHOUT an explicit connectivity_factor so the config is
    # actually exercised (gateway precedence: explicit float wins over config).
    results = [
        {
            "status": "success",
            "reactant_structure": tight,
            "product_structure": tight,
            "transition_state": tight,
        }
    ]
    _apply_surface_ts_geometry_gate(
        results,
        surface_config=surf,
        system_type="surface_adsorbate",
        adsorbate_definition=ads_def,
        cluster_adsorbate_config=permissive,
    )
    assert results[0]["status"] == "success"

    # The default factor (no config) must still reject the same structure.
    rejected = [
        {
            "status": "success",
            "reactant_structure": tight,
            "product_structure": tight,
            "transition_state": tight,
        }
    ]
    _apply_surface_ts_geometry_gate(
        rejected,
        surface_config=surf,
        system_type="surface_adsorbate",
        adsorbate_definition=ads_def,
    )
    assert rejected[0]["status"] == "failed"


# --- Task 1: only ONE single-component message author ------------------------


def test_dead_duplicate_removed() -> None:
    """``validate_connectivity_policy`` is the sole connected-components author.

    ``_validate_mobile_connectivity_policy`` must be gone while
    ``validate_combined_cluster_structure`` (the init/public seam) survives.
    """
    from scgo import validate_combined_cluster_structure as exported
    from scgo.cluster_adsorbate.validation import validate_combined_cluster_structure

    assert exported is validate_combined_cluster_structure
    with pytest.raises(ImportError):
        from scgo.surface.validation import (
            _validate_mobile_connectivity_policy,  # noqa: F401
        )
