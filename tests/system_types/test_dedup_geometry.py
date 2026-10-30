"""Type-aware uniqueness geometry resolution tests."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111

from scgo.constants import (
    DEFAULT_COMPARATOR_TOL,
    DEFAULT_PAIR_COR_MAX,
    DEFAULT_SUPPORTED_SLAB_WEIGHT,
    SUPPORTED_CLUSTER_PAIR_COR_MAX,
)
from scgo.exceptions import SCGOValidationError
from scgo.surface.config import SurfaceSystemConfig
from scgo.surface.partition import prepare_slab_search_surface_config
from scgo.system_types import AdsorbateDefinition
from scgo.system_types.dedup_geometry import resolve_uniqueness_geometry
from scgo.utils.comparators import UniquenessSettings


def _layered_slab(n_per_layer: int = 2, n_layers: int = 3) -> Atoms:
    pos = np.zeros((n_per_layer * n_layers, 3))
    symbols: list[str] = []
    for layer in range(n_layers):
        for j in range(n_per_layer):
            idx = layer * n_per_layer + j
            pos[idx, 0] = j * 1.5
            pos[idx, 2] = float(layer)
            symbols.append("C")
    return Atoms(
        symbols=symbols,
        positions=pos,
        cell=[8, 8, 12],
        pbc=[True, True, False],
    )


def _ads(n_ads: int = 2) -> AdsorbateDefinition:
    return AdsorbateDefinition(
        core_symbols=[],
        adsorbate_symbols=["O"] * n_ads,
        adsorbate_fragment_lengths=[n_ads],
    )


# --- Slab-as-target types ------------------------------------------------------


def test_surface_adsorbate_full_slab_weight_and_generic_gates() -> None:
    cfg, part = prepare_slab_search_surface_config(
        SurfaceSystemConfig(
            slab=_layered_slab(), fix_all_slab_atoms=False, n_relax_top_slab_layers=1
        )
    )
    geo = resolve_uniqueness_geometry(
        system_type="surface_adsorbate",
        n_atoms=len(cfg.slab) + 2,
        surface_config=cfg,
        adsorbate_definition=_ads(2),
    )
    assert geo.blocks is not None
    roles = [b.role for b in geo.blocks.blocks]
    assert roles == ["mobile_slab", "adsorbate"]
    assert len(geo.blocks.blocks[0].indices) == part.n_mobile_slab
    # Top layers are the region of interest: full weight, generic gates.
    assert geo.component_weights["mobile_slab"] == 1.0
    assert geo.settings.comparator_tol == DEFAULT_COMPARATOR_TOL
    assert geo.settings.comparator_pair_cor_max == DEFAULT_PAIR_COR_MAX


def test_bare_surface_type_has_single_role_no_blocks() -> None:
    cfg, _part = prepare_slab_search_surface_config(
        SurfaceSystemConfig(
            slab=_layered_slab(), fix_all_slab_atoms=False, n_relax_top_slab_layers=1
        )
    )
    geo = resolve_uniqueness_geometry(
        system_type="surface",
        n_atoms=len(cfg.slab),
        surface_config=cfg,
    )
    assert geo.blocks is None


# --- Supported-deposit types ----------------------------------------------------


def test_supported_deposit_includes_relaxed_support_at_low_weight() -> None:
    cfg = SurfaceSystemConfig(
        slab=_layered_slab(),
        fix_all_slab_atoms=False,
        n_relax_top_slab_layers=1,
    )
    n_total = len(cfg.slab) + 4 + 2
    geo = resolve_uniqueness_geometry(
        system_type="surface_cluster_adsorbate",
        n_atoms=n_total,
        surface_config=cfg,
        adsorbate_definition=_ads(2),
    )
    assert geo.blocks is not None
    roles = [b.role for b in geo.blocks.blocks]
    assert roles == ["mobile_slab", "deposit", "adsorbate"]
    assert len(geo.blocks.blocks[0].indices) == 2  # top layer only
    assert geo.component_weights["mobile_slab"] == pytest.approx(
        DEFAULT_SUPPORTED_SLAB_WEIGHT
    )
    # Shared cumulative tolerance; tighter max-distance gate for supported clusters.
    assert geo.settings.comparator_tol == DEFAULT_COMPARATOR_TOL
    assert geo.settings.comparator_pair_cor_max == SUPPORTED_CLUSTER_PAIR_COR_MAX


def test_frozen_support_included_for_registry_and_pair_cor_tightens() -> None:
    cfg = SurfaceSystemConfig(slab=_layered_slab())  # fix_all_slab_atoms=True
    n_total = len(cfg.slab) + 4 + 2
    geo = resolve_uniqueness_geometry(
        system_type="surface_cluster_adsorbate",
        n_atoms=n_total,
        surface_config=cfg,
        adsorbate_definition=_ads(2),
    )
    assert geo.blocks is not None
    roles = [b.role for b in geo.blocks.blocks]
    assert roles == ["mobile_slab", "deposit", "adsorbate"]
    assert len(geo.blocks.blocks[0].indices) == len(cfg.slab)
    assert geo.component_weights["mobile_slab"] == pytest.approx(
        DEFAULT_SUPPORTED_SLAB_WEIGHT
    )
    assert geo.settings.comparator_tol == DEFAULT_COMPARATOR_TOL
    assert geo.settings.comparator_pair_cor_max == SUPPORTED_CLUSTER_PAIR_COR_MAX


def test_frozen_slab_matches_lattice_shift_and_rejects_other_sites() -> None:
    """Registry lives in slab–deposit distances, not in the deposit alone."""
    slab = fcc111("Pt", size=(4, 4, 2), vacuum=8.0, orthogonal=True)
    slab.pbc = [True, True, False]
    n_slab = len(slab)
    z = float(slab.positions[:, 2].max()) + 2.2
    # Pt–O, not Pt–Pt: sorted Pt–slab distances on a symmetric slab barely
    # change under rotation, while the unique O atom's registry does.
    dimer = np.array([[2.2, 1.5, z], [2.2, 3.4, z]], dtype=float)
    symbols = ["Pt"] * n_slab + ["Pt", "O"]

    def _placed(deposit: np.ndarray) -> Atoms:
        pos = np.vstack([slab.get_positions(), deposit])
        return Atoms(symbols=symbols, positions=pos, cell=slab.cell, pbc=slab.pbc)

    geo = resolve_uniqueness_geometry(
        system_type="surface_cluster",
        n_atoms=n_slab + 2,
        surface_config=SurfaceSystemConfig(slab=slab),
        counts=(2, 0),
    )
    mic = geo.build_comparator(n_top=2, mic=True)
    base = _placed(dimer)
    other_site = _placed(dimer + np.array([1.8, -0.6, 0.5]))
    lattice = _placed(dimer + np.asarray(slab.cell[0], dtype=float))
    theta = np.deg2rad(80.0)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    center = dimer[:, :2].mean(axis=0)
    rotated = dimer.copy()
    rotated[:, :2] = (dimer[:, :2] - center) @ rot.T + center
    turned = _placed(rotated)

    assert mic.looks_like(base, lattice)
    assert not mic.looks_like(base, other_site)
    assert not mic.looks_like(base, turned)
    cartesian = geo.build_comparator(n_top=2, mic=False)
    assert not cartesian.looks_like(base, lattice)


def test_explicit_tolerance_disables_tightening() -> None:
    cfg = SurfaceSystemConfig(slab=_layered_slab())
    geo = resolve_uniqueness_geometry(
        system_type="surface_cluster",
        n_atoms=len(cfg.slab) + 4,
        surface_config=cfg,
        settings=UniquenessSettings(comparator_tol=0.02, comparator_pair_cor_max=0.6),
    )
    assert geo.settings.comparator_tol == 0.02
    assert geo.settings.comparator_pair_cor_max == 0.6


def test_counts_override_adsorbate_definition() -> None:
    cfg = SurfaceSystemConfig(slab=_layered_slab())
    geo = resolve_uniqueness_geometry(
        system_type="surface_cluster_adsorbate",
        n_atoms=len(cfg.slab) + 3 + 2,
        surface_config=cfg,
        counts=(3, 2),
    )
    sizes = {b.role: len(b.indices) for b in geo.blocks.blocks}
    assert sizes["deposit"] == 3
    assert sizes["adsorbate"] == 2


def test_unknown_weight_role_rejected() -> None:
    with pytest.raises(SCGOValidationError, match="has no 'mobile_slab' block"):
        resolve_uniqueness_geometry(
            system_type="gas_cluster_adsorbate",
            n_atoms=5,
            adsorbate_definition=AdsorbateDefinition(
                core_symbols=["Pt"],
                adsorbate_symbols=["O", "H", "H"],
                adsorbate_fragment_lengths=[3],
            ),
            settings=UniquenessSettings(component_weights={"mobile_slab": 0.5}),
        )


def test_gas_pure_cluster_resolves_no_blocks() -> None:
    geo = resolve_uniqueness_geometry(system_type="gas_cluster", n_atoms=13)
    assert geo.blocks is None


def test_gas_core_adsorbate_splits_blocks() -> None:
    geo = resolve_uniqueness_geometry(
        system_type="gas_cluster_adsorbate",
        n_atoms=5,
        adsorbate_definition=AdsorbateDefinition(
            core_symbols=["Pt"],
            adsorbate_symbols=["O", "H", "H", "H"],
            adsorbate_fragment_lengths=[4],
        ),
    )
    assert geo.blocks is not None
    sizes = {b.role: len(b.indices) for b in geo.blocks.blocks}
    assert sizes["deposit"] == 1
    assert sizes["adsorbate"] == 4
