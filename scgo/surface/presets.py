"""Reusable slab and surface presets for runners and benchmarks."""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.build import graphene

from scgo.exceptions import SCGOValidationError
from scgo.initialization.initialization_config import CONNECTIVITY_FACTOR
from scgo.surface.config import SurfaceSystemConfig
from scgo.surface.pbc import normalize_slab_pbc

DEFAULT_GRAPHITE_SLAB_LAYERS = 5
DEFAULT_GRAPHITE_SLAB_REPEAT_XY = 4
DEFAULT_GRAPHITE_SLAB_VACUUM = 12.0
# Graphite interlayer distance (AB stacking ~3.35 Å)
GRAPHITE_INTERLAYER_DISTANCE = 3.35


def build_graphite_slab(
    *,
    layers: int = DEFAULT_GRAPHITE_SLAB_LAYERS,
    vacuum: float = DEFAULT_GRAPHITE_SLAB_VACUUM,
    repeat_xy: int = DEFAULT_GRAPHITE_SLAB_REPEAT_XY,
) -> Atoms:
    """Build a graphite slab with periodic in-plane boundary conditions.

    Creates a multi-layer graphite slab with correct interlayer spacing
    (~3.35 Angstroms). Each layer is a graphene sheet, and layers are stacked
    with the graphite interlayer distance.
    """
    if layers < 1:
        raise SCGOValidationError(f"layers must be >= 1, got {layers}")

    single_layer = graphene(formula="C2", vacuum=0.0)
    single_layer = single_layer.repeat((repeat_xy, repeat_xy, 1))

    if layers == 1:
        single_layer.center(vacuum=vacuum, axis=2)
        normalize_slab_pbc(single_layer)
        return single_layer

    all_positions = single_layer.get_positions().copy()
    all_symbols = single_layer.get_chemical_symbols()

    cell = single_layer.get_cell()
    # Bernal AB: shift odd layers by (a1 + a2) / 3 in the graphene plane.
    ab_shift = (cell[0] + cell[1]) / 3.0

    for layer_idx in range(1, layers):
        layer_positions = single_layer.get_positions().copy()
        layer_positions[:, 2] += layer_idx * GRAPHITE_INTERLAYER_DISTANCE
        if layer_idx % 2 == 1:
            layer_positions[:, 0] += ab_shift[0]
            layer_positions[:, 1] += ab_shift[1]
        all_positions = np.vstack([all_positions, layer_positions])
        all_symbols.extend(single_layer.get_chemical_symbols())

    slab = Atoms(symbols=all_symbols, positions=all_positions)
    slab.set_cell(single_layer.get_cell())
    cell = slab.get_cell()
    cell[2, 2] = (layers - 1) * GRAPHITE_INTERLAYER_DISTANCE + vacuum
    slab.set_cell(cell)

    positions = slab.get_positions()
    positions[:, 2] += vacuum / 2
    slab.set_positions(positions)

    normalize_slab_pbc(slab)
    slab.wrap()
    return slab


def _top_layer_atom_indices(slab: Atoms, *, surface_normal_axis: int = 2) -> list[int]:
    """Indices of atoms in the uppermost distinct layer of ``slab``."""
    from scgo.surface.constraints import _layer_indices_by_clustering

    mobile = _layer_indices_by_clustering(
        np.asarray(slab.get_positions()),
        surface_normal_axis,
        n_layers=1,
        from_top=True,
    )
    return sorted(mobile)


def build_defected_graphite_slab(
    *,
    layers: int = DEFAULT_GRAPHITE_SLAB_LAYERS,
    vacuum: float = DEFAULT_GRAPHITE_SLAB_VACUUM,
    repeat_xy: int = DEFAULT_GRAPHITE_SLAB_REPEAT_XY,
    n_vacancies: int = 1,
    seed: int = 0,
) -> Atoms:
    """Build graphite with ``n_vacancies`` removed from the top layer."""
    if n_vacancies < 1:
        raise SCGOValidationError(f"n_vacancies must be >= 1, got {n_vacancies}")
    slab = build_graphite_slab(layers=layers, vacuum=vacuum, repeat_xy=repeat_xy)
    top_idx = _top_layer_atom_indices(slab)
    if n_vacancies > len(top_idx):
        raise SCGOValidationError(
            f"n_vacancies={n_vacancies} exceeds top-layer atom count {len(top_idx)}"
        )
    rng = np.random.default_rng(seed)
    remove = sorted(rng.choice(top_idx, size=n_vacancies, replace=False).tolist())
    keep = [i for i in range(len(slab)) if i not in set(remove)]
    out = slab[keep]
    out.set_cell(slab.get_cell())
    out.set_pbc(slab.get_pbc())
    normalize_slab_pbc(out)
    return out


def build_n_doped_graphite_slab(
    *,
    layers: int = DEFAULT_GRAPHITE_SLAB_LAYERS,
    vacuum: float = DEFAULT_GRAPHITE_SLAB_VACUUM,
    repeat_xy: int = DEFAULT_GRAPHITE_SLAB_REPEAT_XY,
    n_dopants: int = 1,
    seed: int = 0,
) -> Atoms:
    """Build graphite with ``n_dopants`` top-layer C atoms substituted by N."""
    if n_dopants < 1:
        raise SCGOValidationError(f"n_dopants must be >= 1, got {n_dopants}")
    slab = build_graphite_slab(layers=layers, vacuum=vacuum, repeat_xy=repeat_xy)
    top_idx = _top_layer_atom_indices(slab)
    if n_dopants > len(top_idx):
        raise SCGOValidationError(
            f"n_dopants={n_dopants} exceeds top-layer atom count {len(top_idx)}"
        )
    rng = np.random.default_rng(seed)
    doped = rng.choice(top_idx, size=n_dopants, replace=False)
    symbols = slab.get_chemical_symbols()
    for i in doped:
        symbols[int(i)] = "N"
    out = slab.copy()
    out.set_chemical_symbols(symbols)
    return out


def make_graphite_surface_config(
    *,
    slab_layers: int = DEFAULT_GRAPHITE_SLAB_LAYERS,
    slab_repeat_xy: int = DEFAULT_GRAPHITE_SLAB_REPEAT_XY,
    vacuum: float = DEFAULT_GRAPHITE_SLAB_VACUUM,
    structure_connectivity_factor: float = CONNECTIVITY_FACTOR,
) -> SurfaceSystemConfig:
    """Graphite slab preset (top layer relaxes with adsorbate during GO/NEB)."""
    slab = build_graphite_slab(
        layers=slab_layers, vacuum=vacuum, repeat_xy=slab_repeat_xy
    )
    return SurfaceSystemConfig(
        slab=slab,
        name="graphite",
        adsorption_height_min=0.5,
        adsorption_height_max=1.0,
        fix_all_slab_atoms=False,
        n_relax_top_slab_layers=1,
        comparator_use_mic=True,
        max_placement_attempts=1000,
        structure_connectivity_factor=structure_connectivity_factor,
    )


def make_defected_graphite_surface_config(
    *,
    slab_layers: int = DEFAULT_GRAPHITE_SLAB_LAYERS,
    slab_repeat_xy: int = DEFAULT_GRAPHITE_SLAB_REPEAT_XY,
    vacuum: float = DEFAULT_GRAPHITE_SLAB_VACUUM,
    n_vacancies: int = 1,
    seed: int = 0,
    structure_connectivity_factor: float = CONNECTIVITY_FACTOR,
) -> SurfaceSystemConfig:
    """Defected graphite preset for ``system_type='surface'`` slab search."""
    slab = build_defected_graphite_slab(
        layers=slab_layers,
        vacuum=vacuum,
        repeat_xy=slab_repeat_xy,
        n_vacancies=n_vacancies,
        seed=seed,
    )
    return SurfaceSystemConfig(
        slab=slab,
        name="defected_graphite",
        adsorption_height_min=0.5,
        adsorption_height_max=1.0,
        fix_all_slab_atoms=False,
        n_relax_top_slab_layers=1,
        comparator_use_mic=True,
        max_placement_attempts=1000,
        structure_connectivity_factor=structure_connectivity_factor,
    )


def make_n_doped_graphite_surface_config(
    *,
    slab_layers: int = DEFAULT_GRAPHITE_SLAB_LAYERS,
    slab_repeat_xy: int = DEFAULT_GRAPHITE_SLAB_REPEAT_XY,
    vacuum: float = DEFAULT_GRAPHITE_SLAB_VACUUM,
    n_dopants: int = 1,
    seed: int = 0,
    structure_connectivity_factor: float = CONNECTIVITY_FACTOR,
) -> SurfaceSystemConfig:
    """N-doped graphite preset for ``system_type='surface_adsorbate'`` search."""
    slab = build_n_doped_graphite_slab(
        layers=slab_layers,
        vacuum=vacuum,
        repeat_xy=slab_repeat_xy,
        n_dopants=n_dopants,
        seed=seed,
    )
    return SurfaceSystemConfig(
        slab=slab,
        name="n_doped_graphite",
        adsorption_height_min=0.5,
        adsorption_height_max=1.5,
        fix_all_slab_atoms=False,
        n_relax_top_slab_layers=1,
        comparator_use_mic=True,
        max_placement_attempts=1000,
        structure_connectivity_factor=structure_connectivity_factor,
    )


__all__ = [
    "DEFAULT_GRAPHITE_SLAB_LAYERS",
    "DEFAULT_GRAPHITE_SLAB_REPEAT_XY",
    "DEFAULT_GRAPHITE_SLAB_VACUUM",
    "build_graphite_slab",
    "build_defected_graphite_slab",
    "build_n_doped_graphite_slab",
    "make_graphite_surface_config",
    "make_defected_graphite_surface_config",
    "make_n_doped_graphite_surface_config",
]
