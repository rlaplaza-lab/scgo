"""Reusable slab and surface presets for runners and benchmarks.

Note on the adsorption-height window: every preset fixes
``adsorption_height_min/max`` to a narrow 0.5-1.0 or 0.5-1.5 Å band above the
slab top. This is a *deliberate* chemisorption-contact bias — it seeds the mobile
cluster at typical chemical-bonding distances rather than far out in the
physisorption tail. It is NOT a free standoff: every accepted placement is still
gated by ``atoms_too_close_two_sets(adsorbate, slab, blmin)`` (blmin_ratio 0.7,
so e.g. Pt-C >= 1.48 Å), and the window can be widened or overridden per-run via
``adsorption_height_min/max``. The tradeoff of the tight window is a hollow /
vacancy-site bias and no physisorption sampling — intended for bonding searches.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.build import graphene

from scgo.exceptions import SCGOValidationError
from scgo.initialization.initialization_config import CONNECTIVITY_FACTOR
from scgo.surface.config import SurfaceSystemConfig
from scgo.surface.layers import _layer_indices_by_clustering
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


def _central_atom_index(atoms: Atoms, *, surface_normal_axis: int = 2) -> int:
    """Index of the atom whose in-plane (scaled) position is closest to the cell center."""
    scaled = atoms.get_scaled_positions(wrap=True)
    in_plane = [i for i in range(3) if i != surface_normal_axis]
    center = np.array([0.5, 0.5], dtype=float)
    pts = scaled[:, in_plane]
    dist2 = np.sum((pts - center) ** 2, axis=1)
    return int(np.argmin(dist2))


def build_graphene_slab(
    *,
    nx: int = 6,
    ny: int = 6,
    a: float = 2.46,
    cell_height: float = 18.0,
) -> Atoms:
    """Build a single-layer (monolayer) graphene slab.

    The graphene sheet lies in the ``x-y`` plane and is centered along ``z``.
    The cell length along the surface normal (``z``) is set explicitly to
    ``cell_height`` (matching the Chepkasov "cell height normal to graphene"
    semantics). The slab uses slab-style PBC — periodic in-plane and open along
    the vacuum axis — consistent with :func:`~scgo.surface.presets.build_graphite_slab` and the rest
    of the surface pipeline (deposition, constraints, comparators).
    """
    if nx < 1:
        raise SCGOValidationError(f"nx must be >= 1, got {nx}")
    if ny < 1:
        raise SCGOValidationError(f"ny must be >= 1, got {ny}")
    if cell_height <= 0:
        raise SCGOValidationError(f"cell_height must be > 0, got {cell_height}")

    slab = graphene(a=a, size=(nx, ny, 1), vacuum=0.0)
    slab.set_pbc((True, True, True))

    cell = slab.get_cell()
    cell[2, 2] = cell_height
    slab.set_cell(cell, scale_atoms=False)

    positions = slab.get_positions()
    positions[:, 2] = cell_height / 2.0
    slab.set_positions(positions)
    slab.wrap()
    normalize_slab_pbc(slab)
    return slab


def build_monovacancy_graphene_slab(
    *,
    nx: int = 6,
    ny: int = 6,
    a: float = 2.46,
    cell_height: float = 18.0,
    reconstruct: bool = False,
    reconstruction_shift: float = 0.10,
) -> Atoms:
    """Build a monolayer graphene slab with a single carbon monovacancy.

    The central atom (closest to the in-plane cell center) is removed. The
    removed atom's Cartesian position and original index are recorded on the
    returned ``Atoms.info`` so the deposition pipeline can bias cluster placement
    onto the vacancy. When ``reconstruct`` is True, the two nearest vacancy
    neighbors are displaced along their bond axes (Jahn-Teller seed).
    """
    pristine = build_graphene_slab(nx=nx, ny=ny, a=a, cell_height=cell_height)
    removed = _central_atom_index(pristine)
    vacancy_position = pristine.get_positions()[removed].copy()

    defective = pristine.copy()
    del defective[removed]

    if reconstruct:
        pos = defective.get_positions()
        dists = np.linalg.norm(pos - vacancy_position, axis=1)
        neighbors = np.argsort(dists)[:3]
        for k, idx in enumerate(neighbors[:2]):
            direction = pos[idx] - vacancy_position
            norm = np.linalg.norm(direction)
            if norm > 1e-12:
                direction = direction / norm
                sign = 1.0 if k % 2 == 0 else -1.0
                pos[idx] += sign * reconstruction_shift * direction
        defective.set_positions(pos)
        defective.wrap()

    defective.info["vacancy_removed_original_index_zero_based"] = int(removed)
    defective.info["vacancy_cartesian_angstrom"] = vacancy_position.tolist()
    return defective


def make_graphene_surface_config(
    *,
    nx: int = 6,
    ny: int = 6,
    a: float = 2.46,
    cell_height: float = 18.0,
    monovacancy: bool = False,
    reconstruct: bool = False,
    reconstruction_shift: float = 0.10,
    name: str = "graphene",
    structure_connectivity_factor: float = CONNECTIVITY_FACTOR,
    defect_bias_probability: float | None = None,
) -> SurfaceSystemConfig:
    """Single-layer graphene (or monovacancy) preset for ``surface_cluster`` search."""
    if defect_bias_probability is None:
        defect_bias_probability = 0.5 if monovacancy else 0.0

    if monovacancy:
        slab = build_monovacancy_graphene_slab(
            nx=nx,
            ny=ny,
            a=a,
            cell_height=cell_height,
            reconstruct=reconstruct,
            reconstruction_shift=reconstruction_shift,
        )
        if name == "graphene":
            name = "graphene_monovacancy"
    else:
        slab = build_graphene_slab(nx=nx, ny=ny, a=a, cell_height=cell_height)

    return SurfaceSystemConfig(
        slab=slab,
        name=name,
        adsorption_height_min=0.5,
        adsorption_height_max=1.5,
        fix_all_slab_atoms=False,
        n_relax_top_slab_layers=1,
        comparator_use_mic=True,
        max_placement_attempts=1000,
        structure_connectivity_factor=structure_connectivity_factor,
        defect_bias_probability=defect_bias_probability,
    )


def _top_layer_atom_indices(slab: Atoms, *, surface_normal_axis: int = 2) -> list[int]:
    """Indices of atoms in the uppermost distinct layer of ``slab``."""
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
    vacancy_position = slab.get_positions()[remove[0]].copy()
    out = slab[keep]
    out.set_cell(slab.get_cell())
    out.set_pbc(slab.get_pbc())
    out.info["vacancy_removed_original_index_zero_based"] = int(remove[0])
    out.info["vacancy_cartesian_angstrom"] = vacancy_position.tolist()
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
    "build_graphene_slab",
    "build_monovacancy_graphene_slab",
    "make_graphene_surface_config",
    "make_graphite_surface_config",
    "make_defected_graphite_surface_config",
    "make_n_doped_graphite_surface_config",
]
