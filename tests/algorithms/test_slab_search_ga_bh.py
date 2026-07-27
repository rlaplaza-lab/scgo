"""GA/BH targeting for slab-as-search-target system types."""

from __future__ import annotations

import numpy as np
from ase import Atoms

from scgo.algorithms.ga_common import (
    SurfaceSlabStartGenerator,
    create_ga_pairing,
    create_mutation_operators,
)
from scgo.surface.config import SurfaceSystemConfig
from scgo.surface.partition import prepare_slab_search_surface_config
from scgo.system_types import get_system_policy
from scgo.surface.partition import resolve_slab_search_partition


def _layered_slab(n_per_layer: int = 2, n_layers: int = 3) -> Atoms:
    pos = np.zeros((n_per_layer * n_layers, 3))
    symbols: list[str] = []
    for layer in range(n_layers):
        for j in range(n_per_layer):
            idx = layer * n_per_layer + j
            pos[idx, 0] = j * 1.5
            pos[idx, 2] = float(layer)
            symbols.append("C" if layer < n_layers - 1 or j == 0 else "N")
    return Atoms(
        symbols=symbols,
        positions=pos,
        cell=[8, 8, 12],
        pbc=[True, True, False],
    )


def test_slab_search_operators_exclude_cluster_shape_ops() -> None:
    slab = _layered_slab()
    cfg = SurfaceSystemConfig(
        slab=slab, fix_all_slab_atoms=False, n_relax_top_slab_layers=1
    )
    cfg, part = prepare_slab_search_surface_config(cfg)
    ops, names = create_mutation_operators(
        composition=list(part.mobile_slab_symbols),
        n_to_optimize=part.n_mobile_slab,
        blmin={(6, 6): 1.2, (6, 7): 1.1, (7, 7): 1.0},
        system_type="surface",
        n_slab=part.n_fixed,
        use_adaptive=True,
    )
    assert "rattle" in names
    assert "in_plane_slide" in names
    assert "flattening" not in names
    assert "rotational" not in names
    assert "breathing" not in names
    assert "fragment_reposition" not in names
    assert len(ops) == len(names)


def test_create_ga_pairing_uses_fixed_prefix_only() -> None:
    slab = _layered_slab()
    cfg = SurfaceSystemConfig(
        slab=slab, fix_all_slab_atoms=False, n_relax_top_slab_layers=1
    )
    cfg, part = prepare_slab_search_surface_config(cfg)
    template = cfg.slab.copy()
    pairing = create_ga_pairing(
        template,
        part.n_mobile_slab,
        rng=np.random.default_rng(0),
        slab_atoms=cfg.slab[: part.n_fixed],
        system_type="surface",
        composition=list(part.mobile_slab_symbols),
        exploratory_crossover_probability=0.0,
    )
    assert len(pairing.slab) == part.n_fixed
    assert pairing.n_top == part.n_mobile_slab


def test_surface_slab_start_generator_rattles_only_mobile() -> None:
    slab = _layered_slab()
    cfg = SurfaceSystemConfig(
        slab=slab, fix_all_slab_atoms=False, n_relax_top_slab_layers=1
    )
    cfg, part = prepare_slab_search_surface_config(cfg)
    gen = SurfaceSlabStartGenerator(
        cfg.slab, n_fixed=part.n_fixed, population_size=2, rng=np.random.default_rng(0)
    )
    cand = gen.get_new_candidate()
    assert np.allclose(
        cand.get_positions()[: part.n_fixed], cfg.slab.get_positions()[: part.n_fixed]
    )
    assert not np.allclose(
        cand.get_positions()[part.n_fixed :], cfg.slab.get_positions()[part.n_fixed :]
    )


def test_bh_movable_indices_for_slab_search() -> None:
    """Resolve movable indices the same way bh_go does for surface search."""
    slab = _layered_slab(n_per_layer=1, n_layers=3)
    slab.set_chemical_symbols(["C", "C", "C"])
    cfg = SurfaceSystemConfig(
        slab=slab,
        fix_all_slab_atoms=False,
        n_relax_top_slab_layers=1,
        name="toy",
    )
    cfg, part = prepare_slab_search_surface_config(cfg)
    atoms = cfg.slab.copy()
    policy = get_system_policy("surface")
    assert policy.slab_is_search_target
    resolved = resolve_slab_search_partition(cfg)
    movable = list(range(resolved.n_fixed, len(atoms)))
    assert movable == list(range(part.n_fixed, len(atoms)))
    assert part.n_fixed == resolved.n_fixed
    assert 0 not in movable
