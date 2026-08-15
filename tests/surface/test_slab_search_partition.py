"""Tests for slab-as-target search partition helpers."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from scgo.exceptions import SCGOValidationError
from scgo.surface.config import SurfaceSystemConfig
from scgo.surface.partition import (
    prepare_slab_search_surface_config,
    resolve_slab_search_partition,
    validate_slab_search_config,
)
from scgo.surface.presets import (
    build_defected_graphite_slab,
    build_graphite_slab,
    build_n_doped_graphite_slab,
    make_defected_graphite_surface_config,
    make_n_doped_graphite_surface_config,
)
from scgo.system_types import (
    AdsorbateDefinition,
    get_system_policy,
    resolve_search_mobile_composition,
    validate_system_type_settings,
)


def _three_layer_c_slab() -> Atoms:
    pos = np.zeros((6, 3))
    pos[0:2, 2] = 0.0
    pos[2:4, 2] = 1.0
    pos[4:6, 2] = 2.0
    return Atoms("C6", positions=pos, cell=[10, 10, 15], pbc=[True, True, False])


def test_validate_slab_search_config_rejects_fully_frozen() -> None:
    cfg = SurfaceSystemConfig(slab=_three_layer_c_slab(), fix_all_slab_atoms=True)
    with pytest.raises(SCGOValidationError, match="fix_all_slab_atoms=False"):
        validate_slab_search_config(cfg)


def test_resolve_slab_search_partition_top_layer() -> None:
    cfg = SurfaceSystemConfig(
        slab=_three_layer_c_slab(),
        fix_all_slab_atoms=False,
        n_relax_top_slab_layers=1,
    )
    part = resolve_slab_search_partition(cfg)
    assert part.n_fixed == 4
    assert part.n_mobile_slab == 2
    assert set(part.mobile_slab_indices) == {4, 5}


def test_prepare_slab_search_surface_config_contiguous_prefix() -> None:
    # Scramble layer order: put top-layer atoms first in the input slab.
    pos = np.zeros((6, 3))
    pos[0:2, 2] = 2.0
    pos[2:4, 2] = 0.0
    pos[4:6, 2] = 1.0
    slab = Atoms("C6", positions=pos, cell=[10, 10, 15], pbc=[True, True, False])
    cfg = SurfaceSystemConfig(
        slab=slab, fix_all_slab_atoms=False, n_relax_top_slab_layers=1
    )
    new_cfg, part = prepare_slab_search_surface_config(cfg)
    assert list(part.fixed_indices) == list(range(part.n_fixed))
    assert list(part.mobile_slab_indices) == list(range(part.n_fixed, part.n_slab))
    z = new_cfg.slab.get_positions()[:, 2]
    assert z[: part.n_fixed].max() < z[part.n_fixed :].min()


def test_system_policies_for_new_types() -> None:
    bare = get_system_policy("surface")
    ads = get_system_policy("surface_adsorbate")
    assert bare.slab_is_search_target and not bare.has_adsorbate
    assert ads.slab_is_search_target and ads.has_adsorbate
    assert bare.uses_surface and ads.uses_surface


def test_validate_system_type_settings_requires_layer_policy() -> None:
    cfg = SurfaceSystemConfig(
        slab=_three_layer_c_slab(),
        fix_all_slab_atoms=False,
        n_relax_top_slab_layers=None,
        n_fix_bottom_slab_layers=None,
    )
    with pytest.raises(SCGOValidationError, match="n_relax_top_slab_layers"):
        validate_system_type_settings(system_type="surface", surface_config=cfg)


def test_resolve_search_mobile_composition_surface() -> None:
    cfg = SurfaceSystemConfig(
        slab=_three_layer_c_slab(),
        fix_all_slab_atoms=False,
        n_relax_top_slab_layers=1,
    )
    cfg, _ = prepare_slab_search_surface_config(cfg)
    mobile = resolve_search_mobile_composition(
        system_type="surface", composition=[], surface_config=cfg
    )
    assert mobile == ["C", "C"]


def test_resolve_search_mobile_composition_requires_surface_config() -> None:
    with pytest.raises(SCGOValidationError, match="requires surface_config"):
        resolve_search_mobile_composition(
            system_type="surface",
            composition=[],
            surface_config=None,
        )


def test_resolve_search_mobile_composition_slab_plus_adsorbate() -> None:
    cfg = SurfaceSystemConfig(
        slab=_three_layer_c_slab(),
        fix_all_slab_atoms=False,
        n_relax_top_slab_layers=1,
    )
    cfg, part = prepare_slab_search_surface_config(cfg)
    ads = AdsorbateDefinition(
        core_symbols=[],
        adsorbate_symbols=["O", "H"],
        adsorbate_fragment_lengths=[2],
    )
    mobile = resolve_search_mobile_composition(
        system_type="surface_adsorbate",
        composition=["O", "H"],
        surface_config=cfg,
        adsorbate_definition=ads,
    )
    assert mobile == list(part.mobile_slab_symbols) + ["O", "H"]


def test_defected_graphite_preset_removes_vacancy() -> None:
    pristine = build_graphite_slab(layers=3, repeat_xy=2)
    defected = build_defected_graphite_slab(
        layers=3, repeat_xy=2, n_vacancies=1, seed=1
    )
    assert len(defected) == len(pristine) - 1
    cfg = make_defected_graphite_surface_config(
        slab_layers=3, slab_repeat_xy=2, n_vacancies=1, seed=1
    )
    assert cfg.fix_all_slab_atoms is False
    assert cfg.n_relax_top_slab_layers == 1
    assert cfg.name == "defected_graphite"


def test_n_doped_graphite_preset_substitutes_n() -> None:
    doped = build_n_doped_graphite_slab(layers=3, repeat_xy=2, n_dopants=3, seed=2)
    assert doped.get_chemical_symbols().count("N") == 3
    cfg = make_n_doped_graphite_surface_config(
        slab_layers=3, slab_repeat_xy=2, n_dopants=3, seed=2
    )
    assert cfg.name == "n_doped_graphite"
    assert "N" in cfg.slab.get_chemical_symbols()
