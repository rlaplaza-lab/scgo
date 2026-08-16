"""Tests for graphene/graphite vacancy builders and defect-biased deposition."""

from __future__ import annotations

import numpy as np
import pytest
from ase.data import atomic_numbers
from numpy.random import default_rng

from scgo.exceptions import SCGOValidationError
from scgo.initialization.atomic_radii import build_blmin_from_zs
from scgo.surface import (
    build_defected_graphite_slab,
    build_graphene_slab,
    build_graphite_slab,
    build_monovacancy_graphene_slab,
    create_deposited_cluster,
    make_defected_graphite_surface_config,
    make_graphene_surface_config,
)
from scgo.surface.config import SurfaceSystemConfig
from scgo.surface.presets import GRAPHITE_INTERLAYER_DISTANCE

_NO_DB_GLOB = "tests/__no_such_db__.db"


def _pt_blmin(slab, composition) -> dict:
    zs = list(slab.get_atomic_numbers()) + [atomic_numbers[s] for s in composition]
    return build_blmin_from_zs(zs, ratio=0.7)


def _deposit(composition, slab, blmin, rng, cfg):
    return create_deposited_cluster(
        composition,
        slab,
        blmin,
        rng,
        cfg,
        previous_search_glob=_NO_DB_GLOB,
        emit_diagnostics=False,
        verbosity=0,
    )


def test_build_graphene_slab_is_monolayer() -> None:
    slab = build_graphene_slab(nx=4, ny=4)
    assert len(slab) == 2 * 4 * 4
    assert set(slab.get_chemical_symbols()) == {"C"}
    assert list(slab.pbc) == [True, True, False]
    assert np.allclose(slab.get_positions()[:, 2], slab.cell[2, 2] / 2.0)


def test_monovacancy_removes_one_and_records_info() -> None:
    pristine = build_graphene_slab(nx=4, ny=4)
    defective = build_monovacancy_graphene_slab(nx=4, ny=4)
    assert len(defective) == len(pristine) - 1
    removed = defective.info["vacancy_removed_original_index_zero_based"]
    assert isinstance(removed, int)
    assert 0 <= removed < len(pristine)
    vac = defective.info["vacancy_cartesian_angstrom"]
    assert isinstance(vac, list)
    assert len(vac) == 3
    assert np.allclose(vac, pristine.get_positions()[removed])


def test_reconstruct_does_not_change_count() -> None:
    pristine = build_graphene_slab(nx=4, ny=4)
    defective = build_monovacancy_graphene_slab(
        nx=4, ny=4, reconstruct=True, reconstruction_shift=0.10
    )
    assert len(defective) == len(pristine) - 1
    assert "vacancy_cartesian_angstrom" in defective.info


def test_make_graphene_surface_config_monovacancy() -> None:
    cfg = make_graphene_surface_config(nx=4, ny=4, monovacancy=True)
    assert isinstance(cfg, SurfaceSystemConfig)
    assert len(cfg.slab) == 31
    assert cfg.defect_bias_probability == 0.5
    assert cfg.slab.info["vacancy_cartesian_angstrom"] is not None
    assert cfg.name == "graphene_monovacancy"


def test_defected_graphite_records_vacancy() -> None:
    slab = build_defected_graphite_slab(n_vacancies=1, seed=0)
    assert "vacancy_cartesian_angstrom" in slab.info
    assert "vacancy_removed_original_index_zero_based" in slab.info

    cfg = make_defected_graphite_surface_config(
        slab_layers=3, slab_repeat_xy=2, n_vacancies=1, seed=0
    )
    assert cfg.defect_bias_probability == 0.5
    assert "vacancy_cartesian_angstrom" in cfg.slab.info
    assert (
        make_defected_graphite_surface_config(
            slab_layers=3,
            slab_repeat_xy=2,
            n_vacancies=1,
            seed=0,
            defect_bias_probability=0.0,
        ).defect_bias_probability
        == 0.0
    )


def test_graphite_vacuum_is_total_padding() -> None:
    mono = build_graphite_slab(layers=1, vacuum=12.0, repeat_xy=2)
    assert mono.cell[2, 2] == pytest.approx(12.0)
    assert abs(float(mono.get_positions()[0, 2]) - 6.0) < 1.0
    bi = build_graphite_slab(layers=2, vacuum=12.0, repeat_xy=2)
    assert bi.cell[2, 2] == pytest.approx(GRAPHITE_INTERLAYER_DISTANCE + 12.0)


def test_placement_bias_lands_on_defect() -> None:
    cfg = make_graphene_surface_config(
        nx=4, ny=4, monovacancy=True, defect_bias_probability=1.0
    )
    slab = cfg.slab
    blmin = _pt_blmin(slab, ["Pt", "Pt", "Pt", "Pt", "Pt"])
    vac = np.array(slab.info["vacancy_cartesian_angstrom"])
    in_plane = [0, 1]
    tol = 1.5

    rng = default_rng(0)
    n_total = 8
    landed = 0
    for _ in range(n_total):
        struct = _deposit(["Pt", "Pt", "Pt", "Pt", "Pt"], slab, blmin, rng, cfg)
        assert struct is not None
        mobile = struct[len(slab) :].get_positions()
        centroid = mobile.mean(axis=0)
        if np.linalg.norm(centroid[in_plane] - vac[in_plane]) < tol:
            landed += 1
    assert landed == n_total

    # Default 0.5 probability gives a strictly positive defect fraction.
    cfg_half = make_graphene_surface_config(
        nx=4, ny=4, monovacancy=True, defect_bias_probability=0.5
    )
    rng2 = default_rng(1)
    n_half = 24
    landed_half = 0
    for _ in range(n_half):
        struct = _deposit(["Pt", "Pt", "Pt", "Pt", "Pt"], slab, blmin, rng2, cfg_half)
        assert struct is not None
        mobile = struct[len(slab) :].get_positions()
        centroid = mobile.mean(axis=0)
        if np.linalg.norm(centroid[in_plane] - vac[in_plane]) < tol:
            landed_half += 1
    assert landed_half / n_half > 0.2


def test_placement_backward_compat() -> None:
    cfg = make_graphene_surface_config(nx=4, ny=4, monovacancy=False)
    slab = cfg.slab
    assert "vacancy_cartesian_angstrom" not in slab.info
    blmin = _pt_blmin(slab, ["Pt", "Pt", "Pt", "Pt", "Pt"])

    rng = default_rng(7)
    x_centers: list[float] = []
    n_total = 8
    for _ in range(n_total):
        struct = _deposit(["Pt", "Pt", "Pt", "Pt", "Pt"], slab, blmin, rng, cfg)
        assert struct is not None
        mobile = struct[len(slab) :].get_positions()
        x_centers.append(float(mobile.mean(axis=0)[0]))
    # Placements are spread across the slab, not pinned to one in-plane point.
    assert max(x_centers) - min(x_centers) > 4.0


def test_defect_bias_probability_out_of_range() -> None:
    slab = build_graphene_slab(nx=3, ny=3)
    with pytest.raises(SCGOValidationError, match="defect_bias_probability must be in"):
        SurfaceSystemConfig(slab=slab, name="graphene", defect_bias_probability=1.5)
    with pytest.raises(SCGOValidationError, match="defect_bias_probability must be in"):
        SurfaceSystemConfig(slab=slab, name="graphene", defect_bias_probability=-0.1)
