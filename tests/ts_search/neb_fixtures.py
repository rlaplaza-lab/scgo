"""Shared NEB test helpers that must stay importable from the CPU suite.

``test_parallel_neb.py`` is module-marked ``requires_cuda`` / ``requires_mace``.
CPU tests import these factories instead of that module.
"""

from __future__ import annotations

from typing import Any

import pytest
from ase import Atoms
from ase.build import fcc111

from scgo.initialization.initialization_config import CLASH_HARD_CAP_FACTOR
from scgo.surface.config import SurfaceSystemConfig
from scgo.surface.partition import prepare_slab_search_surface_config
from scgo.system_types import AdsorbateDefinition
from scgo.ts_search import transition_state_run as ts_run_mod
from scgo.utils.ts_runner_kwargs import NebRunConfig


def _gas_neb_cfg(**overrides) -> NebRunConfig:
    """Minimal gas-cluster NebRunConfig for parallel NEB unit tests."""
    kwargs: dict = {
        "neb_n_images": 3,
        "neb_spring_constant": 0.1,
        "neb_fmax": 0.05,
        "neb_steps": 2,
        "neb_climb": False,
        "neb_interpolation_method": "linear",
        "neb_align_endpoints": False,
        "neb_perturb_sigma": 0.0,
        "neb_interpolation_mic": False,
        "neb_tangent_method": "aseneb",
        "neb_surface_cell_remap": True,
        "neb_surface_lattice_rotation": True,
        "neb_surface_max_lattice_shift": 1,
        "n_slab": 0,
        "n_core_mobile": None,
        "n_adsorbate_mobile": None,
        "adsorbate_fragment_lengths": None,
        "max_endpoint_mismatch": None,
        "adsorbate_definition": None,
        "connectivity_factor": None,
        "allow_cluster_fragmentation": False,
        "allow_adsorbate_surface_detachment": False,
        "enforce_adsorbate_subgraph_integrity": True,
        "system_type": "gas_cluster",
        "surface_config": None,
        "torchsim_params": {},
    }
    # Per-system-type required fields (mirrors TS_DEFAULTS_BY_SYSTEM_TYPE). Set
    # here, before kwargs.update(overrides), so individual tests can still
    # override them. Must stay in sync with NebRunConfig's required fields.
    _surface_defaults = ("surface_cluster", "surface", "surface_adsorbate")
    _is_surface = overrides.get("system_type", "gas_cluster") in _surface_defaults
    if _is_surface:
        _clash, _prom, _spurious = CLASH_HARD_CAP_FACTOR, 0.40, 8.0
    else:
        _clash, _prom, _spurious = CLASH_HARD_CAP_FACTOR, 0.10, 8.0
    kwargs.update(
        neb_prescreen_clash_distance=_clash,
        min_saddle_prominence=_prom,
        neb_max_spurious_barrier=_spurious,
        layer_cluster_threshold_ang=0.4,
        neb_interpolation_bond_tolerance_a=0.5,
    )
    kwargs.update(overrides)
    return NebRunConfig(**kwargs)


def _prepared_search_config() -> tuple[SurfaceSystemConfig, int, int]:
    """Return a slab-search config ordered [fixed...][mobile...] plus sizes."""
    slab = fcc111("Pt", size=(2, 2, 2), vacuum=6.0, orthogonal=True)
    slab.pbc = [True, True, False]
    cfg = SurfaceSystemConfig(
        slab=slab,
        fix_all_slab_atoms=False,
        n_relax_top_slab_layers=1,
    )
    cfg, part = prepare_slab_search_surface_config(cfg)
    return cfg, int(part.n_fixed), len(cfg.slab)


def _capture_pairing_minima(
    monkeypatch: pytest.MonkeyPatch,
    minima_by_formula: dict[str, list[tuple[float, Atoms]]],
) -> dict[str, Any]:
    """Stub minima loading; capture the deduped minima reaching pair selection."""
    captured: dict[str, Any] = {}

    def _fake_load(_minima_dir, composition, **_kwargs):
        captured["loaded_composition"] = list(composition)
        return dict(minima_by_formula)

    def _fake_select_pairs(minima, **_kwargs):
        captured["pairing_minima"] = list(minima)
        return []

    monkeypatch.setattr(ts_run_mod, "load_minima_by_composition", _fake_load)
    monkeypatch.setattr(ts_run_mod, "select_structure_pairs", _fake_select_pairs)
    return captured


def _run_ts_until_pairing(
    *,
    composition: list[str],
    system_type: str,
    surface_config: SurfaceSystemConfig | None,
    adsorbate_definition: AdsorbateDefinition | None,
    tmp_path,
) -> None:
    ts_run_mod.run_transition_state_search(
        composition=composition,
        system_type=system_type,  # type: ignore[arg-type]
        output_dir=str(tmp_path),
        params={"calculator": "EMT", "calculator_kwargs": {}},
        surface_config=surface_config,
        adsorbate_definition=adsorbate_definition,
        verbosity=0,
        use_torchsim=False,
        use_parallel_neb=False,
    )
