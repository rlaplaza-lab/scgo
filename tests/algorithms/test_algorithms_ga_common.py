import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111
from ase.ga.utilities import get_all_atom_types

from scgo.algorithms.ga_common import (
    setup_diversity_scorer,
    validate_structure_for_ga_storage,
)
from scgo.database import close_data_connection, setup_database
from scgo.exceptions import SCGOValidationError
from scgo.initialization.atomic_radii import build_blmin_from_zs
from scgo.surface.config import SurfaceSystemConfig
from scgo.surface.deposition import create_deposited_cluster_batch
from scgo.surface.partition import prepare_slab_search_surface_config
from scgo.surface.presets import make_n_doped_graphite_surface_config
from scgo.surface.validation import (
    _STACKING_GAP_RUMPLE_FACTOR,
    layer_stacking_cutoff_from_template,
)
from scgo.system_types import AdsorbateDefinition, validate_structure_for_system_type
from scgo.utils.logging import get_logger


def test_setup_diversity_scorer_uses_base_dir(tmp_path, rng, monkeypatch):
    """Ensure setup_diversity_scorer resolves reference DBs from base_dir."""
    searches = tmp_path / "Pt3_searches"
    run_dir = searches / "run_001"
    run_dir.mkdir(parents=True)

    atoms = Atoms("Pt3", positions=[[0, 0, 0], [2.5, 0, 0], [1.25, 2.0, 0]])
    da = setup_database(run_dir, "ref_1.db", atoms, initial_candidate=atoms)

    for i in range(3):
        a = atoms.copy()
        a.positions += rng.random((3, 3)) * 0.1
        a.info["key_value_pairs"] = {"raw_score": -30.0 - i}
        a.info["data"] = {"tag": f"test_{i}"}
        if i == 0:
            a.info.setdefault("key_value_pairs", {})["final_unique_minimum"] = True
            a.info.setdefault("key_value_pairs", {})["final_unique_minimum"] = True
        da.add_relaxed_step(a)

    close_data_connection(da)
    del da

    monkeypatch.chdir(tmp_path)
    scorer = setup_diversity_scorer(
        fitness_strategy="diversity",
        diversity_reference_db="run_*/ref_*.db",
        composition=["Pt", "Pt", "Pt"],
        n_to_optimize=3,
        diversity_max_references=10,
        logger=get_logger(__name__),
        base_dir=str(searches),
    )
    assert scorer is not None


def test_validate_structure_for_ga_storage_uses_canonical_frame() -> None:
    """Storage validation must canonicalize before checking eligibility."""
    slab = fcc111("Pt", size=(2, 2, 2), vacuum=6.0, orthogonal=True)
    slab.pbc = True
    surface_config = SurfaceSystemConfig(
        slab=slab,
        adsorption_height_min=1.0,
        adsorption_height_max=2.8,
        fix_all_slab_atoms=True,
        comparator_use_mic=False,
        max_placement_attempts=400,
    )
    n_slab = len(slab)
    blmin = build_blmin_from_zs(get_all_atom_types(slab, [78]), ratio=0.7)
    batch = create_deposited_cluster_batch(
        ["Pt", "Pt"],
        slab,
        blmin,
        12,
        np.random.default_rng(42),
        surface_config,
        n_jobs=1,
    )

    raw_pass_storage_fail = 0
    for atoms in batch:
        raw_ok = True
        try:
            validate_structure_for_system_type(
                atoms,
                system_type="surface_cluster",
                surface_config=surface_config,
                n_slab=n_slab,
            )
        except (ValueError, SCGOValidationError):
            raw_ok = False
        storage_err = validate_structure_for_ga_storage(
            atoms.copy(),
            surface_mode=True,
            n_slab=n_slab,
            system_type="surface_cluster",
            surface_config=surface_config,
        )
        if raw_ok and storage_err is not None:
            raw_pass_storage_fail += 1

    assert raw_pass_storage_fail >= 1


def test_core_adsorbate_partition_allow_empty_core() -> None:
    from scgo.algorithms.ga_common import (
        core_adsorbate_partition_counts,
        core_adsorbate_partition_details,
    )

    ads_def = AdsorbateDefinition(
        core_symbols=[],
        adsorbate_symbols=["O", "H"],
        adsorbate_fragment_lengths=[2],
    )
    assert (
        core_adsorbate_partition_counts("gas_cluster_adsorbate", ["O", "H"], ads_def)
        is None
    )
    assert core_adsorbate_partition_counts(
        "gas_cluster_adsorbate", ["O", "H"], ads_def, allow_empty_core=True
    ) == (0, 2)
    assert core_adsorbate_partition_details(
        "gas_cluster_adsorbate", ["O", "H"], ads_def, allow_empty_core=True
    ) == (0, [2])


def test_ga_storage_slab_search_empty_core_uses_deposit_prefix() -> None:
    """Empty-core surface_adsorbate needs n_slab=full and n_slab_deposit=n_fixed."""

    slab = fcc111("Pt", size=(2, 2, 3), vacuum=8.0, orthogonal=True)
    slab.pbc = [True, True, False]
    cfg = SurfaceSystemConfig(
        slab=slab,
        fix_all_slab_atoms=False,
        n_relax_top_slab_layers=1,
        adsorption_height_min=1.0,
        adsorption_height_max=3.0,
        comparator_use_mic=True,
    )
    cfg, part = prepare_slab_search_surface_config(cfg)
    n_fixed = int(part.n_fixed)
    n_full = len(cfg.slab)
    assert n_fixed < n_full

    z_top = float(np.max(cfg.slab.positions[:, 2]))
    xy = np.mean(cfg.slab.positions[n_fixed:, :2], axis=0)
    oh = Atoms(
        "OH",
        positions=[
            [xy[0], xy[1], z_top + 1.8],
            [xy[0], xy[1], z_top + 2.76],
        ],
        cell=cfg.slab.cell,
        pbc=cfg.slab.pbc,
    )
    combined = cfg.slab.copy() + oh
    tags = np.zeros(len(combined), dtype=int)
    tags[n_full:] = 1
    combined.set_tags(tags)
    ads = AdsorbateDefinition(
        core_symbols=[],
        adsorbate_symbols=["O", "H"],
        adsorbate_fragment_lengths=[2],
    )

    wrong = validate_structure_for_ga_storage(
        combined.copy(),
        surface_mode=True,
        n_slab=n_fixed,
        system_type="surface_adsorbate",
        surface_config=cfg,
        adsorbate_definition=ads,
    )
    assert wrong is not None

    right = validate_structure_for_ga_storage(
        combined.copy(),
        surface_mode=True,
        n_slab=n_full,
        n_slab_deposit=n_fixed,
        system_type="surface_adsorbate",
        surface_config=cfg,
        adsorbate_definition=ads,
    )
    assert right is None, right


def test_ga_storage_slab_search_accepts_graphite_vdw_stacking() -> None:
    """Mobile graphene on a frozen graphite prefix is stacked, not covalently bound."""
    cfg, part = prepare_slab_search_surface_config(
        make_n_doped_graphite_surface_config(
            slab_layers=3, slab_repeat_xy=2, n_dopants=1, seed=0
        )
    )
    n_fixed = int(part.n_fixed)
    n_full = len(cfg.slab)
    z_top = float(np.max(cfg.slab.positions[:, 2]))
    xy = np.mean(cfg.slab.positions[n_fixed:, :2], axis=0)
    oh = Atoms(
        "OH",
        positions=[
            [xy[0], xy[1], z_top + 1.5],
            [xy[0], xy[1], z_top + 2.46],
        ],
        cell=cfg.slab.cell,
        pbc=cfg.slab.pbc,
    )
    combined = cfg.slab.copy() + oh
    tags = np.zeros(len(combined), dtype=int)
    tags[n_full:] = 1
    combined.set_tags(tags)
    ads = AdsorbateDefinition(
        core_symbols=[],
        adsorbate_symbols=["O", "H"],
        adsorbate_fragment_lengths=[2],
    )
    err = validate_structure_for_ga_storage(
        combined.copy(),
        surface_mode=True,
        n_slab=n_full,
        n_slab_deposit=n_fixed,
        system_type="surface_adsorbate",
        surface_config=cfg,
        adsorbate_definition=ads,
    )
    assert err is None, err

    desorbed = combined.copy()
    desorbed.positions[n_fixed:] += np.array([0.0, 0.0, 10.0])
    desorbed_err = validate_structure_for_ga_storage(
        desorbed,
        surface_mode=True,
        n_slab=n_full,
        n_slab_deposit=n_fixed,
        system_type="surface_adsorbate",
        surface_config=cfg,
        adsorbate_definition=ads,
    )
    assert desorbed_err is not None


def test_layer_stacking_cutoff_follows_template_physics() -> None:
    """Metals stay covalent; graphite cutoff is the template gap, not a magic length."""
    graphite, gpart = prepare_slab_search_surface_config(
        make_n_doped_graphite_surface_config(
            slab_layers=3, slab_repeat_xy=2, n_dopants=1, seed=0
        )
    )
    g_cut = layer_stacking_cutoff_from_template(
        graphite.slab,
        int(gpart.n_fixed),
        surface_normal_axis=graphite.surface_normal_axis,
        connectivity_factor=graphite.structure_connectivity_factor,
        use_mic=bool(graphite.comparator_use_mic),
    )
    assert g_cut is not None
    axis = graphite.surface_normal_axis
    n_fixed = int(gpart.n_fixed)
    gap = float(
        np.min(graphite.slab.positions[n_fixed:, axis])
        - np.max(graphite.slab.positions[:n_fixed, axis])
    )
    assert g_cut == pytest.approx(gap * _STACKING_GAP_RUMPLE_FACTOR, rel=0.05)

    pt = fcc111("Pt", size=(2, 2, 3), vacuum=8.0, orthogonal=True)
    pt.pbc = [True, True, False]
    pt_cfg = SurfaceSystemConfig(
        slab=pt,
        fix_all_slab_atoms=False,
        n_relax_top_slab_layers=1,
        adsorption_height_min=1.0,
        adsorption_height_max=3.0,
        comparator_use_mic=True,
    )
    pt_cfg, pt_part = prepare_slab_search_surface_config(pt_cfg)
    pt_cut = layer_stacking_cutoff_from_template(
        pt_cfg.slab,
        int(pt_part.n_fixed),
        surface_normal_axis=pt_cfg.surface_normal_axis,
        connectivity_factor=pt_cfg.structure_connectivity_factor,
        use_mic=True,
    )
    assert pt_cut is None
