import os

import numpy as np
from ase import Atoms
from ase.build import fcc111
from ase.ga.utilities import get_all_atom_types

from scgo.algorithms.ga_common import (
    setup_diversity_scorer,
    validate_structure_for_ga_storage,
)
from scgo.database import close_data_connection, setup_database
from scgo.initialization.atomic_radii import build_blmin_from_zs
from scgo.surface.config import SurfaceSystemConfig
from scgo.surface.deposition import create_deposited_cluster_batch
from scgo.system_types import validate_structure_for_system_type
from scgo.utils.logging import get_logger


def test_setup_diversity_scorer_uses_base_dir(tmp_path, rng):
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
            a.info.setdefault("metadata", {})["final_unique_minimum"] = True
            a.info.setdefault("key_value_pairs", {})["final_unique_minimum"] = True
        da.add_relaxed_step(a)

    close_data_connection(da)
    del da

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
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
    finally:
        os.chdir(old_cwd)


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
        3,
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
        except ValueError:
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
