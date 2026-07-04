import os

from ase import Atoms

from scgo.algorithms.ga_common import setup_diversity_scorer
from scgo.database import close_data_connection, setup_database
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
