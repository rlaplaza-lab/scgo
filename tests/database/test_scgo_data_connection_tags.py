from ase import Atoms

from scgo.database import close_data_connection
from scgo.database.helpers import SCGODataConnection, setup_database
from scgo.metadata.atoms import set_tags
from tests.test_utils import assert_run_id_persisted


def test_db_adapter_persists_tags_in_key_value_pairs(tmp_path):
    """SCGODataConnection persists tags into ``key_value_pairs``."""

    outdir = tmp_path / "test_db_adapter"
    db = setup_database(
        output_dir=outdir,
        db_filename="test.db",
        atoms_template=Atoms("Pt2", positions=[[0, 0, 0], [0, 0, 1]]),
        remove_existing=True,
    )
    assert isinstance(db, SCGODataConnection)

    a = Atoms("Pt2", positions=[[0, 0, 0], [0, 0, 1]])
    # Single bag: tags plus user keys ASE should preserve
    a.info["key_value_pairs"] = {
        "run_id": "run_test_123",
        "generation": 1,
        "raw_score": -10.0,
        "user_note": "hello",
    }
    a.info["data"] = {}

    db.add_unrelaxed_candidate(a, description="test:alpha")

    u = db.get_an_unrelaxed_candidate()
    assert u is not None

    assert_run_id_persisted(u, "run_test_123")
    kv = u.info.get("key_value_pairs", {})
    assert kv.get("user_note") == "hello"
    assert "raw_score" in kv

    close_data_connection(db)


def test_unrelaxed_tags_persisted_for_cross_process_reads(tmp_path):
    """Tags on unrelaxed candidates must be visible to other DB readers."""
    outdir = tmp_path / "test_db_adapter_proc"
    db_file = outdir / "test.db"
    db = setup_database(
        output_dir=outdir,
        db_filename="test.db",
        atoms_template=Atoms("Pt2", positions=[[0, 0, 0], [0, 0, 1]]),
        remove_existing=True,
    )
    assert isinstance(db, SCGODataConnection)

    a = Atoms("Pt2", positions=[[0, 0, 0], [0, 0, 1]])
    set_tags(
        a,
        run_id="run_proc_42",
        trial_id=99,
        confid="c-1",
        raw_score=-5.0,
        user_note="keep-me",
    )
    a.info["data"] = {}
    db.add_unrelaxed_candidate(a, description="test:proc")

    from ase_ga.data import DataConnection

    da2 = DataConnection(str(db_file))
    u = da2.get_an_unrelaxed_candidate()
    assert u is not None
    assert_run_id_persisted(u, "run_proc_42")
    kv = u.info.get("key_value_pairs", {})
    assert kv.get("trial_id") == 99
    assert kv.get("confid") == "c-1"
    assert kv.get("user_note") == "keep-me"

    close_data_connection(db)
    close_data_connection(da2)
