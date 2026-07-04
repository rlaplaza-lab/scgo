from ase import Atoms

from scgo.database.metadata import persist_provenance


def test_persist_provenance_writes_run_id():
    a = Atoms("Pt", positions=[[0, 0, 0]])
    persist_provenance(a, run_id="run_007")

    prov = a.info.get("provenance", {})
    assert prov.get("run_id") == "run_007"

    kv = a.info.get("key_value_pairs", {})
    assert kv.get("run_id") == "run_007"

    meta = a.info.get("metadata", {})
    assert meta.get("run_id") == "run_007"


def test_persist_provenance_no_run_id():
    a = Atoms("Pt", positions=[[0, 0, 0]])
    persist_provenance(a, run_id=None)

    assert "provenance" not in a.info or "run_id" not in a.info.get("provenance", {})
