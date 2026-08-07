"""Structure-tag encode/decode helpers."""

from __future__ import annotations

import numpy as np
from ase import Atoms

from scgo.metadata.atoms import get_tag, get_tags, set_tags


def test_set_tags_writes_run_id():
    a = Atoms("H", positions=[[0, 0, 0]])
    set_tags(a, run_id="run_007")

    assert get_tag(a, "run_id") == "run_007"
    assert a.info["key_value_pairs"]["run_id"] == "run_007"


def test_set_tags_without_run_id():
    a = Atoms("H", positions=[[0, 0, 0]])
    set_tags(a, generation=1)

    assert get_tag(a, "run_id") is None
    assert "run_id" not in a.info.get("key_value_pairs", {})


def test_set_tags_skips_none():
    a = Atoms("H", positions=[[0, 0, 0]])
    set_tags(a, run_id="keep", generation=None)
    assert get_tag(a, "run_id") == "keep"
    assert "generation" not in a.info["key_value_pairs"]


def test_ambiguous_pair_id_round_trips_via_get_tag():
    a = Atoms("Pt2", positions=[[0, 0, 0], [0, 0, 2]])
    set_tags(a, pair_id="0_1", ts_connects_minima="0_1")
    assert get_tag(a, "pair_id") == "0_1"
    assert get_tag(a, "ts_connects_minima") == "0_1"
    wire = a.info["key_value_pairs"]["pair_id"]
    assert wire != "0_1"
    assert wire.startswith("j:")


def test_list_and_dict_tags_round_trip():
    a = Atoms("H", positions=[[0, 0, 0]])
    set_tags(a, connects=[0, 1], payload={"a": 1})
    assert get_tag(a, "connects") == [0, 1]
    assert get_tag(a, "payload") == {"a": 1}
    assert get_tags(a)["connects"] == [0, 1]


def test_plain_string_not_heuristically_json_decoded():
    a = Atoms("H", positions=[[0, 0, 0]])
    set_tags(a, note='"hello"')
    assert get_tag(a, "note") == '"hello"'
    assert a.info["key_value_pairs"]["note"] == '"hello"'


def test_numpy_scalar_tags():
    a = Atoms("H", positions=[[0, 0, 0]])
    set_tags(a, generation=np.int64(3), score=np.float64(-1.5), flag=np.bool_(True))
    assert get_tag(a, "generation") == 3
    assert get_tag(a, "score") == -1.5
    assert get_tag(a, "flag") is True
